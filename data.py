import csv
import hashlib
import io
import math
import os
import subprocess
import wave
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, Sampler, WeightedRandomSampler
from torch.utils.data._utils.collate import default_collate
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode, RandomResizedCrop


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

DEFAULT_VIDEO_AUGMENT = {
    "official_style": False,
    "crop_scale": (0.85, 1.0),
    "crop_ratio": (0.75, 1.3333333333333333),
    "hflip_prob": 0.5,
    "rotate_prob": 0.0,
    "rotate_limit": (-10.0, 10.0),
    "color_prob": 0.0,
    "brightness": 0.0,
    "contrast": 0.0,
    "saturation": 0.0,
    "hue": 0.0,
    "blur_prob": 0.0,
    "blur_limit": None,
    "blur_kernel_size": None,
    "blur_sigma": (0.1, 1.5),
    "noise_prob": 0.0,
    "noise_std": (0.0, 0.02),
    "jpeg_prob": 0.0,
    "jpeg_quality": (40, 100),
    "downsample_prob": 0.0,
    "downsample_scale": (0.5, 1.0),
    "temporal_reverse_prob": 0.0,
    "frame_dropout_prob": 0.0,
}


class BinaryRatioSampler(Sampler):
    def __init__(
        self,
        labels,
        real_parts=1,
        fake_parts=1,
        num_samples=None,
        replacement=True,
        seed=None,
    ):
        self.real_indices = torch.as_tensor(
            [idx for idx, label in enumerate(labels) if int(label) == 0],
            dtype=torch.long,
        )
        self.fake_indices = torch.as_tensor(
            [idx for idx, label in enumerate(labels) if int(label) == 1],
            dtype=torch.long,
        )
        if len(self.real_indices) == 0 or len(self.fake_indices) == 0:
            raise RuntimeError("Binary sampler requires both real and fake samples")

        real_parts = float(real_parts)
        fake_parts = float(fake_parts)
        if real_parts <= 0 or fake_parts <= 0:
            raise ValueError("real/fake ratio parts must be positive")

        self.num_samples = int(num_samples or len(labels))
        self.real_samples = int(round(self.num_samples * real_parts / (real_parts + fake_parts)))
        self.real_samples = max(1, min(self.num_samples - 1, self.real_samples))
        self.fake_samples = self.num_samples - self.real_samples
        self.replacement = bool(replacement)

        if not self.replacement:
            if self.real_samples > len(self.real_indices) or self.fake_samples > len(self.fake_indices):
                raise RuntimeError(
                    "Binary sampler without replacement cannot draw more samples "
                    "than a class contains"
                )

        self.generator = torch.Generator()
        if seed is None:
            self.generator.seed()
        else:
            self.generator.manual_seed(int(seed))

    def _draw(self, indices, count):
        if self.replacement and count > len(indices):
            return indices[torch.randint(len(indices), (count,), generator=self.generator)]
        order = torch.randperm(len(indices), generator=self.generator)[:count]
        return indices[order]

    def __iter__(self):
        real = self._draw(self.real_indices, self.real_samples)
        fake = self._draw(self.fake_indices, self.fake_samples)
        indices = torch.cat([real, fake])
        order = torch.randperm(len(indices), generator=self.generator)
        return iter(indices[order].tolist())

    def __len__(self):
        return self.num_samples


def _as_float_label(value, default=0.0):
    if value is None or value == "":
        return float(default)
    return float(value)


def _load_pcm_wav(wav_path):
    """Load an uncompressed PCM WAV file without torchaudio I/O backends.

    FFmpeg writes the cache as mono 16 kHz PCM, so Python's standard
    library is sufficient and avoids torchaudio backend availability issues.
    """
    wav_path = Path(wav_path)
    with wave.open(str(wav_path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        frame_count = wav_file.getnframes()
        raw = wav_file.readframes(frame_count)

    if channels < 1 or frame_count < 1:
        raise RuntimeError(f"empty WAV file: {wav_path}")

    if sample_width == 1:
        samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        samples = (samples - 128.0) / 128.0
    elif sample_width == 2:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32)
        samples /= 32768.0
    elif sample_width == 4:
        samples = np.frombuffer(raw, dtype="<i4").astype(np.float32)
        samples /= 2147483648.0
    else:
        raise RuntimeError(
            f"unsupported WAV sample width {sample_width} bytes: {wav_path}"
        )

    samples = samples.reshape(-1, channels)
    samples = samples.mean(axis=1, dtype=np.float32)
    waveform = torch.from_numpy(samples.copy()).unsqueeze(0)
    return waveform, int(sample_rate)


def _wav_is_readable(wav_path):
    try:
        wav_path = Path(wav_path)
        if not wav_path.is_file() or wav_path.stat().st_size <= 44:
            return False
        with wave.open(str(wav_path), "rb") as wav_file:
            return wav_file.getnframes() > 0 and wav_file.getnchannels() > 0
    except (OSError, EOFError, wave.Error):
        return False


class AudioVideoDataset(Dataset):
    """
    Dataset for processed AV deepfake CSV files.

    Required CSV columns:
        video_path, overall_label

    Optional CSV columns:
        frame_dir or face_crop_folder, audio_label, visual_label

    Returned tensors:
        audio: [target_length, mel_bins]
        video: [3, num_frames, image_size, image_size]
        labels: scalar float tensors
    """

    def __init__(
        self,
        csv_path,
        mode="train",
        num_frames=16,
        image_size=224,
        target_length=1024,
        mel_bins=128,
        audio_mean=-6.9960,
        audio_std=3.1205,
        audio_cache_dir=None,
        augment=False,
        video_augment=None,
        freq_mask_param=20,
        time_mask_param=80,
        supplemental_real_csv=None,
        supplemental_real_max_samples=None,
        supplemental_real_seed=42,
        supplemental_real_fixed=True,
        skip_bad_media=False,
        video_root=None,
    ):
        self.csv_path = Path(csv_path)
        self.mode = mode
        self.num_frames = num_frames
        self.image_size = image_size
        self.target_length = target_length
        self.mel_bins = mel_bins
        self.audio_mean = audio_mean
        self.audio_std = max(float(audio_std), 1e-6)
        self.augment = augment and mode == "train"
        self.video_augment = self._validate_video_augment(video_augment)
        self.freq_mask_param = max(0, int(freq_mask_param))
        self.time_mask_param = max(0, int(time_mask_param))
        self.audio_cache_dir = Path(audio_cache_dir) if audio_cache_dir else self.csv_path.parent / ".audio_cache"
        self.audio_cache_dir.mkdir(parents=True, exist_ok=True)
        self.skip_bad_media = bool(skip_bad_media)
        self.video_root = Path(video_root) if video_root else None

        with self.csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.rows = list(reader)
        if not self.rows:
            raise RuntimeError(f"No rows found in CSV: {self.csv_path}")

        self.primary_size = len(self.rows)
        self.supplemental_real_size = 0
        if supplemental_real_csv:
            supplemental_path = Path(supplemental_real_csv)
            if not supplemental_path.is_file():
                raise FileNotFoundError(
                    f"Supplemental real CSV not found: {supplemental_path}"
                )
            with supplemental_path.open("r", newline="", encoding="utf-8") as f:
                supplemental_rows = list(csv.DictReader(f))

            real_rows = []
            for row in supplemental_rows:
                overall = int(_as_float_label(row.get("overall_label"), default=0.0))
                audio = int(_as_float_label(row.get("audio_label"), default=overall))
                visual = int(_as_float_label(row.get("visual_label"), default=overall))
                if overall == 0 and audio == 0 and visual == 0:
                    real_rows.append(row)

            available_real = len(real_rows)
            limit = supplemental_real_max_samples
            if isinstance(limit, str):
                if limit.lower() != "auto":
                    raise ValueError(
                        "supplemental_real_max_samples must be an integer, None, "
                        "or 'auto'"
                    )
                primary_overall = [
                    int(_as_float_label(row.get("overall_label"), default=0.0))
                    for row in self.rows
                ]
                limit = max(
                    0,
                    primary_overall.count(1) - primary_overall.count(0),
                )
            elif limit is not None:
                limit = int(limit)
                if limit < 0:
                    raise ValueError(
                        "supplemental_real_max_samples must be non-negative"
                    )

            if limit is not None and len(real_rows) > limit:
                generator = torch.Generator()
                if supplemental_real_fixed:
                    generator.manual_seed(int(supplemental_real_seed))
                    sample_mode = f"fixed(seed={int(supplemental_real_seed)})"
                else:
                    generator.seed()
                    sample_mode = "random"
                indices = torch.randperm(len(real_rows), generator=generator)[:limit]
                real_rows = [real_rows[int(index)] for index in indices]
            else:
                sample_mode = "all"

            self.rows.extend(real_rows)
            self.supplemental_real_size = len(real_rows)
            print(
                "[INFO] supplemental real balance: "
                f"source={supplemental_path} available_real={available_real} "
                f"selected_real={self.supplemental_real_size} mode={sample_mode}"
            )

        # HAVIC-style four-combination labels:
        # 0 = real audio + real video
        # 1 = real audio + fake video
        # 2 = fake audio + real video
        # 3 = fake audio + fake video
        self.audio_labels = []
        self.visual_labels = []
        self.overall_labels = []
        self.comb_labels = []

        for row in self.rows:
            overall = int(_as_float_label(row.get("overall_label"), default=0.0))
            audio = int(_as_float_label(row.get("audio_label"), default=overall))
            visual = int(_as_float_label(row.get("visual_label"), default=overall))

            if overall not in (0, 1) or audio not in (0, 1) or visual not in (0, 1):
                raise ValueError(
                    "overall_label, audio_label and visual_label must be binary "
                    f"(0/1), but got overall={overall}, audio={audio}, "
                    f"visual={visual}"
                )

            self.audio_labels.append(audio)
            self.visual_labels.append(visual)
            self.overall_labels.append(overall)
            self.comb_labels.append(audio * 2 + visual)

    def __len__(self):
        return len(self.rows)

    def labels_distribution(self):
        counts = Counter(self.comb_labels)
        names = {
            0: "real-audio + real-video",
            1: "real-audio + fake-video",
            2: "fake-audio + real-video",
            3: "fake-audio + fake-video",
        }
        print("[INFO] training combination distribution:")
        for label in range(4):
            print(f"       class {label} ({names[label]}): {counts.get(label, 0)}")
        return counts

    def get_comb_weighted_sampler(self, seed=None):
        """Build HAVIC-style weighted sampling over the four AV combinations.

        Every observed combination receives the same total sampling probability.
        Sampling uses replacement and draws len(dataset) samples per epoch.
        """
        counts = self.labels_distribution()
        num_samples = len(self.comb_labels)

        class_weights = {
            label: num_samples / count
            for label, count in counts.items()
            if count > 0
        }
        sample_weights = torch.as_tensor(
            [class_weights[label] for label in self.comb_labels],
            dtype=torch.double,
        )

        generator = None
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(int(seed))

        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=num_samples,
            replacement=True,
            generator=generator,
        )

    def get_binary_weighted_sampler(self, seed=None):
        """Balance real/fake samples using the overall binary label."""
        counts = Counter(self.overall_labels)
        print(
            "[INFO] training binary distribution: "
            f"real={counts.get(0, 0)} fake={counts.get(1, 0)}"
        )
        if not counts.get(0) or not counts.get(1):
            raise RuntimeError(
                "Binary weighted sampling requires both real and fake samples"
            )

        num_samples = len(self.overall_labels)
        class_weights = {
            label: num_samples / count
            for label, count in counts.items()
        }
        sample_weights = torch.as_tensor(
            [class_weights[label] for label in self.overall_labels],
            dtype=torch.double,
        )
        generator = None
        if seed is not None:
            generator = torch.Generator().manual_seed(int(seed))
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=num_samples,
            replacement=True,
            generator=generator,
        )

    def get_binary_ratio_sampler(self, real_fake_ratio=(1, 1), seed=None):
        """Draw a fixed real/fake ratio with replacement when needed."""
        if not isinstance(real_fake_ratio, (list, tuple)) or len(real_fake_ratio) != 2:
            raise ValueError("real_fake_ratio must be a pair, e.g. (1, 3)")
        real_parts, fake_parts = real_fake_ratio
        sampler = BinaryRatioSampler(
            self.overall_labels,
            real_parts=real_parts,
            fake_parts=fake_parts,
            num_samples=len(self.overall_labels),
            replacement=True,
            seed=seed,
        )
        counts = Counter(self.overall_labels)
        print(
            "[INFO] training binary ratio sampler: "
            f"real={counts.get(0, 0)} fake={counts.get(1, 0)} "
            f"ratio={real_parts}:{fake_parts} "
            f"draw_real={sampler.real_samples} draw_fake={sampler.fake_samples}"
        )
        return sampler

    def get_binary_subset_sampler(self, samples_per_class=None, seed=None):
        """Draw a smaller epoch with equal real/fake samples and no replacement."""
        counts = Counter(self.overall_labels)
        if not counts.get(0) or not counts.get(1):
            raise RuntimeError(
                "Binary subset sampler requires both real and fake samples"
            )
        if samples_per_class is None:
            samples_per_class = min(counts[0], counts[1])
        samples_per_class = int(samples_per_class)
        sampler = BinaryRatioSampler(
            self.overall_labels,
            real_parts=1,
            fake_parts=1,
            num_samples=2 * samples_per_class,
            replacement=False,
            seed=seed,
        )
        print(
            "[INFO] training binary subset sampler: "
            f"real={counts.get(0, 0)} fake={counts.get(1, 0)} "
            f"draw_real={sampler.real_samples} draw_fake={sampler.fake_samples}"
        )
        return sampler

    def get_binary_subset_ratio_sampler(self, real_fake_ratio=(1, 3), seed=None):
        """Draw all real samples once and a ratio-controlled fake subset."""
        if not isinstance(real_fake_ratio, (list, tuple)) or len(real_fake_ratio) != 2:
            raise ValueError("real_fake_ratio must be a pair, e.g. (1, 3)")
        real_parts, fake_parts = real_fake_ratio
        real_parts = float(real_parts)
        fake_parts = float(fake_parts)
        if real_parts <= 0 or fake_parts <= 0:
            raise ValueError("real/fake ratio parts must be positive")

        counts = Counter(self.overall_labels)
        if not counts.get(0) or not counts.get(1):
            raise RuntimeError(
                "Binary subset ratio sampler requires both real and fake samples"
            )

        real_samples = counts[0]
        fake_samples = int(round(real_samples * fake_parts / real_parts))
        if fake_samples > counts[1]:
            raise RuntimeError(
                "Binary subset ratio sampler requested more fake samples than "
                f"available: requested={fake_samples}, available={counts[1]}"
            )

        sampler = BinaryRatioSampler(
            self.overall_labels,
            real_parts=real_samples,
            fake_parts=fake_samples,
            num_samples=real_samples + fake_samples,
            replacement=False,
            seed=seed,
        )
        print(
            "[INFO] training binary subset ratio sampler: "
            f"real={counts.get(0, 0)} fake={counts.get(1, 0)} "
            f"ratio={real_fake_ratio[0]}:{real_fake_ratio[1]} "
            f"draw_real={sampler.real_samples} draw_fake={sampler.fake_samples}"
        )
        return sampler

    def __getitem__(self, index):
        row = self.rows[index]
        video_path = row.get("video_path") or row.get("visual_path")
        if not video_path:
            if self.skip_bad_media:
                print("[WARN] skipped row without video_path/visual_path")
                return None
            raise KeyError("CSV must contain video_path or visual_path")

        resolved_video_path = self._resolve_video_path(video_path)
        frame_dir = row.get("frame_dir") or row.get("face_crop_folder") or ""
        try:
            audio = self._load_audio_fbank(
                resolved_video_path,
                strict=self.skip_bad_media,
            )
            video = self._load_video_tensor(
                resolved_video_path,
                frame_dir,
                strict=self.skip_bad_media,
            )
        except Exception as exc:
            if self.skip_bad_media:
                print(f"[WARN] skipped bad media sample {video_path}: {exc}")
                return None
            raise

        overall = _as_float_label(row.get("overall_label"), default=0.0)
        audio_label = _as_float_label(row.get("audio_label"), default=overall)
        visual_label = _as_float_label(row.get("visual_label"), default=overall)

        return {
            "audio": audio,
            "video": video,
            "label": torch.tensor(overall, dtype=torch.float32),
            "audio_label": torch.tensor(audio_label, dtype=torch.float32),
            "visual_label": torch.tensor(visual_label, dtype=torch.float32),
            "video_path": video_path,
            "resolved_video_path": resolved_video_path,
        }

    def _resolve_video_path(self, video_path):
        path = Path(video_path)
        if path.is_file():
            return str(path)
        if self.video_root is not None and not path.is_absolute():
            candidate = self.video_root / path
            if candidate.is_file():
                return str(candidate)
        return str(path)

    def _load_audio_fbank(self, video_path, strict=False):
        try:
            # Keep WAV extraction inside the same try block so a video with
            # no usable audio stream does not terminate a DataLoader worker.
            audio_path = self._ensure_wav(video_path)
            waveform, sr = _load_pcm_wav(audio_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
                sr = 16000
            waveform = waveform - waveform.mean()
            fbank = torchaudio.compliance.kaldi.fbank(
                waveform,
                htk_compat=True,
                sample_frequency=sr,
                use_energy=False,
                window_type="hanning",
                num_mel_bins=self.mel_bins,
                dither=0.0,
                frame_shift=10,
            )
        except Exception as exc:
            if strict:
                raise RuntimeError(
                    f"failed to extract/load audio for {video_path}: {exc}"
                ) from exc
            fbank = torch.zeros(self.target_length, self.mel_bins) + 0.01

        fbank = F.interpolate(
            fbank.unsqueeze(0).transpose(1, 2),
            size=self.target_length,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2).squeeze(0)

        fbank = (fbank - self.audio_mean) / self.audio_std
        if self.augment:
            if self.freq_mask_param > 0 and torch.rand(()) < 0.5:
                fbank = torchaudio.transforms.FrequencyMasking(
                    freq_mask_param=self.freq_mask_param
                )(fbank.T.unsqueeze(0)).squeeze(0).T
            if self.time_mask_param > 0 and torch.rand(()) < 0.5:
                fbank = torchaudio.transforms.TimeMasking(
                    time_mask_param=self.time_mask_param
                )(fbank.T.unsqueeze(0)).squeeze(0).T
        return fbank.float()

    def _ensure_wav(self, video_path):
        path = Path(video_path)
        if path.suffix.lower() == ".wav":
            return str(path)
        digest = hashlib.md5(str(path).encode("utf-8")).hexdigest()
        wav_path = self.audio_cache_dir / f"{path.stem}_{digest}.wav"

        if _wav_is_readable(wav_path):
            return str(wav_path)

        # Remove an empty/truncated cache file left by an interrupted run.
        if wav_path.exists():
            wav_path.unlink()

        temp_path = wav_path.with_suffix(".tmp.wav")
        if temp_path.exists():
            temp_path.unlink()

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(temp_path),
        ]
        try:
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            if not _wav_is_readable(temp_path):
                raise RuntimeError(f"FFmpeg created an invalid WAV file: {temp_path}")
            os.replace(temp_path, wav_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

        return str(wav_path)

    def _load_video_tensor(self, video_path, frame_dir, strict=False):
        if frame_dir and Path(frame_dir).exists():
            frames = self._read_frames_from_dir(frame_dir)
        else:
            frames = self._sample_frames_from_video(video_path, strict=strict)
        frames = self._transform_frames(frames)
        return torch.stack(frames, dim=1).float()

    @staticmethod
    def _validate_video_augment(config):
        supplied = dict(config or {})
        unknown = set(supplied).difference(DEFAULT_VIDEO_AUGMENT)
        if unknown:
            raise ValueError(f"Unknown video_augment options: {sorted(unknown)}")
        result = {**DEFAULT_VIDEO_AUGMENT, **supplied}
        result["official_style"] = bool(result["official_style"])

        for name in (
            "hflip_prob",
            "rotate_prob",
            "color_prob",
            "blur_prob",
            "noise_prob",
            "jpeg_prob",
            "downsample_prob",
            "temporal_reverse_prob",
            "frame_dropout_prob",
        ):
            value = float(result[name])
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"video_augment.{name} must be in [0,1]")
            result[name] = value

        for name in ("brightness", "contrast", "saturation", "hue"):
            value = float(result[name])
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"video_augment.{name} must be in [0,1]")
            result[name] = value

        def pair(name, minimum=None, maximum=None, integer=False, optional=False):
            value = result[name]
            if optional and value is None:
                return
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError(f"video_augment.{name} must contain [low, high]")
            low, high = (int(v) for v in value) if integer else (float(v) for v in value)
            if low > high:
                raise ValueError(f"video_augment.{name} must satisfy low <= high")
            if minimum is not None and low < minimum:
                raise ValueError(f"video_augment.{name} must be >= {minimum}")
            if maximum is not None and high > maximum:
                raise ValueError(f"video_augment.{name} must be <= {maximum}")
            result[name] = (low, high)

        pair("crop_scale", minimum=1e-6, maximum=1.0)
        pair("crop_ratio", minimum=1e-6)
        pair("rotate_limit")
        pair("blur_limit", minimum=1, integer=True, optional=True)
        pair("blur_sigma", minimum=0.0)
        pair("noise_std", minimum=0.0)
        pair("jpeg_quality", minimum=1, maximum=100, integer=True)
        pair("downsample_scale", minimum=1e-6, maximum=1.0)
        if result["blur_kernel_size"] is not None:
            kernel_size = int(result["blur_kernel_size"])
            if kernel_size < 1 or kernel_size % 2 == 0:
                raise ValueError(
                    "video_augment.blur_kernel_size must be a positive odd integer"
                )
            result["blur_kernel_size"] = kernel_size
        return result

    @staticmethod
    def _sample_odd_int(bounds):
        low, high = (int(bounds[0]), int(bounds[1]))
        low = low if low % 2 else low + 1
        high = high if high % 2 else high - 1
        if low > high:
            raise ValueError("video_augment.blur_limit must include an odd value")
        count = ((high - low) // 2) + 1
        return int(low + 2 * int(torch.randint(count, (), dtype=torch.long)))

    @staticmethod
    def _uniform(bounds):
        low, high = bounds
        if low == high:
            return float(low)
        return float(torch.empty(()).uniform_(float(low), float(high)))

    def _sample_video_augment(self):
        cfg = self.video_augment
        enabled = lambda name: bool(torch.rand(()) < cfg[name])
        color_enabled = enabled("color_prob")
        return {
            "crop_scale": self._uniform(cfg["crop_scale"]),
            "crop_x": 0.5,
            "crop_y": 0.5,
            "flip": enabled("hflip_prob"),
            "rotate": enabled("rotate_prob"),
            "rotate_angle": self._uniform(cfg["rotate_limit"]),
            "color": color_enabled,
            "brightness": self._uniform(
                (1.0 - cfg["brightness"], 1.0 + cfg["brightness"])
            ),
            "contrast": self._uniform(
                (1.0 - cfg["contrast"], 1.0 + cfg["contrast"])
            ),
            "saturation": self._uniform(
                (1.0 - cfg["saturation"], 1.0 + cfg["saturation"])
            ),
            "color_order": torch.randperm(3).tolist(),
            "blur": enabled("blur_prob"),
            "blur_kernel": (
                self._sample_odd_int(cfg["blur_limit"])
                if cfg["blur_limit"] is not None
                else None
            ),
            "blur_sigma": self._uniform(cfg["blur_sigma"]),
            "downsample": enabled("downsample_prob"),
            "downsample_scale": self._uniform(cfg["downsample_scale"]),
            "jpeg": enabled("jpeg_prob"),
            "jpeg_quality": int(
                torch.randint(
                    cfg["jpeg_quality"][0],
                    cfg["jpeg_quality"][1] + 1,
                    (),
                )
            ),
            "noise": enabled("noise_prob"),
            "noise_std": self._uniform(cfg["noise_std"]),
        }

    @staticmethod
    def _jpeg_roundtrip(frame, quality):
        buffer = io.BytesIO()
        try:
            frame.save(
                buffer,
                format="JPEG",
                quality=int(quality),
                subsampling=2,
            )
            buffer.seek(0)
            with Image.open(buffer) as decoded:
                return decoded.convert("RGB").copy()
        finally:
            buffer.close()

    def _read_frames_from_dir(self, frame_dir):
        frame_dir = Path(frame_dir)
        frame_paths = sorted(
            p for p in frame_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        )
        if not frame_paths:
            raise RuntimeError(f"No frames found in {frame_dir}")
        indices = np.linspace(0, len(frame_paths) - 1, self.num_frames).astype(int)
        return [Image.open(frame_paths[i]).convert("RGB") for i in indices]

    def _sample_frames_from_video(self, video_path, strict=False):
        try:
            from decord import VideoReader, cpu
            vr = VideoReader(video_path, ctx=cpu(0))
            if len(vr) == 0:
                raise RuntimeError("empty video")
            indices = np.linspace(0, len(vr) - 1, self.num_frames).astype(int)
            batch = vr.get_batch(indices).asnumpy()
            return [Image.fromarray(frame).convert("RGB") for frame in batch]
        except Exception as exc:
            if strict:
                raise RuntimeError(
                    f"failed to sample video frames for {video_path}: {exc}"
                ) from exc
            print(f"[WARN] failed to sample video frames for {video_path}: {exc}")
            blank = Image.new("RGB", (self.image_size, self.image_size), color=(0, 0, 0))
            return [blank.copy() for _ in range(self.num_frames)]

    def _transform_frames(self, frames):
        if self.augment and self.video_augment["official_style"]:
            return self._transform_frames_official(frames)

        params = self._sample_video_augment() if self.augment else None
        if params is not None and self.video_augment["temporal_reverse_prob"] > 0.0:
            if torch.rand(()) < self.video_augment["temporal_reverse_prob"]:
                frames = list(reversed(frames))
        if params is not None and self.video_augment["frame_dropout_prob"] > 0.0:
            frames = list(frames)
            drop = torch.rand(len(frames)) < self.video_augment["frame_dropout_prob"]
            for index in range(1, len(frames)):
                if bool(drop[index]):
                    frames[index] = frames[index - 1].copy()
        noise = None
        if params is not None and params["noise"] and params["noise_std"] > 0.0:
            # One residual map per clip avoids synthetic frame-to-frame flicker.
            noise = torch.randn(3, self.image_size, self.image_size)
            noise = noise * params["noise_std"]

        out = []
        for frame in frames:
            if params is not None:
                crop_size = max(
                    1,
                    int(round(min(frame.size) * params["crop_scale"])),
                )
                left = int(round(params["crop_x"] * max(0, frame.width - crop_size)))
                top = int(round(params["crop_y"] * max(0, frame.height - crop_size)))
                frame = TF.crop(frame, top, left, crop_size, crop_size)
                if params["flip"]:
                    frame = TF.hflip(frame)
                if params["rotate"]:
                    frame = TF.rotate(
                        frame,
                        params["rotate_angle"],
                        interpolation=InterpolationMode.BILINEAR,
                        fill=0,
                    )
            frame = TF.resize(
                frame,
                [self.image_size, self.image_size],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )

            if params is not None and params["color"]:
                color_ops = (
                    lambda image: TF.adjust_brightness(image, params["brightness"]),
                    lambda image: TF.adjust_contrast(image, params["contrast"]),
                    lambda image: TF.adjust_saturation(image, params["saturation"]),
                )
                for index in params["color_order"]:
                    frame = color_ops[index](frame)

            if params is not None and params["blur"]:
                if params["blur_kernel"] is not None:
                    radius = max(0.0, (params["blur_kernel"] - 1) / 2.0)
                    frame = frame.filter(ImageFilter.BoxBlur(radius=radius))
                elif params["blur_sigma"] > 0.0:
                    kernel = max(3, 2 * math.ceil(3 * params["blur_sigma"]) + 1)
                    frame = TF.gaussian_blur(
                        frame,
                        [kernel, kernel],
                        [params["blur_sigma"], params["blur_sigma"]],
                    )

            if params is not None and params["downsample"]:
                side = max(1, int(round(self.image_size * params["downsample_scale"])))
                frame = TF.resize(
                    frame,
                    [side, side],
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,
                )
                frame = TF.resize(
                    frame,
                    [self.image_size, self.image_size],
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,
                )

            if params is not None and params["jpeg"]:
                frame = self._jpeg_roundtrip(frame, params["jpeg_quality"])

            tensor = TF.to_tensor(frame)
            if noise is not None:
                tensor = (tensor + noise).clamp_(0.0, 1.0)
            out.append(tensor)
        return out

    def _transform_frames_official(self, frames):
        """Match the visual augmentation order used by the official HAVIC code."""
        cfg = self.video_augment
        frames = [
            TF.resize(
                frame,
                [self.image_size, self.image_size],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            for frame in frames
        ]
        top, left, height, width = RandomResizedCrop.get_params(
            frames[0],
            scale=cfg["crop_scale"],
            ratio=cfg["crop_ratio"],
        )
        flip = bool(torch.rand(()) < cfg["hflip_prob"])
        brightness = self._uniform((1.0 - cfg["brightness"], 1.0 + cfg["brightness"]))
        contrast = self._uniform((1.0 - cfg["contrast"], 1.0 + cfg["contrast"]))
        saturation = self._uniform((1.0 - cfg["saturation"], 1.0 + cfg["saturation"]))
        hue = self._uniform((-cfg["hue"], cfg["hue"]))
        apply_color = bool(torch.rand(()) < cfg["color_prob"])
        apply_blur = bool(torch.rand(()) < cfg["blur_prob"])
        blur_sigma = self._uniform(cfg["blur_sigma"])
        apply_jpeg = bool(torch.rand(()) < cfg["jpeg_prob"])
        jpeg_quality = int(
            torch.randint(cfg["jpeg_quality"][0], cfg["jpeg_quality"][1] + 1, ())
        )
        apply_downsample = bool(torch.rand(()) < cfg["downsample_prob"])
        downsample_scale = self._uniform(cfg["downsample_scale"])

        out = []
        for frame in frames:
            if flip:
                frame = TF.hflip(frame)
            frame = TF.resized_crop(
                frame,
                top,
                left,
                height,
                width,
                [self.image_size, self.image_size],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            if apply_downsample:
                side = max(1, int(self.image_size * downsample_scale))
                frame = TF.resize(frame, [side, side])
                frame = TF.resize(frame, [self.image_size, self.image_size])
            if apply_jpeg:
                frame = self._jpeg_roundtrip(frame, jpeg_quality)
            if apply_blur:
                kernel = cfg["blur_kernel_size"] or 13
                frame = TF.gaussian_blur(frame, [kernel, kernel], [blur_sigma, blur_sigma])
            if apply_color:
                frame = TF.adjust_brightness(frame, brightness)
                frame = TF.adjust_contrast(frame, contrast)
                frame = TF.adjust_saturation(frame, saturation)
                frame = TF.adjust_hue(frame, hue)
            out.append(TF.to_tensor(frame))
        return out


def skip_none_collate(batch):
    """Drop dataset items returned as None and report how many were skipped."""

    skipped = sum(item is None for item in batch)
    kept = [item for item in batch if item is not None]
    if not kept:
        return None
    collated = default_collate(kept)
    collated["_skipped_samples"] = skipped
    return collated
