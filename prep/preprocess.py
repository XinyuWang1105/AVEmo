import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

import torch

if torch.cuda.is_available():
    torch.cuda.init()

    # Force CUDA context creation before Decord is loaded.
    _cuda_init_tensor = torch.empty(
        1,
        device="cuda:0",
    )

import cv2
import numpy as np
import yaml

# Decord MUST be imported after CUDA initialization.
from decord import VideoReader, cpu

from PIL import Image
from tqdm import tqdm

# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FACEX_ROOT = PROJECT_ROOT / "FaceX_Zoo"
FACEX_SDK_DIR = FACEX_ROOT / "face_sdk"
FACEX_NET_DIR = FACEX_SDK_DIR / "models" / "network_def"

# FaceX-Zoo checkpoints reference face_sdk module paths.
sys.path.insert(0, str(FACEX_SDK_DIR))
sys.path.insert(1, str(PROJECT_ROOT))
sys.path.insert(2, str(FACEX_NET_DIR))


from FaceX_Zoo.face_sdk.core.model_handler.face_detection.FaceDetModelHandler import (
    FaceDetModelHandler,
)
from FaceX_Zoo.face_sdk.core.model_loader.face_detection.FaceDetModelLoader import (
    FaceDetModelLoader,
)


# ============================================================
# Output CSV fields
# ============================================================

OUTPUT_FIELDS = [
    "video_path",
    "face_crop_folder",
    "audio_label",
    "visual_label",
    "overall_label",
]


# ============================================================
# Basic utilities
# ============================================================

def is_valid_file(path):
    path = Path(path)

    return (
        path.is_file()
        and path.stat().st_size > 0
    )


def is_valid_video(path):
    path = Path(path)

    if not is_valid_file(path):
        return False

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "csv=p=0",
        str(path),
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    return (
        result.returncode == 0
        and "video" in result.stdout
    )


def expected_files(
    output_dir,
    prefix,
    num_frames,
):
    output_dir = Path(output_dir)

    return [
        output_dir / f"{prefix}_{idx:02d}.png"
        for idx in range(num_frames)
    ]


def files_are_complete(paths):
    return all(
        is_valid_file(path)
        for path in paths
    )


def safe_name(
    path,
    fallback,
):
    stem = Path(path).stem or fallback

    stem = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        stem,
    )

    return stem[:120]


# ============================================================
# FaceX-Zoo
# ============================================================

def load_face_detection_model(
    device="cuda:0",
):
    config_path = (
        FACEX_SDK_DIR
        / "config"
        / "model_conf.yaml"
    )

    model_path = (
        FACEX_SDK_DIR
        / "models"
    )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as f:

        model_conf = yaml.load(
            f,
            Loader=yaml.SafeLoader,
        )

    scene = "non-mask"
    model_category = "face_detection"

    model_name = model_conf[
        scene
    ][
        model_category
    ]

    loader = FaceDetModelLoader(
        str(model_path),
        model_category,
        model_name,
        meta_file="model_meta.json",
    )

    model, cfg = loader.load_model()

    return FaceDetModelHandler(
        model,
        device,
        cfg,
    )


# ============================================================
# Face cropper
# ============================================================

class FaceCropper:

    def __init__(
        self,
        handler,
        scale=1.3,
    ):
        self.handler = handler
        self.scale = scale


    def detect_box(
        self,
        frame_bgr,
        frame_name="<array>",
    ):
        """
        Detect face on a BGR numpy image.

        This is exactly what the old pipeline effectively did:
        RGB PNG -> cv2.imread -> BGR -> FaceX-Zoo.
        """

        dets = self.handler.inference_on_image(
            frame_bgr
        )

        if len(dets) == 0:
            raise RuntimeError(
                f"no face detected: {frame_name}"
            )

        x_min, y_min, x_max, y_max = map(
            int,
            dets[0][:4],
        )

        h, w = frame_bgr.shape[:2]

        cx = (
            x_min + x_max
        ) / 2.0

        cy = (
            y_min + y_max
        ) / 2.0

        bw = (
            x_max - x_min
        ) * self.scale

        bh = (
            y_max - y_min
        ) * self.scale

        x_min = max(
            int(cx - bw / 2),
            0,
        )

        x_max = min(
            int(cx + bw / 2),
            w,
        )

        y_min = max(
            int(cy - bh / 2),
            0,
        )

        y_max = min(
            int(cy + bh / 2),
            h,
        )

        return (
            x_min,
            y_min,
            x_max,
            y_max,
        )


    def crop_array_with_box(
        self,
        frame_bgr,
        save_path,
        box,
        frame_name="<array>",
    ):
        """
        Crop a face directly from an in-memory BGR numpy image.
        """

        if frame_bgr is None:
            raise RuntimeError(
                f"invalid frame: {frame_name}"
            )

        h, w = frame_bgr.shape[:2]

        (
            x_min,
            y_min,
            x_max,
            y_max,
        ) = box

        x_min = max(
            min(
                int(x_min),
                w - 1,
            ),
            0,
        )

        x_max = max(
            min(
                int(x_max),
                w,
            ),
            x_min + 1,
        )

        y_min = max(
            min(
                int(y_min),
                h - 1,
            ),
            0,
        )

        y_max = max(
            min(
                int(y_max),
                h,
            ),
            y_min + 1,
        )

        face = frame_bgr[
            y_min:y_max,
            x_min:x_max,
        ]

        if face.size == 0:
            raise RuntimeError(
                f"empty face crop: {frame_name}"
            )

        save_path = Path(save_path)

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        tmp_path = save_path.with_suffix(
            save_path.suffix + ".tmp"
        )

        # Convert BGR -> RGB before saving,
        # exactly matching the previous result.
        face_rgb = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB,
        )

        Image.fromarray(
            face_rgb
        ).save(
            tmp_path,
            format="PNG",
        )

        tmp_path.replace(
            save_path
        )


    def crop_from_rgb_array(
        self,
        frame_rgb,
        save_path,
        frame_name="<array>",
    ):
        """
        Decord gives RGB images.

        The old pipeline saved RGB PNG and then used cv2.imread(),
        which converted it to BGR.

        We explicitly do RGB -> BGR here so FaceX-Zoo receives
        the same image format as before.
        """

        if frame_rgb is None:
            raise RuntimeError(
                f"invalid frame: {frame_name}"
            )

        frame_bgr = cv2.cvtColor(
            frame_rgb,
            cv2.COLOR_RGB2BGR,
        )

        box = self.detect_box(
            frame_bgr,
            frame_name,
        )

        self.crop_array_with_box(
            frame_bgr,
            save_path,
            box,
            frame_name,
        )


# ============================================================
# Video segmentation
# ============================================================

def probe_duration(
    video_path,
):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )

    return float(
        result.stdout.strip()
    )


def write_segment(
    video_path,
    out_path,
    start,
    duration,
    segment_codec="copy",
):
    out_path = Path(out_path)

    tmp_path = out_path.with_name(
        f"{out_path.stem}.tmp{out_path.suffix}"
    )

    if tmp_path.exists():
        tmp_path.unlink()

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
    ]

    if segment_codec == "copy":

        cmd.extend(
            [
                "-c",
                "copy",
                "-avoid_negative_ts",
                "make_zero",
            ]
        )

    else:

        cmd.extend(
            [
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
            ]
        )

    cmd.extend(
        [
            "-movflags",
            "+faststart",
            str(tmp_path),
        ]
    )

    subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )

    if not is_valid_video(
        tmp_path
    ):
        raise RuntimeError(
            f"FFmpeg created an invalid segment: "
            f"{tmp_path}"
        )

    tmp_path.replace(
        out_path
    )


def split_video(
    video_path,
    output_dir,
    segment_seconds=3.2,
    keep_short=False,
    segment_codec="copy",
    max_segments_per_video=None,
):
    """
    Split video into fixed-length clips.

    max_segments_per_video:
        None -> original behavior, use all clips
        1    -> only use the first clip
        N    -> use at most N clips
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    video_path = Path(
        video_path
    )

    duration = probe_duration(
        video_path
    )

    if (
        duration <= segment_seconds
        and not keep_short
    ):
        return []

    if duration < segment_seconds:

        num_segments = 1

    else:

        num_segments = int(
            duration // segment_seconds
        )

    if max_segments_per_video is not None:

        if max_segments_per_video <= 0:
            raise ValueError(
                "max_segments_per_video "
                "must be greater than 0"
            )

        num_segments = min(
            num_segments,
            max_segments_per_video,
        )

    outputs = []

    for idx in range(
        num_segments
    ):

        start = (
            idx * segment_seconds
        )

        end = min(
            start + segment_seconds,
            duration,
        )

        part_duration = (
            end - start
        )

        if part_duration <= 0:
            continue

        out_path = (
            output_dir
            / (
                f"{safe_name(video_path, 'video')}"
                f"_part_{idx + 1:04d}.mp4"
            )
        )

        if (
            out_path.exists()
            and not is_valid_video(out_path)
        ):
            out_path.unlink()

        if not out_path.exists():

            try:

                write_segment(
                    video_path,
                    out_path,
                    start,
                    part_duration,
                    segment_codec=segment_codec,
                )

            except Exception:

                if segment_codec != "copy":
                    raise

                # Re-encode clips that cannot be copied cleanly.
                write_segment(
                    video_path,
                    out_path,
                    start,
                    part_duration,
                    segment_codec="libx264",
                )

        outputs.append(
            out_path
        )

    return outputs


# ============================================================
# Frame sampling and face cropping
# ============================================================

def sample_and_crop_faces(
    video_path,
    output_dir,
    cropper,
    num_frames=16,
    detect_once=False,
):
    """Decode selected frames and crop faces in memory."""

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_paths = expected_files(
        output_dir,
        prefix="face",
        num_frames=num_frames,
    )

    # Process only missing face crops.
    missing_indices = [
        idx
        for idx, path in enumerate(
            out_paths
        )
        if not is_valid_file(path)
    ]

    if not missing_indices:
        return output_dir

    vr = VideoReader(
        str(video_path),
        ctx=cpu(0),
    )

    if len(vr) < num_frames:
        raise RuntimeError(
            f"not enough frames: {video_path}"
        )

    source_indices = np.linspace(
        0,
        len(vr) - 1,
        num_frames,
        dtype=int,
    )

    frames_rgb = vr.get_batch(
        source_indices
    ).asnumpy()


    if (
        detect_once
        and missing_indices
    ):

        probe_indices = list(
            dict.fromkeys(
                [
                    num_frames // 2,
                    0,
                    num_frames - 1,
                ]
                + missing_indices
            )
        )

        box = None

        for idx in probe_indices:

            frame_bgr = cv2.cvtColor(
                frames_rgb[idx],
                cv2.COLOR_RGB2BGR,
            )

            frame_name = (
                f"{video_path}"
                f"[frame={source_indices[idx]}]"
            )

            try:

                box = cropper.detect_box(
                    frame_bgr,
                    frame_name,
                )

                break

            except Exception:

                continue


        if box is not None:

            for idx in missing_indices:

                frame_bgr = cv2.cvtColor(
                    frames_rgb[idx],
                    cv2.COLOR_RGB2BGR,
                )

                frame_name = (
                    f"{video_path}"
                    f"[frame={source_indices[idx]}]"
                )

                cropper.crop_array_with_box(
                    frame_bgr,
                    out_paths[idx],
                    box,
                    frame_name,
                )

            return output_dir


    for idx in missing_indices:

        frame_name = (
            f"{video_path}"
            f"[frame={source_indices[idx]}]"
        )

        cropper.crop_from_rgb_array(
            frames_rgb[idx],
            out_paths[idx],
            frame_name,
        )

    return output_dir


# ============================================================
# Resume CSV
# ============================================================

def load_existing_manifest(
    output_csv,
):
    completed_keys = set()

    if not output_csv.exists():
        return completed_keys

    if output_csv.stat().st_size == 0:
        return completed_keys

    with output_csv.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as f:

        reader = csv.DictReader(
            f
        )

        for row in reader:

            video_path = row.get(
                "video_path"
            )

            face_folder = row.get(
                "face_crop_folder"
            )

            if (
                not video_path
                or not face_folder
            ):
                continue

            completed_keys.add(
                (
                    video_path,
                    face_folder,
                )
            )

    return completed_keys


# ============================================================
# Main HAVIC-style preprocessing
# ============================================================

def preprocess_havic_style(
    input_csv,
    output_csv,
    work_dir,
    segment_seconds=3.2,
    num_frames=16,
    face_scale=1.3,
    keep_short=False,
    segment_codec="copy",
    detect_once_per_segment=False,
    face_device="cuda:0",
    max_segments_per_video=None,
):
    input_csv = Path(
        input_csv
    )

    output_csv = Path(
        output_csv
    )

    work_dir = Path(
        work_dir
    )

    segment_dir = (
        work_dir
        / "segments"
    )

    face_dir = (
        work_dir
        / "face_crops"
    )


    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    work_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    with input_csv.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as f:

        rows = list(
            csv.DictReader(f)
        )


    completed_keys = load_existing_manifest(
        output_csv
    )


    manifest_exists = (
        output_csv.exists()
        and output_csv.stat().st_size > 0
    )


    manifest_file = output_csv.open(
        "a",
        newline="",
        encoding="utf-8",
    )


    writer = csv.DictWriter(
        manifest_file,
        fieldnames=OUTPUT_FIELDS,
    )


    if not manifest_exists:

        writer.writeheader()
        manifest_file.flush()

    # Load FaceX-Zoo only when a clip needs processing.
    cropper = None


    added = 0
    reused = 0


    try:

        for row_idx, row in enumerate(
            tqdm(
                rows,
                desc="HAVIC-style preprocess",
            )
        ):

            video_path = (
                row.get("video_path")
                or row.get("visual_path")
            )


            if not video_path:

                print(
                    f"[WARN] row {row_idx} "
                    f"has no video_path, skip"
                )

                continue


            video_path = Path(
                video_path
            )


            if not video_path.exists():

                print(
                    f"[WARN] missing video: "
                    f"{video_path}"
                )

                continue


            overall = row.get(
                "overall_label",
                row.get(
                    "label",
                    "0",
                ),
            )


            audio_label = row.get(
                "audio_label",
                overall,
            )


            visual_label = row.get(
                "visual_label",
                overall,
            )


            sample_prefix = (
                f"{row_idx:07d}_"
                f"{safe_name(video_path, 'video')}"
            )


            # =================================================
            # 1. Split source video
            # =================================================

            try:

                segments = split_video(
                    video_path,
                    segment_dir / sample_prefix,
                    segment_seconds=segment_seconds,
                    keep_short=keep_short,
                    segment_codec=segment_codec,
                    max_segments_per_video=(
                        max_segments_per_video
                    ),
                )


            except Exception as exc:

                print(
                    f"[WARN] split failed for "
                    f"{video_path}: {exc}"
                )

                continue


            # =================================================
            # 2. Process every selected segment
            # =================================================

            for (
                seg_idx,
                segment_path,
            ) in enumerate(
                segments
            ):

                seg_name = (
                    f"{sample_prefix}_part_"
                    f"{seg_idx + 1:04d}"
                )


                face_folder_path = (
                    face_dir
                    / seg_name
                )


                manifest_key = (
                    str(segment_path),
                    str(face_folder_path),
                )


                expected_faces = expected_files(
                    face_folder_path,
                    prefix="face",
                    num_frames=num_frames,
                )


                if (
                    manifest_key in completed_keys
                    and files_are_complete(
                        expected_faces
                    )
                ):

                    reused += 1
                    continue


                # =============================================
                # 3. Decode frames + FaceX directly in memory
                # =============================================

                try:

                    if not files_are_complete(
                        expected_faces
                    ):

                        if cropper is None:

                            print(
                                "[INFO] loading "
                                "FaceX-Zoo face detector"
                            )

                            cropper = FaceCropper(
                                load_face_detection_model(
                                    face_device
                                ),
                                scale=face_scale,
                            )


                        face_folder = (
                            sample_and_crop_faces(
                                segment_path,
                                face_folder_path,
                                cropper,
                                num_frames=num_frames,
                                detect_once=(
                                    detect_once_per_segment
                                ),
                            )
                        )


                    else:

                        face_folder = (
                            face_folder_path
                        )


                except Exception as exc:

                    print(
                        "[WARN] frame/face preprocess "
                        f"failed for {segment_path}: "
                        f"{exc}"
                    )

                    continue


                # =============================================
                # 4. Write manifest row
                # =============================================

                output_row = {
                    "video_path": str(
                        segment_path
                    ),
                    "face_crop_folder": str(
                        face_folder
                    ),
                    "audio_label": (
                        audio_label
                    ),
                    "visual_label": (
                        visual_label
                    ),
                    "overall_label": (
                        overall
                    ),
                }


                if (
                    manifest_key
                    not in completed_keys
                ):

                    writer.writerow(
                        output_row
                    )

                    manifest_file.flush()

                    completed_keys.add(
                        manifest_key
                    )

                    added += 1


                else:

                    reused += 1


    finally:

        manifest_file.close()


    print(
        "[OK] resume preprocess finished: "
        f"total={len(completed_keys)}, "
        f"added={added}, "
        f"reused={reused}, "
        f"output={output_csv}"
    )


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "HAVIC-style preprocessing: "
            "split video, uniformly sample frames, "
            "FaceX-Zoo face crop."
        )
    )


    parser.add_argument(
        "--input_csv",
        required=True,
    )


    parser.add_argument(
        "--output_csv",
        required=True,
    )


    parser.add_argument(
        "--work_dir",
        required=True,
    )


    parser.add_argument(
        "--segment_seconds",
        type=float,
        default=3.2,
    )


    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
    )


    parser.add_argument(
        "--face_scale",
        type=float,
        default=1.3,
    )


    parser.add_argument(
        "--face_device",
        default="cuda:0",
    )


    parser.add_argument(
        "--keep_short",
        action="store_true",
        help=(
            "Keep videos shorter than "
            "segment_seconds."
        ),
    )


    parser.add_argument(
        "--segment_codec",
        choices=(
            "copy",
            "libx264",
        ),
        default="copy",
        help=(
            "Use stream copy for fast segmentation, "
            "or libx264 for safer re-encoding."
        ),
    )


    parser.add_argument(
        "--detect_once_per_segment",
        action="store_true",
        help=(
            "Detect one face box per segment "
            "and reuse it for all sampled frames."
        ),
    )


    parser.add_argument(
        "--max_segments_per_video",
        type=int,
        default=None,
        help=(
            "Maximum number of clips generated "
            "from each source video. "
            "Default: process all clips."
        ),
    )


    return parser.parse_args()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    args = parse_args()


    preprocess_havic_style(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        work_dir=args.work_dir,
        segment_seconds=args.segment_seconds,
        num_frames=args.num_frames,
        face_scale=args.face_scale,
        keep_short=args.keep_short,
        segment_codec=args.segment_codec,
        detect_once_per_segment=(
            args.detect_once_per_segment
        ),
        face_device=args.face_device,
        max_segments_per_video=(
            args.max_segments_per_video
        ),
    )
