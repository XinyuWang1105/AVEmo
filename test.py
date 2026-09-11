from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data import (
    AudioVideoDataset,
    skip_none_collate,
)
from metrics import binary_metrics
from models import build_model

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate an audio-visual forgery detector checkpoint."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output_csv", default="runs/test_predictions.csv")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--video_root",
        default=None,
        help=(
            "Root for CSVs that store relative video paths, e.g. "
            "kling2.5/230_generated.mp4."
        ),
    )
    return parser.parse_args()


def print_metrics(name: str, result: dict) -> None:
    collapse = ""
    if result.get("single_class_prediction"):
        collapse = (
            " ALL_FAKE"
            if result.get("predicted_only_class") == 1
            else " ALL_REAL"
        )
    print(
        f"{name}: ACC={result['acc']:.4f} "
        f"BACC={result['balanced_acc']:.4f} "
        f"AUC={result['auc']:.4f} "
        f"AP={result['ap']:.4f} "
        f"F1={result['f1']:.4f} "
        f"P={result['precision']:.4f} "
        f"R={result['recall']:.4f} "
        f"SPEC={result['specificity']:.4f} "
        f"MCC={result['mcc']:.4f} "
        f"CM=[{result['tn']},{result['fp']},{result['fn']},{result['tp']}] "
        f"PRED_FAKE={result['pred_fake']}/{result['num_samples']} "
        f"({result['pred_fake_rate']:.2%}){collapse}"
    )


def load_checkpoint(path: str | Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu")
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise RuntimeError(
            "Checkpoint does not contain the config saved by the replacement train.py"
        )
    model_cfg = deepcopy(config["model"])
    if model_cfg.get("name") in {"ecn", "avemo"}:
        model_cfg.setdefault("params", {})["pretrained_path"] = None
        model_cfg["params"]["require_pretrained"] = False
    model = build_model(
        model_cfg["name"],
        model_cfg.get("params", {}),
    )
    state = checkpoint["model"]
    target_state = model.state_dict()
    compatible = {}
    mismatched = []
    unexpected_state_keys = []
    for key, tensor in state.items():
        if key not in target_state:
            unexpected_state_keys.append(key)
            continue
        if tensor.shape != target_state[key].shape:
            mismatched.append(
                f"{key}: {tuple(tensor.shape)} != {tuple(target_state[key].shape)}"
            )
            continue
        compatible[key] = tensor
    incompatible = model.load_state_dict(compatible, strict=False)
    allowed_missing = {
        "missing_audio_token",
        "missing_video_token",
    }
    allowed_missing_prefixes = (
        "emotion_extractor.video_dominant.audio_to_video_aligner.",
        "emotion_extractor.audio_dominant.video_to_audio_aligner.",
        "final_decision.",
    )
    allowed_unexpected_prefixes = ("video_backbone.spatial_pool.",)
    unexpected = [
        key
        for key in [*unexpected_state_keys, *incompatible.unexpected_keys]
        if not key.startswith(allowed_unexpected_prefixes)
    ]
    missing = [
        key
        for key in incompatible.missing_keys
        if key not in allowed_missing
        and not key.startswith(allowed_missing_prefixes)
    ]
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint is incompatible with the configured model: "
            f"missing={missing}, unexpected={unexpected}"
        )
    if incompatible.missing_keys or mismatched:
        print(
            "[WARN] initialized newly introduced modules while loading an "
            f"older checkpoint: missing={list(incompatible.missing_keys)} "
            f"mismatched={mismatched}"
        )
    return model.to(device).eval(), config, checkpoint


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, checkpoint = load_checkpoint(args.checkpoint, device)
    binary_only = bool(
        config.get("model", {}).get("params", {}).get("binary_only", False)
    )

    data_cfg = config["data"]
    train_cfg = config.get("train", {})
    dataset = AudioVideoDataset(
        csv_path=args.csv,
        mode="eval",
        num_frames=int(data_cfg["num_frames"]),
        image_size=int(data_cfg["image_size"]),
        target_length=int(data_cfg["target_length"]),
        mel_bins=int(data_cfg["mel_bins"]),
        audio_mean=float(data_cfg["audio_mean"]),
        audio_std=float(data_cfg["audio_std"]),
        audio_cache_dir=data_cfg.get("audio_cache_dir"),
        freq_mask_param=int(data_cfg.get("freq_mask_param", 20)),
        time_mask_param=int(data_cfg.get("time_mask_param", 80)),
        augment=False,
        video_augment=data_cfg.get("video_augment"),
        skip_bad_media=bool(data_cfg.get("skip_bad_eval_media", True)),
        video_root=args.video_root,
    )

    workers = (
        int(args.num_workers)
        if args.num_workers is not None
        else int(train_cfg.get("num_workers", 0))
    )
    batch_size = (
        int(args.batch_size)
        if args.batch_size is not None
        else int(train_cfg.get("eval_batch_size") or train_cfg.get("batch_size", 1))
    )
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "drop_last": False,
        "num_workers": workers,
        "pin_memory": True,
        "collate_fn": skip_none_collate,
    }
    if workers > 0:
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(dataset, **loader_kwargs)

    rows = []
    labels = {"fusion": [], "audio": [], "video": [], "av": []}
    scores = {"fusion": [], "audio": [], "video": [], "av": []}

    fusion_weight_names = ["audio", "video", "state", "dynamic"]
    skipped_samples = 0
    with torch.no_grad():
        for batch in tqdm(
            loader,
            desc="Test",
            unit="batch",
            dynamic_ncols=True,
        ):
            if batch is None:
                continue
            skipped_samples += int(batch.pop("_skipped_samples", 0))
            output = model(
                batch["audio"].to(device, non_blocking=True),
                batch["video"].to(device, non_blocking=True),
            )
            fusion_prob = torch.sigmoid(output["logit"]).cpu().numpy()
            audio_prob = (
                torch.sigmoid(output["audio_logit"]).cpu().numpy()
                if "audio_logit" in output
                else np.full_like(fusion_prob, np.nan)
            )
            video_prob = (
                torch.sigmoid(output["video_logit"]).cpu().numpy()
                if "video_logit" in output
                else np.full_like(fusion_prob, np.nan)
            )
            av_prob = (
                torch.sigmoid(output["av_logit"]).cpu().numpy()
                if "av_logit" in output
                else np.full_like(fusion_prob, np.nan)
            )
            state_score = (
                output["state_score"].cpu().numpy()
                if "state_score" in output
                else np.full_like(fusion_prob, np.nan)
            )
            dynamic_score = (
                output["dynamic_score"].cpu().numpy()
                if "dynamic_score" in output
                else np.full_like(fusion_prob, np.nan)
            )
            fusion_weights = (
                output["fusion_weights"].cpu().numpy()
                if "fusion_weights" in output
                else np.full((len(fusion_prob), 4), np.nan, dtype=np.float32)
            )

            for index, path in enumerate(batch["video_path"]):
                row = {
                    "video_path": path,
                    "resolved_video_path": batch["resolved_video_path"][index],
                    "overall_label": int(batch["label"][index].item()),
                    "fusion_fake_score": float(fusion_prob[index]),
                }
                if not binary_only:
                    row.update(
                        audio_label=int(batch["audio_label"][index].item()),
                        visual_label=int(batch["visual_label"][index].item()),
                        audio_fake_score=float(audio_prob[index]),
                        video_fake_score=float(video_prob[index]),
                        av_fake_score=float(av_prob[index]),
                        state_discrepancy=float(state_score[index]),
                        dynamic_discrepancy=float(dynamic_score[index]),
                    )
                    for weight_index, name in enumerate(fusion_weight_names):
                        row[f"fusion_weight_{name}"] = float(
                            fusion_weights[index, weight_index]
                        )
                rows.append(row)

            labels["fusion"].extend(batch["label"].numpy().tolist())
            scores["fusion"].extend(fusion_prob.tolist())
            if "audio_logit" in output:
                labels["audio"].extend(batch["audio_label"].numpy().tolist())
                labels["video"].extend(batch["visual_label"].numpy().tolist())
                scores["audio"].extend(audio_prob.tolist())
                scores["video"].extend(video_prob.tolist())
            if "av_logit" in output:
                labels["av"].extend(batch["label"].numpy().tolist())
                scores["av"].extend(av_prob.tolist())

    if not labels["fusion"]:
        raise RuntimeError(
            "No valid samples remained after skipping "
            f"{skipped_samples} bad media samples"
        )
    metrics = {
        "fusion": binary_metrics(labels["fusion"], scores["fusion"]),
    }
    if labels["audio"]:
        metrics["audio"] = binary_metrics(labels["audio"], scores["audio"])
        metrics["video"] = binary_metrics(labels["video"], scores["video"])
    if labels["av"]:
        metrics["av"] = binary_metrics(labels["av"], scores["av"])

    for name, result in metrics.items():
        print_metrics(name, result)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["video_path"]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
        "csv": str(Path(args.csv).resolve()),
        "num_samples": len(dataset),
        "num_scored_samples": len(rows),
        "num_skipped_samples": skipped_samples,
        "video_root": str(args.video_root) if args.video_root else "",
        "metrics": metrics,
    }
    metrics_path = output_path.with_suffix(".metrics.json")
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    print(f"saved predictions: {output_path}")
    print(f"saved metrics: {metrics_path}")
    if skipped_samples:
        print(f"[WARN] skipped bad media samples: {skipped_samples}")


if __name__ == "__main__":
    main()
