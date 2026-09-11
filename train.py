from __future__ import annotations

import argparse
import csv
import importlib.util
import inspect
import json
import logging
import math
import random
import shutil
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import ConcatDataset, DataLoader
from tqdm.auto import tqdm

from data import (
    AudioVideoDataset,
    skip_none_collate,
)
from metrics import binary_metrics
from models import build_model


METRIC_FIELDS = [
    "acc",
    "balanced_acc",
    "auc",
    "ap",
    "precision",
    "recall",
    "sensitivity",
    "specificity",
    "f1",
    "mcc",
    "npv",
    "tn",
    "fp",
    "fn",
    "tp",
    "true_real",
    "true_fake",
    "true_fake_rate",
    "pred_real",
    "pred_fake",
    "pred_fake_rate",
    "single_class_prediction",
    "predicted_only_class",
    "score_mean",
    "score_std",
    "score_min",
    "score_max",
    "num_samples",
    "threshold",
]

CSV_FIELDS = [
    "epoch",
    "dataset",
    "head",
    "loss",
    "train_loss",
    "lr_backbone",
    "lr_module",
    "epoch_seconds",
] + METRIC_FIELDS

AP_AUC_SUMMARY_FIELDS = [
    "epoch",
    "dataset",
    "head",
    "ap",
    "auc",
    "num_samples",
    "skipped_samples",
    "train_loss",
    "lr_backbone",
    "lr_module",
    "epoch_seconds",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an audio-visual consistency detector from a Python config."
    )
    parser.add_argument("config", help="Path to config/*.py")
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    spec = importlib.util.spec_from_file_location("experiment_config", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "CONFIG") or not isinstance(module.CONFIG, dict):
        raise TypeError(f"Config file must define a dict named CONFIG: {path}")
    config = deepcopy(module.CONFIG)
    config["_config_path"] = str(path)
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def release_cuda_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def create_run_dir(base_dir: str | Path, name: str) -> Path:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"{name}_{timestamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = base / f"{name}_{timestamp}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True)
    (base / "latest_run.txt").write_text(str(run_dir.resolve()), encoding="utf-8")
    return run_dir


def setup_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("av_forgery.train")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    file_handler = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def csv_paths(value) -> List[Path]:
    values = value if isinstance(value, (list, tuple)) else [value]
    paths = [Path(item) for item in values]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Dataset CSV not found: {path}")
    return paths


def make_dataset(
    csv_path: Path,
    config: dict,
    mode: str,
    dataset_name: str | None = None,
) -> AudioVideoDataset:
    data_cfg = config["data"]
    train_cfg = config["train"]
    real_balance = (
        dict(data_cfg.get("train", {}).get("lrs2_real_balance", {}))
        if mode == "train"
        else {}
    )
    balance_enabled = bool(real_balance.get("enabled", False))
    return AudioVideoDataset(
        csv_path=str(csv_path),
        mode=mode,
        num_frames=int(data_cfg["num_frames"]),
        image_size=int(data_cfg["image_size"]),
        target_length=int(data_cfg["target_length"]),
        mel_bins=int(data_cfg["mel_bins"]),
        audio_mean=float(data_cfg["audio_mean"]),
        audio_std=float(data_cfg["audio_std"]),
        audio_cache_dir=data_cfg.get("audio_cache_dir"),
        augment=bool(train_cfg.get("augment", False) and mode == "train"),
        video_augment=data_cfg.get("video_augment"),
        freq_mask_param=int(data_cfg.get("freq_mask_param", 20)),
        time_mask_param=int(data_cfg.get("time_mask_param", 80)),
        supplemental_real_csv=(
            real_balance.get("csv") if balance_enabled else None
        ),
        supplemental_real_max_samples=(
            real_balance.get("max_samples", "auto")
            if balance_enabled
            else None
        ),
        supplemental_real_seed=int(
            real_balance.get("seed", train_cfg.get("seed", 42))
        ),
        supplemental_real_fixed=bool(real_balance.get("fixed", True)),
        skip_bad_media=bool(
            data_cfg.get("skip_bad_eval_media", mode == "eval")
            if mode == "eval"
            else data_cfg.get("skip_bad_train_media", False)
        ),
    )


def loader_kwargs(config: dict) -> dict:
    workers = int(config["train"].get("num_workers", 0))
    kwargs = {
        "num_workers": workers,
        "pin_memory": bool(config["train"].get("pin_memory", True)),
    }
    if workers > 0:
        kwargs["persistent_workers"] = True
    return kwargs


def build_train_loader(config: dict):
    train_csv = Path(config["data"]["train"]["csv"])
    dataset = make_dataset(train_csv, config, mode="train")
    sampler = None
    shuffle = True
    if config["train"].get("weighted_sampling", False):
        sampler_seed = int(config["train"].get("seed", 42))
        sampler_name = str(config["train"].get("sampler", "auto")).lower()
        if sampler_name == "auto":
            binary_only = bool(
                config.get("model", {})
                .get("params", {})
                .get("binary_only", False)
            )
            sampler_name = "binary" if binary_only else "combination"

        if sampler_name in {"binary", "overall"}:
            sampler = dataset.get_binary_weighted_sampler(seed=sampler_seed)
        elif sampler_name in {"binary_ratio", "ratio"}:
            sampler = dataset.get_binary_ratio_sampler(
                real_fake_ratio=config["train"].get("real_fake_ratio", (1, 1)),
                seed=sampler_seed,
            )
        elif sampler_name in {"binary_subset", "balanced_subset", "subset"}:
            sampler = dataset.get_binary_subset_sampler(
                samples_per_class=config["train"].get("samples_per_class"),
                seed=sampler_seed,
            )
        elif sampler_name in {"binary_subset_ratio", "subset_ratio"}:
            sampler = dataset.get_binary_subset_ratio_sampler(
                real_fake_ratio=config["train"].get("real_fake_ratio", (1, 3)),
                seed=sampler_seed,
            )
        elif sampler_name in {"comb", "combination", "four_combination"}:
            sampler = dataset.get_comb_weighted_sampler(seed=sampler_seed)
        else:
            raise ValueError(
                "train.sampler must be one of: auto, binary, binary_ratio, "
                "binary_subset, binary_subset_ratio, combination"
            )
        shuffle = False
    loader = DataLoader(
        dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=shuffle,
        sampler=sampler,
        drop_last=bool(config["train"].get("drop_last", True)),
        **loader_kwargs(config),
    )
    return loader, dataset


def build_eval_loaders(config: dict):
    loaders = {}
    for name, csv_value in config["data"]["eval"].items():
        paths = csv_paths(csv_value)
        datasets = [
            make_dataset(path, config, mode="eval", dataset_name=name)
            for path in paths
        ]
        dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
        loader = DataLoader(
            dataset,
            batch_size=int(config["train"].get("eval_batch_size") or config["train"]["batch_size"]),
            shuffle=False,
            drop_last=False,
            collate_fn=skip_none_collate,
            **loader_kwargs(config),
        )
        loaders[name] = (loader, dataset, paths)
    return loaders


def base_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def build_optimizer(model: nn.Module, config: dict):
    cfg = config["optimizer"]
    if cfg.get("name", "adamw").lower() != "adamw":
        raise ValueError("Only AdamW is currently supported")
    if hasattr(model, "parameter_groups"):
        group_options = dict(
            backbone_lr=float(cfg.get("backbone_lr", cfg["lr"])),
            module_lr=float(cfg["lr"]),
        )
        if "adapter_lr" in inspect.signature(model.parameter_groups).parameters:
            group_options["adapter_lr"] = float(
                cfg.get("adapter_lr", cfg["lr"])
            )
        parameters = model.parameter_groups(**group_options)
    else:
        parameters = model.parameters()
    return torch.optim.AdamW(
        parameters,
        lr=float(cfg["lr"]),
        betas=tuple(cfg.get("betas", (0.9, 0.999))),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )


def build_scheduler(optimizer, config: dict):
    cfg = config.get("scheduler", {})
    name = cfg.get("name", "none").lower()
    if name in ("none", ""):
        return None
    if name == "cosine_warm_restarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=int(cfg.get("t0", 5)),
            T_mult=int(cfg.get("t_mult", 2)),
            eta_min=float(cfg.get("eta_min", 1e-8)),
        )
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(cfg.get("t_max", config["train"]["epochs"])),
            eta_min=float(cfg.get("eta_min", 1e-8)),
        )
    raise ValueError(f"Unknown scheduler: {name}")


def current_lrs(optimizer) -> tuple[float, float]:
    by_name = {
        group.get("group_name", str(index)): float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }
    if "new_modules" in by_name:
        backbone = by_name.get(
            "pretrained_modules",
            by_name.get(
                "backbone_adapters",
                by_name.get("backbones", 0.0),
            ),
        )
        module = by_name["new_modules"]
    else:
        backbone = float(optimizer.param_groups[0]["lr"])
        module = float(optimizer.param_groups[-1]["lr"])
    return backbone, module


def forward_loss(
    model: nn.Module,
    batch: dict,
    criterion: nn.Module,
    device: torch.device,
    loss_cfg: dict,
):
    audio = batch["audio"].to(device, non_blocking=True)
    video = batch["video"].to(device, non_blocking=True)
    overall_label = batch["label"].to(device, non_blocking=True)
    audio_label = batch["audio_label"].to(device, non_blocking=True)
    video_label = batch["visual_label"].to(device, non_blocking=True)

    output = model(audio, video)
    fusion_loss = criterion(output["logit"], overall_label)
    audio_loss = torch.zeros((), device=device)
    video_loss = torch.zeros((), device=device)
    av_loss = torch.zeros((), device=device)
    consistency_loss = torch.zeros((), device=device)
    content_consistency_loss = torch.zeros((), device=device)
    temporal_sync_loss = torch.zeros((), device=device)

    if "audio_logit" in output:
        audio_loss = criterion(output["audio_logit"], audio_label)
        video_loss = criterion(output["video_logit"], video_label)
    if "av_logit" in output:
        av_loss = criterion(output["av_logit"], overall_label)

    consistency_weight = float(loss_cfg.get("consistency", 0.0))
    content_consistency_weight = float(loss_cfg.get("content_consistency", 0.0))
    temporal_sync_weight = float(loss_cfg.get("temporal_sync", 0.0))
    needs_content = consistency_weight > 0 or content_consistency_weight > 0
    needs_sync = consistency_weight > 0 or temporal_sync_weight > 0
    signed_label = 1.0 - 2.0 * overall_label
    if needs_content:
        if "content_score" not in output:
            raise RuntimeError(
                "content_consistency/consistency loss requires "
                "model output['content_score']"
            )
        content_consistency_loss = (
            signed_label * output["content_score"]
        ).mean()
    if needs_sync:
        if "temporal_sync_score" not in output:
            raise RuntimeError(
                "temporal_sync/consistency loss requires "
                "model output['temporal_sync_score']"
            )
        temporal_sync_loss = (
            signed_label * output["temporal_sync_score"]
        ).mean()
    if consistency_weight > 0:
        consistency_loss = content_consistency_loss + temporal_sync_loss

    total = (
        float(loss_cfg.get("fusion", 1.0)) * fusion_loss
        + float(loss_cfg.get("audio", 0.0)) * audio_loss
        + float(loss_cfg.get("video", 0.0)) * video_loss
        + float(loss_cfg.get("av", 0.0)) * av_loss
        + consistency_weight * consistency_loss
        + content_consistency_weight * content_consistency_loss
        + temporal_sync_weight * temporal_sync_loss
    )
    parts = {
        "fusion": float(fusion_loss.detach().cpu()),
        "audio": float(audio_loss.detach().cpu()),
        "video": float(video_loss.detach().cpu()),
        "av": float(av_loss.detach().cpu()),
        "consistency": float(consistency_loss.detach().cpu()),
        "content_consistency": float(content_consistency_loss.detach().cpu()),
        "temporal_sync": float(temporal_sync_loss.detach().cpu()),
    }
    return total, output, parts


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    loss_cfg: dict,
    description: str,
    amp_enabled: bool,
):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    skipped_samples = 0
    labels = {"fusion": [], "audio": [], "video": [], "av": []}
    scores = {"fusion": [], "audio": [], "video": [], "av": []}

    progress = tqdm(
        loader,
        desc=description,
        unit="batch",
        dynamic_ncols=True,
        leave=False,
    )
    with torch.inference_mode():
        for batch in progress:
            if batch is None:
                continue
            skipped_samples += int(batch.pop("_skipped_samples", 0))
            with autocast(enabled=amp_enabled):
                loss, output, _ = forward_loss(
                    model,
                    batch,
                    criterion,
                    device,
                    loss_cfg,
                )
            batch_size = int(batch["label"].shape[0])
            total_loss += float(loss.detach().cpu()) * batch_size
            total_samples += batch_size

            fusion_prob = torch.sigmoid(output["logit"]).detach().cpu().numpy()
            labels["fusion"].extend(batch["label"].numpy().tolist())
            scores["fusion"].extend(fusion_prob.tolist())

            if "audio_logit" in output:
                audio_prob = torch.sigmoid(output["audio_logit"]).detach().cpu().numpy()
                video_prob = torch.sigmoid(output["video_logit"]).detach().cpu().numpy()
                labels["audio"].extend(batch["audio_label"].numpy().tolist())
                labels["video"].extend(batch["visual_label"].numpy().tolist())
                scores["audio"].extend(audio_prob.tolist())
                scores["video"].extend(video_prob.tolist())
            if "av_logit" in output:
                av_prob = torch.sigmoid(output["av_logit"]).detach().cpu().numpy()
                labels["av"].extend(batch["label"].numpy().tolist())
                scores["av"].extend(av_prob.tolist())

            progress.set_postfix(
                loss=f"{total_loss / max(total_samples, 1):.4f}",
                skipped=skipped_samples,
                refresh=False,
            )
            del loss, output, batch

    if not labels["fusion"]:
        raise RuntimeError(
            f"{description}: no valid samples remained after skipping "
            f"{skipped_samples} bad media samples"
        )
    result = {
        "loss": total_loss / max(total_samples, 1),
        "skipped_samples": skipped_samples,
        "heads": {
            "fusion": binary_metrics(labels["fusion"], scores["fusion"]),
        },
    }
    if labels["audio"]:
        result["heads"]["audio"] = binary_metrics(labels["audio"], scores["audio"])
        result["heads"]["video"] = binary_metrics(labels["video"], scores["video"])
    if labels["av"]:
        result["heads"]["av"] = binary_metrics(labels["av"], scores["av"])
    return result


def format_metrics(metrics: dict) -> str:
    collapse = ""
    if metrics.get("single_class_prediction"):
        label = metrics.get("predicted_only_class")
        collapse = " ALL_FAKE" if label == 1 else " ALL_REAL"
    return (
        f"ACC={metrics['acc']:.4f} "
        f"BACC={metrics['balanced_acc']:.4f} "
        f"AUC={metrics['auc']:.4f} "
        f"AP={metrics['ap']:.4f} "
        f"F1={metrics['f1']:.4f} "
        f"P={metrics['precision']:.4f} "
        f"R={metrics['recall']:.4f} "
        f"SPEC={metrics['specificity']:.4f} "
        f"MCC={metrics['mcc']:.4f} "
        f"CM=[{metrics['tn']},{metrics['fp']},{metrics['fn']},{metrics['tp']}] "
        f"PRED_FAKE={metrics['pred_fake']}/{metrics['num_samples']} "
        f"({metrics['pred_fake_rate']:.2%}){collapse}"
    )


def append_metrics_csv(
    path: Path,
    epoch: int,
    dataset_name: str,
    result: dict,
    train_loss: float,
    lr_backbone: float,
    lr_module: float,
    epoch_seconds: float,
):
    exists = path.is_file() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        for head, metrics in result["heads"].items():
            row = {
                "epoch": epoch,
                "dataset": dataset_name,
                "head": head,
                "loss": result["loss"],
                "train_loss": train_loss,
                "lr_backbone": lr_backbone,
                "lr_module": lr_module,
                "epoch_seconds": epoch_seconds,
            }
            row.update(metrics)
            writer.writerow(row)


def finite_mean(values: Iterable[float]) -> float:
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    if not finite_values:
        return float("nan")
    return float(np.mean(finite_values))


def append_ap_auc_summary_csv(
    path: Path,
    epoch: int,
    evaluation_results: dict,
    head: str,
    mean_dataset_names: set[str] | None,
    train_loss: float,
    lr_backbone: float,
    lr_module: float,
    epoch_seconds: float,
) -> tuple[float, float]:
    exists = path.is_file() and path.stat().st_size > 0
    rows = []
    aps = []
    aucs = []
    for dataset_name, result in evaluation_results.items():
        metrics = result["heads"].get(head)
        if metrics is None:
            continue
        ap = float(metrics["ap"])
        auc = float(metrics["auc"])
        if mean_dataset_names is None or dataset_name in mean_dataset_names:
            aps.append(ap)
            aucs.append(auc)
        rows.append(
            {
                "epoch": epoch,
                "dataset": dataset_name,
                "head": head,
                "ap": ap,
                "auc": auc,
                "num_samples": metrics.get("num_samples"),
                "skipped_samples": int(result.get("skipped_samples", 0)),
                "train_loss": train_loss,
                "lr_backbone": lr_backbone,
                "lr_module": lr_module,
                "epoch_seconds": epoch_seconds,
            }
        )

    mean_ap = finite_mean(aps)
    mean_auc = finite_mean(aucs)
    rows.append(
        {
            "epoch": epoch,
            "dataset": "MEAN",
            "head": head,
            "ap": mean_ap,
            "auc": mean_auc,
            "num_samples": "",
            "skipped_samples": "",
            "train_loss": train_loss,
            "lr_backbone": lr_backbone,
            "lr_module": lr_module,
            "epoch_seconds": epoch_seconds,
        }
    )

    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=AP_AUC_SUMMARY_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
    return mean_ap, mean_auc


def log_ap_auc_summary(
    logger: logging.Logger,
    epoch: int,
    evaluation_results: dict,
    head: str,
    mean_dataset_names: set[str] | None,
    mean_ap: float,
    mean_auc: float,
) -> None:
    pieces = []
    for dataset_name, result in evaluation_results.items():
        metrics = result["heads"].get(head)
        if metrics is None:
            continue
        pieces.append(
            f"{dataset_name}:AP={float(metrics['ap']):.4f},AUC={float(metrics['auc']):.4f}"
        )
    mean_scope = "configured" if mean_dataset_names is not None else "all"
    logger.info(
        "epoch=%d head=%s AP/AUC summary %s MEAN[%s]:AP=%.4f,AUC=%.4f",
        epoch,
        head,
        " | ".join(pieces),
        mean_scope,
        mean_ap,
        mean_auc,
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    metrics: dict,
    config: dict,
):
    torch.save(
        {
            "model": base_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "metrics": metrics,
            "config": json_safe(config),
        },
        path,
    )


def is_better(value: float, best: float, mode: str) -> bool:
    if not math.isfinite(value):
        return False
    return value > best if mode == "max" else value < best


def main():
    args = parse_args()
    config = load_config(args.config)
    train_cfg = config["train"]
    set_seed(int(train_cfg.get("seed", 42)))
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass

    run_dir = create_run_dir(config.get("output", "runs"), config["name"])
    logger = setup_logger(run_dir)
    shutil.copy2(config["_config_path"], run_dir / "config.py")
    with (run_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(json_safe(config), file, indent=2, ensure_ascii=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("run=%s", run_dir.resolve())
    logger.info("device=%s", device)
    logger.info("model=%s", config["model"])

    train_loader, train_dataset = build_train_loader(config)
    eval_loaders = build_eval_loaders(config)
    logger.info(
        "train=%s samples=%d batches=%d",
        config["data"]["train"]["csv"],
        len(train_dataset),
        len(train_loader),
    )
    logger.info(
        "training_mode=%s primary_samples=%d supplemental_real=%d weighted_sampling=%s",
        "binary_only"
        if bool(config["model"].get("params", {}).get("binary_only", False))
        else "multi_task",
        getattr(train_dataset, "primary_size", len(train_dataset)),
        getattr(train_dataset, "supplemental_real_size", 0),
        bool(config["train"].get("weighted_sampling", False)),
    )
    for name, (_, dataset, paths) in eval_loaders.items():
        logger.info("eval=%s samples=%d csv=%s", name, len(dataset), [str(p) for p in paths])

    model = build_model(
        config["model"]["name"],
        config["model"].get("params", {}),
    ).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("parameters total=%.2fM trainable=%.2fM", total / 1e6, trainable / 1e6)

    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    if torch.cuda.device_count() > 1:
        logger.info("using DataParallel GPUs=%d", torch.cuda.device_count())
        model = nn.DataParallel(model)

    criterion = nn.BCEWithLogitsLoss()
    amp_enabled = bool(train_cfg.get("amp", True) and device.type == "cuda")
    scaler = GradScaler(enabled=amp_enabled)
    grad_accum = max(1, int(train_cfg.get("grad_accum_steps", 1)))
    eval_every = max(1, int(train_cfg.get("eval_every", 1)))
    binary_only = bool(
        config["model"].get("params", {}).get("binary_only", False)
    )
    metrics_csv = run_dir / "metrics.csv"
    ap_auc_csv = run_dir / "eval_ap_auc.csv"
    logger.info(
        "micro_batch=%d grad_accum=%d effective_batch=%d amp=%s",
        int(train_cfg["batch_size"]),
        grad_accum,
        int(train_cfg["batch_size"]) * grad_accum,
        amp_enabled,
    )

    save_cfg = config["save"]
    ap_auc_summary_cfg = train_cfg.get("ap_auc_summary", {})
    ap_auc_summary_head = ap_auc_summary_cfg.get("head", save_cfg.get("head", "fusion"))
    ap_auc_mean_datasets = ap_auc_summary_cfg.get("mean_datasets")
    if ap_auc_mean_datasets:
        ap_auc_mean_datasets = set(ap_auc_mean_datasets)
    else:
        ap_auc_mean_datasets = None
    mode = save_cfg.get("mode", "max")
    initial_best = -float("inf") if mode == "max" else float("inf")
    best = {name: initial_best for name in save_cfg.get("datasets", eval_loaders.keys())}
    best_epoch = {name: 0 for name in best}

    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_start = time.time()
        running_total = 0.0
        running_parts = {
            "fusion": 0.0,
            "audio": 0.0,
            "video": 0.0,
            "av": 0.0,
            "consistency": 0.0,
            "content_consistency": 0.0,
            "temporal_sync": 0.0,
        }
        seen_batches = 0

        progress = tqdm(
            train_loader,
            desc=f"Train {epoch}/{train_cfg['epochs']}",
            unit="batch",
            dynamic_ncols=True,
            leave=True,
        )
        for step, batch in enumerate(progress, start=1):
            group_start = ((step - 1) // grad_accum) * grad_accum
            group_size = min(
                grad_accum,
                len(train_loader) - group_start,
            )
            with autocast(enabled=amp_enabled):
                loss, output, parts = forward_loss(
                    model,
                    batch,
                    criterion,
                    device,
                    config["loss"],
                )
                scaled_loss = loss / group_size

            scaler.scale(scaled_loss).backward()
            should_step = step % grad_accum == 0 or step == len(train_loader)
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=float(train_cfg.get("grad_clip", 1.0)),
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None and config["scheduler"].get("step", "iter") == "iter":
                    scheduler.step(epoch - 1 + step / max(len(train_loader), 1))

            running_total += float(loss.detach().cpu())
            for key in running_parts:
                running_parts[key] += parts[key]
            seen_batches += 1
            lr_backbone, lr_module = current_lrs(optimizer)
            postfix = {
                "loss": f"{running_total / seen_batches:.4f}",
                "fusion": f"{running_parts['fusion'] / seen_batches:.4f}",
                "lr": f"{lr_module:.2e}",
            }
            if not binary_only:
                postfix.update(
                    audio=f"{running_parts['audio'] / seen_batches:.4f}",
                    video=f"{running_parts['video'] / seen_batches:.4f}",
                    av=f"{running_parts['av'] / seen_batches:.4f}",
                    content=f"{running_parts['content_consistency'] / seen_batches:.4f}",
                    sync=f"{running_parts['temporal_sync'] / seen_batches:.4f}",
                )
            progress.set_postfix(**postfix, refresh=False)
            del batch, loss, scaled_loss, output, parts

        if scheduler is not None and config["scheduler"].get("step") == "epoch":
            scheduler.step()

        release_cuda_cache(device)

        train_loss = running_total / max(seen_batches, 1)
        epoch_seconds = time.time() - epoch_start
        lr_backbone, lr_module = current_lrs(optimizer)
        logger.info(
            "epoch=%d/%d train_loss=%.6f time=%.1fs lr_backbone=%.8g lr_module=%.8g",
            epoch,
            int(train_cfg["epochs"]),
            train_loss,
            epoch_seconds,
            lr_backbone,
            lr_module,
        )

        evaluation_results = {}
        if epoch % eval_every == 0:
            for dataset_name, (loader, _, _) in eval_loaders.items():
                result = evaluate(
                    model,
                    loader,
                    criterion,
                    device,
                    config["loss"],
                    description=f"Eval {dataset_name} {epoch}/{train_cfg['epochs']}",
                    amp_enabled=amp_enabled,
                )
                evaluation_results[dataset_name] = result
                for head, metrics in result["heads"].items():
                    logger.info(
                        "epoch=%d dataset=%s head=%s skipped=%d %s",
                        epoch,
                        dataset_name,
                        head,
                        int(result.get("skipped_samples", 0)),
                        format_metrics(metrics),
                    )
                append_metrics_csv(
                    metrics_csv,
                    epoch,
                    dataset_name,
                    result,
                    train_loss,
                    lr_backbone,
                    lr_module,
                    epoch_seconds,
                )
            mean_ap, mean_auc = append_ap_auc_summary_csv(
                ap_auc_csv,
                epoch,
                evaluation_results,
                ap_auc_summary_head,
                ap_auc_mean_datasets,
                train_loss,
                lr_backbone,
                lr_module,
                epoch_seconds,
            )
            log_ap_auc_summary(
                logger,
                epoch,
                evaluation_results,
                ap_auc_summary_head,
                ap_auc_mean_datasets,
                mean_ap,
                mean_auc,
            )
        release_cuda_cache(device)

        checkpoint_metrics = evaluation_results
        if save_cfg.get("last", True):
            save_checkpoint(
                run_dir / "last.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                checkpoint_metrics,
                config,
            )

        target_head = save_cfg.get("head", "fusion")
        target_metric = save_cfg.get("metric", "auc")
        for dataset_name in best:
            if dataset_name not in evaluation_results:
                continue
            value = float(
                evaluation_results[dataset_name]["heads"][target_head][target_metric]
            )
            if is_better(value, best[dataset_name], mode):
                best[dataset_name] = value
                best_epoch[dataset_name] = epoch
                filename = f"best_{dataset_name}.pth"
                save_checkpoint(
                    run_dir / filename,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    checkpoint_metrics,
                    config,
                )
                logger.info(
                    "saved %s epoch=%d %s=%.6f",
                    filename,
                    epoch,
                    target_metric,
                    value,
                )

        summary = {
            "run": str(run_dir.resolve()),
            "last_epoch": epoch,
            "best": best,
            "best_epoch": best_epoch,
        }
        with (run_dir / "summary.json").open("w", encoding="utf-8") as file:
            json.dump(json_safe(summary), file, indent=2, ensure_ascii=False)

    logger.info("training finished run=%s", run_dir.resolve())


if __name__ == "__main__":
    main()
