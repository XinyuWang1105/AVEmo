"""Visualize AVEmo emotion-content and temporal-synchronization discrepancy distributions.

The plotted quantities are the exact model outputs used by the current training code:
    content_score       = mean L1(audio_emotion, video_emotion)
    temporal_sync_score = mean MSE(audio_bottleneck, video_bottleneck)

Each point used for the main distribution is one source video. When a source video
contains multiple processed clips, its clip-level scores are averaged first.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data import AudioVideoDataset, skip_none_collate
from test import load_checkpoint
from vis.avemo_vis_utils import (
    annotate_fakeav_clips,
    balance_binary_source_records,
    resolve_chinese_font,
    resolve_device,
    set_seed,
    write_csv,
)

DEFAULT_OUTPUT = Path("runs/avemo_emotion_vis")

FOUR_CLASS_ORDER = ("RVRA", "RVFA", "FVRA", "FVFA")
FOUR_CLASS_RAW_CATEGORIES = {
    "RVRA": ("Real",),
    "RVFA": ("RVFA",),
    "FVRA": ("FVRA-FS", "FVRA-GAN", "FVRA-WL"),
    "FVFA": ("FVFA-FS", "FVFA-GAN", "FVFA-WL"),
}
FOUR_CLASS_COLORS = {
    "RVRA": "#2F8F5B",
    "RVFA": "#E9A11A",
    "FVRA": "#3976C3",
    "FVFA": "#D65252",
}


class IndexedDataset:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        if item is None:
            return None
        return dict(item, sample_index=index)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--processed_test", type=Path, required=True)
    parser.add_argument("--test_set", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_per_class",
        type=int,
        default=0,
        help="Optional cap for each class in the binary and four-class plotting subsets.",
    )
    return parser.parse_args()


def build_dataset(records, config, manifest_path):
    write_csv(manifest_path, records)
    data_cfg = config["data"]
    return AudioVideoDataset(
        csv_path=manifest_path,
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
    )


def score_clips(model, config, records, args, device):
    import torch
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm

    manifest = args.output_dir / ".emotion_score_manifest.csv"
    dataset = build_dataset(records, config, manifest)
    loader = DataLoader(
        IndexedDataset(dataset),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=skip_none_collate,
        persistent_workers=args.num_workers > 0,
    )

    output_rows = []
    skipped = 0
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Emotion scores", unit="batch", dynamic_ncols=True):
            if batch is None:
                continue
            skipped += int(batch.pop("_skipped_samples", 0))
            output = model(
                batch["audio"].to(device, non_blocking=True),
                batch["video"].to(device, non_blocking=True),
            )
            content = output["content_score"].detach().float().cpu().numpy()
            temporal = output["temporal_sync_score"].detach().float().cpu().numpy()
            probability = torch.sigmoid(output["logit"]).detach().float().cpu().numpy()
            indices = batch["sample_index"].cpu().numpy().tolist()

            for local_index, dataset_index in enumerate(indices):
                record = records[int(dataset_index)]
                output_rows.append(
                    {
                        "sample_id": record["sample_id"],
                        "source_id": record["source_id"],
                        "source_video_path": record.get("source_video_path", ""),
                        "video_path": record.get("video_path", ""),
                        "category": record["category"],
                        "overall_label": int(float(record.get("overall_label", 0))),
                        "content_score": float(content[local_index]),
                        "temporal_sync_score": float(temporal[local_index]),
                        "fake_probability": float(probability[local_index]),
                    }
                )
    return output_rows, skipped


def aggregate_sources(clip_rows):
    grouped = defaultdict(list)
    for row in clip_rows:
        grouped[(str(row["category"]), str(row["source_id"]))].append(row)

    source_rows = []
    for (category, source_id), rows in grouped.items():
        source_rows.append(
            {
                "source_id": source_id,
                "source_video_path": rows[0].get("source_video_path", ""),
                "category": category,
                "overall_label": 0 if category == "Real" else 1,
                "num_clips": len(rows),
                "content_score": float(np.mean([float(r["content_score"]) for r in rows])),
                "temporal_sync_score": float(
                    np.mean([float(r["temporal_sync_score"]) for r in rows])
                ),
                "fake_probability": float(
                    np.mean([float(r["fake_probability"]) for r in rows])
                ),
            }
        )
    return source_rows


def four_class_source_records(source_rows, max_per_class, seed):
    """Map raw subtypes to four classes and draw an equal number from each class."""
    category_map = {
        raw_category: four_class
        for four_class, raw_categories in FOUR_CLASS_RAW_CATEGORIES.items()
        for raw_category in raw_categories
    }
    grouped = defaultdict(list)
    for row in source_rows:
        raw_category = str(row["category"])
        four_class = category_map.get(raw_category)
        if four_class is None:
            continue
        grouped[four_class].append({**row, "raw_category": raw_category, "category": four_class})

    ordered_groups = {}
    for category in FOUR_CLASS_ORDER:
        rows = sorted(grouped[category], key=lambda row: str(row["source_id"]))
        if not rows:
            raise RuntimeError(f"No source videos found for four-class category {category}")
        ordered_groups[category] = rows

    target_per_class = min(len(rows) for rows in ordered_groups.values())
    if max_per_class > 0:
        target_per_class = min(target_per_class, max_per_class)

    rng = np.random.default_rng(seed)
    selected = []
    for category in FOUR_CLASS_ORDER:
        rows = ordered_groups[category]
        if target_per_class < len(rows):
            indices = rng.choice(len(rows), size=target_per_class, replace=False)
            rows = [rows[index] for index in sorted(indices.tolist())]
        selected.extend(rows)

    audit = {
        "available_counts": {
            category: len(rows) for category, rows in ordered_groups.items()
        },
        "target_per_class": target_per_class,
        "selected_counts": {
            category: sum(row["category"] == category for row in selected)
            for category in FOUR_CLASS_ORDER
        },
        "raw_category_mapping": FOUR_CLASS_RAW_CATEGORIES,
    }
    return selected, audit


def kde_curve(values, points=300):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or np.allclose(values, values[0]):
        return None, None
    low, high = float(values.min()), float(values.max())
    pad = max((high - low) * 0.08, 1e-8)
    grid = np.linspace(low - pad, high + pad, points)
    try:
        from scipy.stats import gaussian_kde

        density = gaussian_kde(values)(grid)
        return grid, density
    except Exception:
        hist, edges = np.histogram(values, bins=min(30, max(8, int(np.sqrt(len(values))))), density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        return centers, hist


def configure_chinese_style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chinese_font = resolve_chinese_font()

    plt.rcParams.update(
        {
            "font.family": chinese_font,
            "font.sans-serif": [chinese_font, "Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )
    return plt


def draw_distribution(records, output_dir):
    plt = configure_chinese_style()

    real = [row for row in records if row["category"] == "Real"]
    fake = [row for row in records if row["category"] != "Real"]
    if len(real) < 2 or len(fake) < 2:
        raise RuntimeError("Need at least two real and two fake source videos to draw distributions")

    colors = {"Real": "#2F7D4A", "Fake": "#C44E52"}
    labels = {"Real": "真实样本", "Fake": "伪造样本"}
    panels = [
        ("content_score", "情感内容差异分数", "（a）情感内容一致性"),
        ("temporal_sync_score", "时间同步差异分数", "（b）时间同步一致性"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.25))
    for axis, (key, xlabel, title) in zip(axes, panels):
        for name, rows in (("Real", real), ("Fake", fake)):
            values = np.asarray([float(row[key]) for row in rows], dtype=float)
            x, density = kde_curve(values)
            axis.plot(x, density, linewidth=1.8, color=colors[name], label=labels[name])
            axis.fill_between(x, 0, density, color=colors[name], alpha=0.12)
            axis.axvline(np.median(values), color=colors[name], linewidth=1.0, linestyle="--", alpha=0.85)

        axis.set_title(title, fontsize=10)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("密度")
        axis.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.25)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.legend(frameon=False, fontsize=8)

    fig.suptitle("FakeAVCeleb 测试集", fontsize=11, y=0.99)
    fig.tight_layout(w_pad=2.0, rect=(0, 0, 1, 0.94))
    fig.savefig(
        output_dir / "emotion_score_distribution.png",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(output_dir / "emotion_score_distribution.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / "emotion_score_distribution.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_four_class_distribution(records, output_dir):
    plt = configure_chinese_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.55))
    panels = [
        ("content_score", "情感内容差异分数", "（a）情感内容差异"),
        ("temporal_sync_score", "时间同步差异分数", "（b）时间同步差异"),
    ]

    for axis, (key, xlabel, title) in zip(axes, panels):
        for category in FOUR_CLASS_ORDER:
            values = np.asarray(
                [float(row[key]) for row in records if row["category"] == category],
                dtype=float,
            )
            x, density = kde_curve(values)
            if x is None:
                axis.axvline(
                    values[0],
                    color=FOUR_CLASS_COLORS[category],
                    linewidth=1.4,
                    label=category,
                )
                continue
            # Each class has the same number of source videos, but its numerical
            # spread can still differ sharply. Peak-normalization keeps all four
            # distributions readable while preserving each curve's location/width.
            density = density / max(float(np.max(density)), 1e-12)
            axis.plot(
                x,
                density,
                linewidth=1.45,
                color=FOUR_CLASS_COLORS[category],
                label=category,
            )
        axis.set_title(title, fontsize=10)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("归一化密度")
        axis.set_ylim(0, 1.08)
        axis.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.25)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=7,
        columnspacing=1.2,
        handlelength=1.8,
        bbox_to_anchor=(0.5, 0.93),
    )
    fig.suptitle("FakeAVCeleb 测试集：四类样本分布", fontsize=11, y=0.995)
    fig.tight_layout(w_pad=2.1, rect=(0, 0, 1, 0.80))
    fig.savefig(
        output_dir / "emotion_score_distribution_four_classes.png",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        output_dir / "emotion_score_distribution_four_classes.pdf",
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        output_dir / "emotion_score_distribution_four_classes.svg",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in (args.checkpoint, args.processed_test, args.test_set, args.metadata):
        if not path.is_file():
            raise FileNotFoundError(path)

    set_seed(args.seed)
    device = resolve_device(args.device)
    annotated, mapping_audit = annotate_fakeav_clips(
        args.processed_test, args.test_set, args.metadata
    )
    model, config, checkpoint = load_checkpoint(args.checkpoint, device)

    clip_rows, skipped = score_clips(model, config, annotated, args, device)
    if not clip_rows:
        raise RuntimeError("No valid clips were scored")
    source_rows = aggregate_sources(clip_rows)
    binary_rows, binary_balance_audit = balance_binary_source_records(
        source_rows, max_per_class=args.max_per_class, seed=args.seed
    )
    four_class_rows, four_class_audit = four_class_source_records(
        source_rows, max_per_class=args.max_per_class, seed=args.seed
    )

    write_csv(args.output_dir / "clip_scores.csv", clip_rows)
    write_csv(args.output_dir / "source_scores.csv", source_rows)
    write_csv(args.output_dir / "plot_source_scores.csv", binary_rows)
    write_csv(args.output_dir / "plot_binary_source_scores.csv", binary_rows)
    write_csv(args.output_dir / "plot_four_class_source_scores.csv", four_class_rows)
    draw_distribution(binary_rows, args.output_dir)
    draw_four_class_distribution(four_class_rows, args.output_dir)

    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "processed_test": str(args.processed_test.resolve()),
        "mapping": mapping_audit,
        "binary_balance": binary_balance_audit,
        "four_class_selection": four_class_audit,
        "scored_clips": len(clip_rows),
        "scored_source_videos": len(source_rows),
        "skipped_clips": skipped,
        "scores": {
            "content_score": "mean L1(audio_emotion, video_emotion)",
            "temporal_sync_score": "mean MSE(audio_bottleneck, video_bottleneck)",
        },
    }
    with (args.output_dir / "emotion_score_config.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(f"Saved emotion-score analysis to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
