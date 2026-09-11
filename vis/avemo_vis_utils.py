from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from prep.prepare_fakeav_cross_manipulation import (
    annotate_processed_rows,
    build_metadata_index,
    read_csv,
)

PLOT_ORDER = (
    "Real",
    "RVFA",
    "FVRA-FS",
    "FVFA-FS",
    "FVRA-GAN",
    "FVFA-GAN",
    "FVRA-WL",
    "FVFA-WL",
)

PROTOCOL_TO_PLOT = {
    "RVRA": "Real",
    "RVFA": "RVFA",
    "FVRA-FS": "FVRA-FS",
    "FVFA-FS": "FVFA-FS",
    "FVRA-GAN": "FVRA-GAN",
    "FVFA-GAN": "FVFA-GAN",
    "FVRA-WL": "FVRA-WL",
    "FVFA-WL": "FVFA-WL",
}


def write_csv(path: str | Path, records: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for record in records for key in record.keys()))
    # AudioVideoDataset reads manifests as plain UTF-8. Writing a BOM would
    # turn the first header into "\ufeffvideo_path" and make every row invalid.
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def load_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def annotate_fakeav_clips(
    processed_csv: str | Path,
    raw_csv: str | Path,
    metadata_csv: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach source-video ids and verified FakeAVCeleb manipulation labels."""
    _, processed_rows = read_csv(Path(processed_csv))
    _, raw_rows = read_csv(Path(raw_csv))
    _, metadata_rows = read_csv(Path(metadata_csv))

    annotated, ambiguous, failures = annotate_processed_rows(
        "test",
        raw_rows,
        processed_rows,
        build_metadata_index(metadata_rows),
    )
    if failures:
        examples = "\n".join(f"  {item['reason']}" for item in failures[:5])
        raise RuntimeError(
            f"Failed to map {len(failures)} processed clips to metadata:\n{examples}"
        )
    if not annotated:
        raise RuntimeError("No processed FakeAVCeleb clips could be mapped to metadata")

    records: list[dict[str, Any]] = []
    for index, row in enumerate(annotated):
        manipulation = str(row["manipulation_type"])
        if manipulation not in PROTOCOL_TO_PLOT:
            continue
        record = dict(row)
        record.update(
            sample_id=str(index),
            source_id=str(row["source_row_index"]),
            category=PROTOCOL_TO_PLOT[manipulation],
        )
        records.append(record)

    audit = {
        "processed_clips": len(processed_rows),
        "mapped_clips": len(records),
        "ambiguous_source_videos": len(ambiguous),
        "clip_counts": dict(Counter(str(row["category"]) for row in records)),
    }
    return records, audit


def balance_source_records(
    records: list[dict[str, Any]],
    max_per_class: int = 0,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Deterministically balance source-level records across available classes."""
    by_class: dict[str, list[dict[str, Any]]] = {}
    for category in PLOT_ORDER:
        rows = [row for row in records if str(row["category"]) == category]
        if rows:
            by_class[category] = rows
    if not by_class:
        raise RuntimeError("No source-level records are available for plotting")

    target = min(len(rows) for rows in by_class.values())
    if max_per_class > 0:
        target = min(target, int(max_per_class))
    if target < 2:
        raise RuntimeError(f"Need at least two source videos per class, got target={target}")

    rng = np.random.default_rng(seed)
    selected: list[dict[str, Any]] = []
    for category in PLOT_ORDER:
        rows = by_class.get(category)
        if not rows:
            continue
        indices = rng.choice(len(rows), size=target, replace=False)
        selected.extend(rows[int(i)] for i in sorted(indices.tolist()))

    audit = {
        "target_per_class": target,
        "selected_counts": dict(Counter(str(row["category"]) for row in selected)),
        "seed": seed,
        "max_per_class": max_per_class,
    }
    return selected, audit


def balance_binary_source_records(
    records: list[dict[str, Any]],
    max_per_class: int = 0,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Balance real/fake source videos while stratifying fake samples by subtype."""
    real = [row for row in records if str(row["category"]) == "Real"]
    fake_by_subtype = {
        category: [row for row in records if str(row["category"]) == category]
        for category in PLOT_ORDER
        if category != "Real"
    }
    fake_by_subtype = {
        category: rows for category, rows in fake_by_subtype.items() if rows
    }
    if not real or not fake_by_subtype:
        raise RuntimeError("Both real and fake source videos are required for binary balancing")

    target = min(len(real), sum(len(rows) for rows in fake_by_subtype.values()))
    if max_per_class > 0:
        target = min(target, int(max_per_class))
    if target < 2:
        raise RuntimeError(f"Need at least two videos per binary class, got target={target}")

    rng = np.random.default_rng(seed)
    real_indices = rng.choice(len(real), size=target, replace=False)
    selected = [real[int(index)] for index in sorted(real_indices.tolist())]

    categories = list(fake_by_subtype)
    base, remainder = divmod(target, len(categories))
    allocations = {
        category: min(len(fake_by_subtype[category]), base + int(index < remainder))
        for index, category in enumerate(categories)
    }
    remaining = target - sum(allocations.values())
    while remaining:
        progressed = False
        for category in categories:
            if allocations[category] < len(fake_by_subtype[category]):
                allocations[category] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            raise RuntimeError("Not enough fake source videos for binary balancing")

    for category in categories:
        rows = fake_by_subtype[category]
        indices = rng.choice(len(rows), size=allocations[category], replace=False)
        selected.extend(rows[int(index)] for index in sorted(indices.tolist()))

    audit = {
        "target_per_binary_class": target,
        "selected_counts": {
            "Real": sum(str(row["category"]) == "Real" for row in selected),
            "Fake": sum(str(row["category"]) != "Real" for row in selected),
        },
        "fake_subtype_counts": {
            category: sum(str(row["category"]) == category for row in selected)
            for category in categories
        },
        "seed": seed,
        "max_per_class": max_per_class,
    }
    return selected, audit


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def resolve_device(name: str):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def resolve_chinese_font() -> str:
    """Register an installed CJK font with Matplotlib and return its family name."""
    from matplotlib import font_manager

    for family in (
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Microsoft YaHei",
        "SimHei",
        "WenQuanYi Micro Hei",
    ):
        try:
            path = font_manager.findfont(
                font_manager.FontProperties(family=family),
                fallback_to_default=False,
            )
        except ValueError:
            continue
        font_manager.fontManager.addfont(path)
        return font_manager.FontProperties(fname=path).get_name()
    raise RuntimeError(
        "No Chinese font was found. Install fonts-noto-cjk and refresh the Matplotlib font cache."
    )


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
