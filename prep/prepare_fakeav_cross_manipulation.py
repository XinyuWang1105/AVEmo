"""Create AVFF-style cross-manipulation splits from processed FakeAVCeleb clips.

The processed manifests keep the row index of their source CSV in each output
path.  This script uses that index to recover the original video path, then
joins it to ``meta_data.csv`` by the unambiguous dataset location key:

    (type, race, gender, source, filename)

Some FakeAVCeleb metadata records share the same full location but disagree on
the manipulation method.  Those source videos are reported and excluded rather
than assigned to an arbitrary cross-manipulation class.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any


REAL_CLASS = "RVRA"
HOLDOUT_CLASSES = (
    "RVFA",
    "FVRA-WL",
    "FVFA-FS",
    "FVFA-GAN",
    "FVFA-WL",
)
OUTPUT_CLASSES = (REAL_CLASS, *HOLDOUT_CLASSES)

EXPECTED_LABELS = {
    "RVRA": (0, 0, 0),
    "RVFA": (1, 0, 1),
    "FVRA-WL": (0, 1, 1),
    "FVRA-FS": (0, 1, 1),
    "FVRA-GAN": (0, 1, 1),
    "FVFA-FS": (1, 1, 1),
    "FVFA-GAN": (1, 1, 1),
    "FVFA-WL": (1, 1, 1),
}

METHOD_TO_CLASS = {
    ("real", "realvideo-realaudio"): "RVRA",
    ("rtvc", "realvideo-fakeaudio"): "RVFA",
    ("wav2lip", "fakevideo-realaudio"): "FVRA-WL",
    ("faceswap", "fakevideo-realaudio"): "FVRA-FS",
    ("fsgan", "fakevideo-realaudio"): "FVRA-GAN",
    ("faceswap-wav2lip", "fakevideo-fakeaudio"): "FVFA-FS",
    ("fsgan-wav2lip", "fakevideo-fakeaudio"): "FVFA-GAN",
    ("wav2lip", "fakevideo-fakeaudio"): "FVFA-WL",
}

MEDIA_TYPES = {
    "realvideo-realaudio",
    "realvideo-fakeaudio",
    "fakevideo-realaudio",
    "fakevideo-fakeaudio",
}
ROW_INDEX_PATTERN = re.compile(r"(?:^|/)(\d{7})_")


def normalize(value: object) -> str:
    """Normalize metadata and path components for deterministic comparison."""
    text = "" if value is None else str(value)
    text = text.strip().replace("\\", "/")
    return re.sub(r"\s+", " ", text).casefold()


def parse_binary_label(value: object) -> int:
    text = normalize(value)
    if text in {"0", "0.0", "false", "real", "no"}:
        return 0
    if text in {"1", "1.0", "true", "fake", "yes"}:
        return 1
    raise ValueError(f"Expected a binary label, got {value!r}")


def labels_from_row(row: dict[str, str]) -> tuple[int, int, int]:
    return (
        parse_binary_label(row["audio_label"]),
        parse_binary_label(row["visual_label"]),
        parse_binary_label(row["overall_label"]),
    )


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        fields = list(reader.fieldnames or [])
        rows = list(reader)

    if not fields or not rows:
        raise RuntimeError(f"CSV is empty or missing a header: {path}")
    return fields, rows


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def source_path_from_raw_row(row: dict[str, str]) -> str:
    for key in ("video_path", "visual_path", "path"):
        value = row.get(key)
        if value:
            return str(value)
    raise KeyError("Raw CSV requires a video_path, visual_path, or path column.")


def source_row_index(processed_row: dict[str, str]) -> int:
    for key in ("video_path", "face_crop_folder", "frame_dir"):
        value = processed_row.get(key, "")
        matches = ROW_INDEX_PATTERN.findall(str(value).replace("\\", "/"))
        if matches:
            return int(matches[-1])
    raise ValueError(
        "Cannot recover the original CSV row index from processed row: "
        f"{processed_row}"
    )


def key_from_raw_path(video_path: str) -> tuple[str, str, str, str, str]:
    """Extract (type, race, gender, source, filename) from a FakeAVCeleb path."""
    parts = [part for part in str(video_path).replace("\\", "/").split("/") if part]
    media_index = next(
        (index for index, part in enumerate(parts) if normalize(part) in MEDIA_TYPES),
        None,
    )
    if media_index is None or len(parts) < media_index + 5:
        raise ValueError(
            "Expected FakeAVCeleb path layout "
            "<type>/<race>/<gender>/<source>/<filename>, got: "
            f"{video_path}"
        )

    media_type, race, gender, source = parts[media_index : media_index + 4]
    filename = PurePosixPath(parts[-1]).name
    return (
        normalize(media_type),
        normalize(race),
        normalize(gender),
        normalize(source),
        normalize(filename),
    )


def key_from_metadata_row(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    required = ("type", "race", "gender", "source", "path")
    missing = [key for key in required if not row.get(key)]
    if missing:
        raise ValueError(f"Metadata row has missing fields {missing}: {row}")
    return (
        normalize(row["type"]),
        normalize(row["race"]),
        normalize(row["gender"]),
        normalize(row["source"]),
        normalize(PurePosixPath(str(row["path"]).replace("\\", "/")).name),
    )


def protocol_class(metadata_row: dict[str, str]) -> str:
    key = (normalize(metadata_row["method"]), normalize(metadata_row["type"]))
    try:
        return METHOD_TO_CLASS[key]
    except KeyError as exc:
        raise ValueError(
            "Unsupported FakeAVCeleb method/type pair: "
            f"method={metadata_row.get('method')!r}, type={metadata_row.get('type')!r}"
        ) from exc


def build_metadata_index(
    metadata_rows: list[dict[str, str]],
) -> dict[tuple[str, str, str, str, str], list[dict[str, str]]]:
    index: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in metadata_rows:
        index[key_from_metadata_row(row)].append(row)
    return index


def unique_metadata_match(
    candidates: list[dict[str, str]],
) -> tuple[dict[str, str] | None, list[str]]:
    """Return a unique class assignment, or every conflicting class name."""
    classes = sorted({protocol_class(row) for row in candidates})
    if len(classes) != 1:
        return None, classes

    matching = [row for row in candidates if protocol_class(row) == classes[0]]
    return matching[0], classes


def annotate_processed_rows(
    split_name: str,
    raw_rows: list[dict[str, str]],
    processed_rows: list[dict[str, str]],
    metadata_index: dict[tuple[str, str, str, str, str], list[dict[str, str]]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[dict[str, str]]]:
    annotated: list[dict[str, Any]] = []
    ambiguous_sources: dict[tuple[str, int], dict[str, str]] = {}
    failures: dict[tuple[str, int], dict[str, str]] = {}

    for processed_index, processed_row in enumerate(processed_rows):
        try:
            raw_index = source_row_index(processed_row)
            raw_row = raw_rows[raw_index]
            raw_path = source_path_from_raw_row(raw_row)
            location_key = key_from_raw_path(raw_path)
        except (IndexError, KeyError, ValueError) as exc:
            failures[(split_name, processed_index)] = {
                "split": split_name,
                "processed_row_index": str(processed_index),
                "source_row_index": "",
                "source_video_path": "",
                "reason": str(exc),
            }
            continue

        candidates = metadata_index.get(location_key, [])
        if not candidates:
            failures[(split_name, raw_index)] = {
                "split": split_name,
                "processed_row_index": str(processed_index),
                "source_row_index": str(raw_index),
                "source_video_path": raw_path,
                "reason": f"No metadata match for key={location_key}",
            }
            continue

        metadata_row, classes = unique_metadata_match(candidates)
        if metadata_row is None:
            ambiguous_sources[(split_name, raw_index)] = {
                "split": split_name,
                "source_row_index": str(raw_index),
                "source_video_path": raw_path,
                "metadata_key": " | ".join(location_key),
                "candidate_classes": ",".join(classes),
                "candidate_methods": ",".join(
                    sorted({normalize(row["method"]) for row in candidates})
                ),
            }
            continue

        manipulation_type = classes[0]
        expected = EXPECTED_LABELS[manipulation_type]
        actual = labels_from_row(processed_row)
        if actual != expected:
            failures[(split_name, raw_index)] = {
                "split": split_name,
                "processed_row_index": str(processed_index),
                "source_row_index": str(raw_index),
                "source_video_path": raw_path,
                "reason": (
                    f"Label mismatch for {manipulation_type}: "
                    f"expected={expected}, actual={actual}"
                ),
            }
            continue

        row = dict(processed_row)
        row.update(
            {
                "source_row_index": raw_index,
                "source_video_path": raw_path,
                "metadata_method": metadata_row["method"],
                "metadata_type": metadata_row["type"],
                "metadata_race": metadata_row["race"],
                "metadata_gender": metadata_row["gender"],
                "metadata_source": metadata_row["source"],
                "metadata_path": metadata_row["path"],
                "manipulation_type": manipulation_type,
            }
        )
        annotated.append(row)

    return annotated, list(ambiguous_sources.values()), list(failures.values())


def source_count(rows: list[dict[str, Any]]) -> int:
    return len(
        {
            (str(row["source_row_index"]), str(row["source_video_path"]))
            for row in rows
        }
    )


def class_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row["manipulation_type"]) for row in rows).items()))


def write_class_csvs(
    output_root: Path,
    processed_fields: list[str],
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> None:
    for class_name in OUTPUT_CLASSES:
        class_dir = output_root / "classes" / class_name
        write_csv(
            class_dir / "train.csv",
            processed_fields,
            [row for row in train_rows if row["manipulation_type"] == class_name],
        )
        write_csv(
            class_dir / "test.csv",
            processed_fields,
            [row for row in test_rows if row["manipulation_type"] == class_name],
        )


def build_folds(
    output_root: Path,
    processed_fields: list[str],
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []

    for offset, holdout in enumerate(HOLDOUT_CLASSES):
        seen_types = {REAL_CLASS, *(set(HOLDOUT_CLASSES) - {holdout})}
        train_split = [row for row in train_rows if row["manipulation_type"] in seen_types]
        test_split = [
            row
            for row in test_rows
            if row["manipulation_type"] in {REAL_CLASS, holdout}
        ]

        random.Random(seed + offset * 2).shuffle(train_split)
        random.Random(seed + offset * 2 + 1).shuffle(test_split)

        train_types = set(class_counts(train_split))
        test_types = set(class_counts(test_split))
        if train_types != seen_types:
            raise RuntimeError(
                f"Invalid training fold for {holdout}: expected={sorted(seen_types)}, "
                f"actual={sorted(train_types)}"
            )
        if test_types != {REAL_CLASS, holdout}:
            raise RuntimeError(
                f"Invalid test fold for {holdout}: expected={[REAL_CLASS, holdout]}, "
                f"actual={sorted(test_types)}"
            )

        fold_dir = output_root / "folds" / holdout
        write_csv(fold_dir / "train.csv", processed_fields, train_split)
        write_csv(fold_dir / "test.csv", processed_fields, test_split)

        summary.append(
            {
                "holdout": holdout,
                "train_clips": len(train_split),
                "train_source_videos": source_count(train_split),
                "train_real_clips": sum(
                    row["manipulation_type"] == REAL_CLASS for row in train_split
                ),
                "train_fake_clips": sum(
                    row["manipulation_type"] != REAL_CLASS for row in train_split
                ),
                "test_clips": len(test_split),
                "test_source_videos": source_count(test_split),
                "test_real_clips": sum(
                    row["manipulation_type"] == REAL_CLASS for row in test_split
                ),
                "test_fake_clips": sum(
                    row["manipulation_type"] == holdout for row in test_split
                ),
            }
        )

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create AVFF five-fold cross-manipulation CSVs from processed FakeAVCeleb clips."
    )
    parser.add_argument(
        "--metadata",
        required=True,
        help="FakeAVCeleb metadata CSV.",
    )
    parser.add_argument(
        "--raw-train",
        required=True,
        help="Original train CSV used as preprocess.py input.",
    )
    parser.add_argument(
        "--processed-train",
        required=True,
        help="Processed train clip CSV.",
    )
    parser.add_argument(
        "--raw-test",
        required=True,
        help="Original test CSV used as preprocess.py input.",
    )
    parser.add_argument(
        "--processed-test",
        required=True,
        help="Processed test clip CSV.",
    )
    parser.add_argument(
        "--output-root",
        default="data/processed/fakeavceleb/cross_manipulation",
        help="Directory for class CSVs and cross-manipulation folds.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Stable row-order shuffle seed.")
    parser.add_argument(
        "--write-audit",
        action="store_true",
        help="Also write annotation, ambiguity, and summary audit files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)

    processed_fields, processed_train = read_csv(Path(args.processed_train))
    processed_test_fields, processed_test = read_csv(Path(args.processed_test))
    if processed_fields != processed_test_fields:
        raise RuntimeError(
            "Processed train and test CSV fields differ: "
            f"train={processed_fields}, test={processed_test_fields}"
        )

    _, raw_train = read_csv(Path(args.raw_train))
    _, raw_test = read_csv(Path(args.raw_test))
    _, metadata_rows = read_csv(Path(args.metadata))
    metadata_index = build_metadata_index(metadata_rows)

    annotated_train, train_ambiguous, train_failures = annotate_processed_rows(
        "train", raw_train, processed_train, metadata_index
    )
    annotated_test, test_ambiguous, test_failures = annotate_processed_rows(
        "test", raw_test, processed_test, metadata_index
    )

    failures = train_failures + test_failures
    if failures:
        print("[ERROR] metadata annotation failed for processed sources")
        for failure in failures[:20]:
            print(
                "  "
                f"split={failure['split']} source_row={failure['source_row_index']} "
                f"reason={failure['reason']}"
            )
        raise SystemExit(
            f"Refusing to write splits with {len(failures)} unmatched or invalid sources."
        )

    output_root.mkdir(parents=True, exist_ok=True)
    ambiguous = train_ambiguous + test_ambiguous
    write_class_csvs(output_root, processed_fields, annotated_train, annotated_test)
    fold_summary = build_folds(
        output_root,
        processed_fields,
        annotated_train,
        annotated_test,
        args.seed,
    )
    if args.write_audit:
        annotation_fields = [
            *processed_fields,
            "source_row_index",
            "source_video_path",
            "metadata_method",
            "metadata_type",
            "metadata_race",
            "metadata_gender",
            "metadata_source",
            "metadata_path",
            "manipulation_type",
        ]
        write_csv(output_root / "annotated_training.csv", annotation_fields, annotated_train)
        write_csv(output_root / "annotated_test.csv", annotation_fields, annotated_test)
        write_csv(
            output_root / "ambiguous_sources_dropped.csv",
            [
                "split",
                "source_row_index",
                "source_video_path",
                "metadata_key",
                "candidate_classes",
                "candidate_methods",
            ],
            ambiguous,
        )
        write_csv(output_root / "fold_summary.csv", list(fold_summary[0]), fold_summary)

        report = {
            "metadata_rows": len(metadata_rows),
            "metadata_location_keys": len(metadata_index),
            "train": {
                "processed_clips": len(processed_train),
                "annotated_clips": len(annotated_train),
                "annotated_source_videos": source_count(annotated_train),
                "classes": class_counts(annotated_train),
                "ambiguous_source_videos_dropped": len(train_ambiguous),
            },
            "test": {
                "processed_clips": len(processed_test),
                "annotated_clips": len(annotated_test),
                "annotated_source_videos": source_count(annotated_test),
                "classes": class_counts(annotated_test),
                "ambiguous_source_videos_dropped": len(test_ambiguous),
            },
            "folds": fold_summary,
        }
        with (output_root / "report.json").open("w", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)

    print(f"[OK] annotated train clips: {len(annotated_train)} {class_counts(annotated_train)}")
    print(f"[OK] annotated test clips: {len(annotated_test)} {class_counts(annotated_test)}")
    print(f"[OK] dropped ambiguous source videos: {len(ambiguous)}")
    print(f"[OK] folds: {output_root / 'folds'}")
    if args.write_audit:
        print(f"[OK] audit files: {output_root}")


if __name__ == "__main__":
    main()
