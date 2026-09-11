import argparse
import csv
import random
from pathlib import Path


VIDEO_PATH_COLUMNS = ("video_path", "visual_path", "path", "file", "filename", "video")
LABEL_COLUMNS = ("overall_label", "label", "target", "is_fake", "fake")
AUDIO_LABEL_COLUMNS = ("audio_label", "audio_fake", "audio_target")
VISUAL_LABEL_COLUMNS = ("visual_label", "video_label", "visual_fake", "video_fake", "visual_target")
SPLIT_COLUMNS = ("split", "set", "subset", "partition")


def first_value(row, names):
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return None


def as_label(value, field):
    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "fake", "forged", "manipulated"}:
        return 1
    if text in {"0", "0.0", "false", "real", "original", "pristine"}:
        return 0
    raise ValueError(f"Cannot parse {field} label: {value}")


def split_name_from_csv(csv_path):
    name = csv_path.stem.lower()
    if "train" in name:
        return "train"
    if "dev" in name or "val" in name:
        return "test"
    if "test" in name:
        return "test"
    return None


def normalize_split(value):
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"train", "training"}:
        return "train"
    if text in {"dev", "val", "valid", "validation", "test", "testing"}:
        return "test"
    return None


def resolve_video_path(dataset_root, rel_path):
    path = Path(str(rel_path))
    if path.is_absolute():
        return path
    return dataset_root / path


def collect_from_csvs(dataset_root):
    dataset_root = Path(dataset_root)
    rows_by_split = {"train": [], "test": []}
    unsplit_rows = []
    missing = []
    csv_paths = sorted(dataset_root.glob("*.csv"))
    if not csv_paths:
        raise RuntimeError(f"No HiFi CSV metadata found under {dataset_root}")

    for csv_path in csv_paths:
        csv_split = split_name_from_csv(csv_path)
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader):
                rel_path = first_value(row, VIDEO_PATH_COLUMNS)
                if not rel_path:
                    continue
                video_path = resolve_video_path(dataset_root, rel_path)
                if not video_path.exists():
                    missing.append(str(video_path))
                    continue

                overall_value = first_value(row, LABEL_COLUMNS)
                if overall_value is None:
                    raise RuntimeError(f"{csv_path}:{row_idx + 2} has no overall label column")
                overall = as_label(overall_value, "overall")
                audio_value = first_value(row, AUDIO_LABEL_COLUMNS)
                visual_value = first_value(row, VISUAL_LABEL_COLUMNS)
                sample = {
                    "video_path": str(video_path),
                    "audio_label": as_label(audio_value, "audio") if audio_value is not None else overall,
                    "visual_label": as_label(visual_value, "visual") if visual_value is not None else overall,
                    "overall_label": overall,
                }

                row_split = normalize_split(first_value(row, SPLIT_COLUMNS)) or csv_split
                if row_split in rows_by_split:
                    rows_by_split[row_split].append(sample)
                else:
                    unsplit_rows.append(sample)

    if missing:
        print(f"[WARN] {len(missing)} videos listed in CSV files are missing; first 5:")
        for path in missing[:5]:
            print(f"       {path}")

    return rows_by_split, unsplit_rows


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["video_path", "audio_label", "visual_label", "overall_label"],
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Create HiFi-AVDF train/test CSV files from official CSV metadata.")
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rows_by_split, unsplit_rows = collect_from_csvs(args.dataset_root)

    if unsplit_rows:
        if args.seed is None:
            random.shuffle(unsplit_rows)
        else:
            random.Random(args.seed).shuffle(unsplit_rows)
        split = int(len(unsplit_rows) * args.train_ratio)
        rows_by_split["train"].extend(unsplit_rows[:split])
        rows_by_split["test"].extend(unsplit_rows[split:])

    if not rows_by_split["train"] or not rows_by_split["test"]:
        raise RuntimeError(
            "HiFi metadata did not produce both train and test rows. "
            "Provide official CSV files with split information or train/test CSV filenames."
        )

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "hifi_training_set.csv", rows_by_split["train"])
    write_csv(output_dir / "hifi_test_set.csv", rows_by_split["test"])
    print(f"[OK] train={len(rows_by_split['train'])}, test={len(rows_by_split['test'])}, output={output_dir}")
