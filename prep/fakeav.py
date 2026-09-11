import argparse
import csv
import random
from pathlib import Path


LABELS = {
    "RealVideo-RealAudio": (0, 0, 0),
    "RealVideo-FakeAudio": (1, 0, 1),
    "FakeVideo-RealAudio": (0, 1, 1),
    "FakeVideo-FakeAudio": (1, 1, 1),
}


def collect(dataset_root):
    dataset_root = Path(dataset_root)
    rows = []
    for path in sorted(dataset_root.rglob("*.mp4")):
        matched = False
        path_text = str(path)
        for dirname, (audio_label, visual_label, overall_label) in LABELS.items():
            if dirname not in path_text:
                continue
            rows.append({
                "video_path": str(path),
                "audio_label": audio_label,
                "visual_label": visual_label,
                "overall_label": overall_label,
            })
            matched = True
            break
        if not matched:
            print(f"[WARN] skip video outside FakeAVCeleb label folders: {path}")
    return rows


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
    parser = argparse.ArgumentParser(description="Create FakeAVCeleb train/test CSV files.")
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rows = collect(args.dataset_root)
    if not rows:
        raise RuntimeError("No FakeAVCeleb videos found.")
    if args.seed is None:
        random.shuffle(rows)
    else:
        random.Random(args.seed).shuffle(rows)
    split = int(len(rows) * args.train_ratio)
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "training_set.csv", rows[:split])
    write_csv(output_dir / "test_set.csv", rows[split:])
    print(f"[OK] train={split}, test={len(rows) - split}, output={output_dir}")
