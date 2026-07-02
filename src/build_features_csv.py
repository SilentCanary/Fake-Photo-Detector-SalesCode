from pathlib import Path
import argparse
import pandas as pd
from tqdm import tqdm

from src.utils.feature_orchestrator import extract_all_features


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def collect_images(folder: Path):
    images = []

    for path in folder.rglob("*"):
        if path.suffix.lower() in VALID_EXTENSIONS:
            images.append(path)

    return sorted(images)


def build_features_csv(data_dir: Path, out_path: Path):
    real_dir = data_dir / "real"
    screen_dir = data_dir / "screen"

    if not real_dir.exists():
        raise FileNotFoundError(f"Missing folder: {real_dir}")

    if not screen_dir.exists():
        raise FileNotFoundError(f"Missing folder: {screen_dir}")

    samples = []

    for img_path in collect_images(real_dir):
        samples.append((img_path, 0))

    for img_path in collect_images(screen_dir):
        samples.append((img_path, 1))

    if not samples:
        raise ValueError("No images found. Check your data folder.")

    rows = []

    for img_path, label in tqdm(samples, desc="Extracting features"):
        try:
            feats = extract_all_features(img_path)
            feats["label"] = label
            rows.append(feats)

        except Exception as e:
            print(f"[WARN] Failed on {img_path}: {e}")

    df = pd.DataFrame(rows)

    # Put important columns first
    first_cols = ["image_path", "label"]
    other_cols = [c for c in df.columns if c not in first_cols]
    df = df[first_cols + other_cols]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"\nSaved features to: {out_path}")
    print(f"Total rows: {len(df)}")
    print(f"Total features: {len(df.columns) - 2}")
    print(df["label"].value_counts())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out", type=str, default="outputs/features.csv")

    args = parser.parse_args()

    build_features_csv(
        data_dir=Path(args.data_dir),
        out_path=Path(args.out)
    )


if __name__ == "__main__":
    main()