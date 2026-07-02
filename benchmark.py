from pathlib import Path
import argparse
import time
import statistics

from predict import predict


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def collect_images(path: Path):
    if path.is_file():
        return [path]

    images = []
    for p in path.rglob("*"):
        if p.suffix.lower() in VALID_EXTENSIONS:
            images.append(p)

    return sorted(images)


def benchmark(input_path: Path, model_path: str, warmup: int, repeat: int):
    images = collect_images(input_path)

    if not images:
        raise ValueError(f"No images found in: {input_path}")

    print(f"Images found: {len(images)}")
    print(f"Model: {model_path}")
    print(f"Warmup runs per image: {warmup}")
    print(f"Measured runs per image: {repeat}")

    # Warmup
    for img in images[: min(5, len(images))]:
        for _ in range(warmup):
            _ = predict(str(img), model_path)

    times_ms = []

    for img in images:
        for _ in range(repeat):
            start = time.perf_counter()
            _ = predict(str(img), model_path)
            end = time.perf_counter()

            times_ms.append((end - start) * 1000.0)

    mean_ms = statistics.mean(times_ms)
    median_ms = statistics.median(times_ms)
    min_ms = min(times_ms)
    max_ms = max(times_ms)

    p95_ms = sorted(times_ms)[int(0.95 * (len(times_ms) - 1))]

    print("\nLatency results")
    print("-" * 50)
    print(f"Total measured predictions: {len(times_ms)}")
    print(f"Mean latency:   {mean_ms:.2f} ms/image")
    print(f"Median latency: {median_ms:.2f} ms/image")
    print(f"P95 latency:    {p95_ms:.2f} ms/image")
    print(f"Min latency:    {min_ms:.2f} ms/image")
    print(f"Max latency:    {max_ms:.2f} ms/image")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="data",
        help="Image file or folder to benchmark on",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/recapture_logistic.joblib",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
    )

    args = parser.parse_args()

    benchmark(
        input_path=Path(args.input),
        model_path=args.model,
        warmup=args.warmup,
        repeat=args.repeat,
    )


if __name__ == "__main__":
    main()