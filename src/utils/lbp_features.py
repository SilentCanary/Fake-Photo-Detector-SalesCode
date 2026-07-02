import numpy as np
import cv2


def compute_lbp(gray: np.ndarray) -> np.ndarray:
    """
    Simple 8-neighbor Local Binary Pattern implementation.
    """
    gray = gray.astype(np.uint8)

    center = gray[1:-1, 1:-1]

    lbp = np.zeros_like(center, dtype=np.uint8)

    neighbors = [
        gray[:-2, :-2],
        gray[:-2, 1:-1],
        gray[:-2, 2:],
        gray[1:-1, 2:],
        gray[2:, 2:],
        gray[2:, 1:-1],
        gray[2:, :-2],
        gray[1:-1, :-2],
    ]

    for i, n in enumerate(neighbors):
        lbp |= ((n >= center).astype(np.uint8) << i)

    return lbp


def lbp_histogram_features(lbp: np.ndarray, prefix: str = "lbp") -> dict:
    hist, _ = np.histogram(lbp.ravel(), bins=16, range=(0, 256), density=True)

    features = {}

    for i, value in enumerate(hist):
        features[f"{prefix}_hist_{i}"] = float(value)

    entropy = -np.sum(hist * np.log2(hist + 1e-8))
    features[f"{prefix}_entropy"] = float(entropy)

    features[f"{prefix}_uniformity"] = float(np.sum(hist ** 2))

    return features


def extract_lbp_features(gray: np.ndarray) -> dict:
    """
    Extracts global LBP texture and local 4x4 grid texture features.

    Reason:
    - Real objects usually have organic varied texture.
    - Screen recaptures may show repetitive rigid texture.
    - 4x4 windowing helps distinguish a real laptop in a scene
      from the whole image being a recaptured screen.
    """
    features = {}

    # Smooth slightly to reduce random sensor noise
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    lbp = compute_lbp(gray_blur)
    features.update(lbp_histogram_features(lbp, prefix="lbp_global"))

    h, w = gray_blur.shape
    grid_entropy = []
    grid_uniformity = []

    rows = 4
    cols = 4

    for r in range(rows):
        for c in range(cols):
            y1 = int(r * h / rows)
            y2 = int((r + 1) * h / rows)
            x1 = int(c * w / cols)
            x2 = int((c + 1) * w / cols)

            patch = gray_blur[y1:y2, x1:x2]

            if patch.shape[0] < 10 or patch.shape[1] < 10:
                continue

            patch_lbp = compute_lbp(patch)
            patch_feats = lbp_histogram_features(
                patch_lbp,
                prefix=f"lbp_grid_{r}_{c}"
            )

            features.update(patch_feats)

            grid_entropy.append(patch_feats[f"lbp_grid_{r}_{c}_entropy"])
            grid_uniformity.append(patch_feats[f"lbp_grid_{r}_{c}_uniformity"])

    if grid_entropy:
        features["lbp_grid_entropy_mean"] = float(np.mean(grid_entropy))
        features["lbp_grid_entropy_std"] = float(np.std(grid_entropy))
        features["lbp_grid_uniformity_mean"] = float(np.mean(grid_uniformity))
        features["lbp_grid_uniformity_std"] = float(np.std(grid_uniformity))
    else:
        features["lbp_grid_entropy_mean"] = 0.0
        features["lbp_grid_entropy_std"] = 0.0
        features["lbp_grid_uniformity_mean"] = 0.0
        features["lbp_grid_uniformity_std"] = 0.0

    return features