import numpy as np
import cv2


def extract_color_features(img_bgr: np.ndarray) -> dict:
    """
    HSV and brightness features.

    Screen recaptures may show:
    - over-saturated regions
    - clipped highlights due to screen glare
    - unnatural brightness/saturation distribution
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    v = hsv[:, :, 2].astype(np.float32) / 255.0

    features = {}

    features["sat_mean"] = float(np.mean(s))
    features["sat_std"] = float(np.std(s))
    features["sat_p90"] = float(np.percentile(s, 90))
    features["sat_p95"] = float(np.percentile(s, 95))
    features["sat_p99"] = float(np.percentile(s, 99))
    features["sat_high_ratio"] = float(np.mean(s > 0.90))

    features["val_mean"] = float(np.mean(v))
    features["val_std"] = float(np.std(v))
    features["val_p90"] = float(np.percentile(v, 90))
    features["val_p95"] = float(np.percentile(v, 95))
    features["val_p99"] = float(np.percentile(v, 99))

    # Strong glare / clipped white regions
    features["overexposed_ratio"] = float(np.mean(v > 0.97))

    # Bright and low-saturation pixels often correspond to white glare
    glare_mask = (v > 0.90) & (s < 0.25)
    features["glare_like_ratio"] = float(np.mean(glare_mask))

    # Dark border / bezel clue
    dark_mask = v < 0.08
    features["dark_pixel_ratio"] = float(np.mean(dark_mask))

    # Border darkness specifically
    border_width = max(4, img_bgr.shape[0] // 25)

    top = v[:border_width, :]
    bottom = v[-border_width:, :]
    left = v[:, :border_width]
    right = v[:, -border_width:]

    border_pixels = np.concatenate([
        top.ravel(),
        bottom.ravel(),
        left.ravel(),
        right.ravel(),
    ])

    features["border_dark_ratio"] = float(np.mean(border_pixels < 0.08))
    features["border_brightness_mean"] = float(np.mean(border_pixels))

    return features