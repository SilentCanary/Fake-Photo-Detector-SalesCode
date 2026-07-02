from pathlib import Path
import numpy as np

from src.utils.image_io import load_image_bgr, resize_for_features, to_gray
from src.utils.fft_features import extract_fft_features
from src.utils.lbp_features import extract_lbp_features
from src.utils.color_features import extract_color_features
from src.utils.rgb_phase_features import extract_rgb_phase_features
from src.utils.edge_features import extract_edge_features


def extract_all_features(image_path: str | Path) -> dict:
    """
    Master feature extraction function.

    Input:
        image_path

    Output:
        dict of feature_name -> value
    """
    image_path = Path(image_path)

    img_bgr = load_image_bgr(image_path)
    img_bgr = resize_for_features(img_bgr, size=(256, 256))

    gray = to_gray(img_bgr)

    features = {}

    features["image_path"] = str(image_path)

    features.update(extract_fft_features(gray))
    features.update(extract_lbp_features(gray))
    features.update(extract_color_features(img_bgr))
    features.update(extract_rgb_phase_features(img_bgr))
    features.update(extract_edge_features(gray))

    # Safety: replace NaN/inf with 0
    for key, value in list(features.items()):
        if key == "image_path":
            continue

        if value is None:
            features[key] = 0.0
        elif isinstance(value, (float, int, np.number)):
            if not np.isfinite(value):
                features[key] = 0.0
            else:
                features[key] = float(value)

    return features