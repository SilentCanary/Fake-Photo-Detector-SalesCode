import numpy as np
import cv2


def extract_edge_features(gray: np.ndarray) -> dict:
    """
    Edge and sharpness features.

    Screen recaptures may be:
    - slightly blurred due to rephotographing
    - unnaturally sharp in screen pixel areas
    - filled with repeated hard edges
    """
    gray_float = gray.astype(np.float32) / 255.0

    features = {}

    lap = cv2.Laplacian(gray_float, cv2.CV_32F)
    features["laplacian_var"] = float(np.var(lap))
    features["laplacian_mean_abs"] = float(np.mean(np.abs(lap)))

    edges = cv2.Canny(gray, 80, 160)
    features["edge_density"] = float(np.mean(edges > 0))

    # Sobel gradients
    gx = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=3)

    grad_mag = np.sqrt(gx * gx + gy * gy)

    features["grad_mean"] = float(np.mean(grad_mag))
    features["grad_std"] = float(np.std(grad_mag))
    features["grad_p95"] = float(np.percentile(grad_mag, 95))

    # Horizontal/vertical dominance can catch grid/scanline-like structures
    features["sobel_x_mean_abs"] = float(np.mean(np.abs(gx)))
    features["sobel_y_mean_abs"] = float(np.mean(np.abs(gy)))
    features["sobel_xy_ratio"] = float(
        np.mean(np.abs(gx)) / (np.mean(np.abs(gy)) + 1e-8)
    )

    return features