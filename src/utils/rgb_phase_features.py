import numpy as np
import cv2


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float32)
    b = b.ravel().astype(np.float32)

    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0

    return float(np.corrcoef(a, b)[0, 1])


def extract_rgb_phase_features(img_bgr: np.ndarray) -> dict:
    """
    Features related to RGB channel behavior.

    Screen photos can show subtle color-channel artifacts due to display
    subpixels and camera sampling. We approximate this by checking how
    similarly RGB channels behave around edges and gradients.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    r = img_rgb[:, :, 0]
    g = img_rgb[:, :, 1]
    b = img_rgb[:, :, 2]

    features = {}

    # Raw channel correlations
    features["corr_rg"] = _safe_corr(r, g)
    features["corr_rb"] = _safe_corr(r, b)
    features["corr_gb"] = _safe_corr(g, b)

    # Gradient magnitude per channel
    def grad_mag(channel: np.ndarray) -> np.ndarray:
        gx = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(gx * gx + gy * gy)

    gr = grad_mag(r)
    gg = grad_mag(g)
    gb = grad_mag(b)

    features["grad_corr_rg"] = _safe_corr(gr, gg)
    features["grad_corr_rb"] = _safe_corr(gr, gb)
    features["grad_corr_gb"] = _safe_corr(gg, gb)

    # Channel gradient mismatch
    features["grad_mismatch_rg_mean"] = float(np.mean(np.abs(gr - gg)))
    features["grad_mismatch_rb_mean"] = float(np.mean(np.abs(gr - gb)))
    features["grad_mismatch_gb_mean"] = float(np.mean(np.abs(gg - gb)))

    # Chromatic edge ratio:
    # strong color edge but weak luminance edge can be suspicious
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    lum_grad = grad_mag(luminance)

    color_diff = np.sqrt((r - g) ** 2 + (r - b) ** 2 + (g - b) ** 2)
    color_grad = grad_mag(color_diff)

    features["chromatic_edge_mean"] = float(np.mean(color_grad))
    features["chromatic_edge_p95"] = float(np.percentile(color_grad, 95))

    features["chromatic_to_luma_edge_ratio"] = float(
        np.mean(color_grad) / (np.mean(lum_grad) + 1e-8)
    )

    return features