"""
Native-patch recapture features.

These features are computed on 256x256 crops from the ORIGINAL image resolution,
not on the globally resized image. This preserves high-frequency moire/screen-grid
signals that can get weakened by resizing the whole image first.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter, ImageOps

PATCH_SIZE = 256
MAX_PATCHES = 10
SAFETY_SIDE = 6000
RADIAL_BINS = 40

# Fixed deterministic sampling. This keeps build_features_csv and predict stable.
RNG_SEED = 0
_GEOMETRY_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}


def _load_rgb_native(image_path: str | Path) -> np.ndarray:
    """Load image as RGB float32, preserving normal phone resolution."""
    img = Image.open(image_path).convert("RGB")
    img = ImageOps.exif_transpose(img)

    w, h = img.size
    scale = max(w, h) / SAFETY_SIDE
    if scale > 1.0:
        img = img.resize((int(w / scale), int(h / scale)), Image.LANCZOS)

    return np.asarray(img, dtype=np.float32)


def _luma(rgb: np.ndarray) -> np.ndarray:
    return rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _patch_coords(h: int, w: int) -> list[tuple[int, int, int, int]]:
    """Return deterministic native-resolution patch coordinates."""
    p = min(PATCH_SIZE, h, w)
    if p < 32:
        return [(0, 0, h, w)]

    ys = list(range(0, max(1, h - p + 1), p))
    xs = list(range(0, max(1, w - p + 1), p))

    if ys[-1] != h - p:
        ys.append(h - p)
    if xs[-1] != w - p:
        xs.append(w - p)

    coords = [(y, x, y + p, x + p) for y in ys for x in xs]

    if len(coords) > MAX_PATCHES:
        rng = np.random.default_rng(RNG_SEED)
        idx = rng.choice(len(coords), MAX_PATCHES, replace=False)
        coords = [coords[i] for i in idx]

    return coords


def _geometry(shape: tuple[int, int]):
    cached = _GEOMETRY_CACHE.get(shape)
    if cached is not None:
        return cached

    h, w = shape
    cy, cx = h / 2.0, w / 2.0
    y, x = np.indices((h, w))
    rr = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    r_idx = (rr / (rr.max() + 1e-8) * (RADIAL_BINS - 1)).astype(np.int32)
    r_flat = r_idx.ravel()
    cnt = np.maximum(np.bincount(r_flat, minlength=RADIAL_BINS), 1)

    win = np.hanning(h)[:, None] * np.hanning(w)[None, :]

    # Ignore DC neighbourhood and frequency axes. JPEG/blocking often dominates axes.
    axis_mask = np.ones((h, w), dtype=bool)
    cy_i, cx_i, band = h // 2, w // 2, 6
    axis_mask[max(0, cy_i - band) : min(h, cy_i + band + 1), :] = False
    axis_mask[:, max(0, cx_i - band) : min(w, cx_i + band + 1)] = False

    cached = (r_idx, r_flat, cnt, win, axis_mask)
    _GEOMETRY_CACHE[shape] = cached
    return cached


def _radial_profile(mag: np.ndarray, r_flat: np.ndarray, cnt: np.ndarray) -> np.ndarray:
    return np.bincount(r_flat, mag.ravel(), minlength=RADIAL_BINS) / cnt


def _fft_patch_features(gray: np.ndarray) -> np.ndarray:
    r_idx, r_flat, cnt, win, axis_mask = _geometry(gray.shape)

    g = (gray - gray.mean()) * win
    fft = np.fft.fftshift(np.fft.fft2(g))
    logmag = np.log1p(np.abs(fft))

    prof = _radial_profile(logmag, r_flat, cnt)
    total = prof[1:].sum() + 1e-8

    low = prof[1 : RADIAL_BINS // 4].sum()
    mid = prof[RADIAL_BINS // 4 : RADIAL_BINS // 2].sum()
    high = prof[RADIAL_BINS // 2 :].sum()

    # Radial whitening: periodic moire peaks stand out above natural 1/f falloff.
    expected = prof[np.clip(r_idx, 0, RADIAL_BINS - 1)]
    whitened = logmag / (expected + 1e-6)
    vals = whitened[axis_mask]

    if vals.size == 0:
        vals = whitened.ravel()

    peak = float(vals.max())
    p999 = float(np.percentile(vals, 99.9))
    n_peaks = float((vals > (vals.mean() + 5.0 * vals.std())).sum())
    prof_cv = float(prof[1:].std() / (prof[1:].mean() + 1e-6))

    return np.array(
        [
            low / total,
            mid / total,
            high / total,
            (high + mid) / (low + 1e-6),
            peak,
            p999,
            np.log1p(n_peaks),
            prof_cv,
        ],
        dtype=np.float32,
    )


def _residual_patch_features(gray: np.ndarray) -> np.ndarray:
    im = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8))
    blur = np.asarray(im.filter(ImageFilter.GaussianBlur(radius=1.5)), dtype=np.float32)
    residual = gray - blur

    r = residual.ravel()
    std = float(r.std())
    mad = float(np.mean(np.abs(r)))
    mean = float(r.mean())
    sigma = float(r.std()) + 1e-6
    kurt = float(np.mean(((r - mean) / sigma) ** 4) - 3.0)

    # Periodic screen grid can create secondary autocorrelation peaks.
    crop = residual[:128, :128] if min(residual.shape) >= 128 else residual
    centered = crop - crop.mean()
    fft = np.fft.fft2(centered)
    ac = np.fft.fftshift(np.fft.ifft2(fft * np.conj(fft)).real)
    ac = ac / (ac.max() + 1e-8)

    cy, cx = ac.shape[0] // 2, ac.shape[1] // 2
    ac[max(0, cy - 3) : min(ac.shape[0], cy + 4), max(0, cx - 3) : min(ac.shape[1], cx + 4)] = 0.0
    ac_peak = float(ac.max())

    return np.array([std, mad, kurt, ac_peak], dtype=np.float32)


def _color_patch_features(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    sat = (mx - mn) / (mx + 1e-6)

    clipped = float(np.mean(mx > 250))
    dark = float(np.mean(mx < 12))

    # Unique quantized colors. Lower ratio can indicate display/posterization.
    q = (rgb[::2, ::2] / 4).astype(np.int32)
    codes = (q[..., 0] << 12) | (q[..., 1] << 6) | q[..., 2]
    uniq_ratio = float(np.unique(codes).size / max(1, codes.size))

    return np.array(
        [
            float(sat.mean()),
            float(sat.std()),
            clipped,
            dark,
            uniq_ratio,
        ],
        dtype=np.float32,
    )


def _patch_vector(rgb_patch: np.ndarray) -> np.ndarray:
    gray = _luma(rgb_patch)
    return np.concatenate(
        [
            _fft_patch_features(gray),
            _residual_patch_features(gray),
            _color_patch_features(rgb_patch),
        ]
    ).astype(np.float32)


_BASE_NAMES = (
    [
        f"native_fft_{name}"
        for name in [
            "low_ratio",
            "mid_ratio",
            "high_ratio",
            "high_mid_to_low",
            "whitened_peak",
            "whitened_p999",
            "num_periodic_peaks_log",
            "radial_profile_cv",
        ]
    ]
    + [
        f"native_residual_{name}"
        for name in ["std", "mean_abs", "kurtosis", "autocorr_peak"]
    ]
    + [
        f"native_color_{name}"
        for name in ["sat_mean", "sat_std", "clipped_ratio", "dark_ratio", "unique_color_ratio"]
    ]
)

FEATURE_NAMES = [f"{stat}__{name}" for stat in ("mean", "max", "p90") for name in _BASE_NAMES]


def extract_native_patch_features(image_path: str | Path) -> dict[str, float]:
    """Return aggregated native-patch feature dictionary for one image."""
    rgb = _load_rgb_native(image_path)
    h, w = rgb.shape[:2]

    patch_vectors = []
    for y0, x0, y1, x1 in _patch_coords(h, w):
        patch_vectors.append(_patch_vector(rgb[y0:y1, x0:x1]))

    values = np.stack(patch_vectors, axis=0)
    agg = np.concatenate(
        [
            values.mean(axis=0),
            values.max(axis=0),
            np.percentile(values, 90, axis=0),
        ]
    ).astype(np.float32)

    agg = np.nan_to_num(agg, nan=0.0, posinf=0.0, neginf=0.0)
    return {name: float(value) for name, value in zip(FEATURE_NAMES, agg)}
