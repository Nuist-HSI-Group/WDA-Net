#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wavelet subband feature visualization for hyperspectral images (direction-aware).

- Computes SWT (Stationary Wavelet Transform) coefficients per band:
    LL (approx), cH (horizontal detail), cV (vertical detail), cD (diagonal detail)
- Visualizes 4 feature maps similar to typical papers:
    Low-frequency Feature (LL)
    High-frequency Feature H (cH)
    High-frequency Feature V (cV)
    High-frequency Feature D (cD)

Key improvement vs energy-map visualization:
- cH/cV/cD are signed coefficients. We keep sign and use symmetric normalization
  + diverging colormap, so orientation differences become visible.

Supports:
- Loading .mat (bands,pixels) or (pixels,bands) or (H,W,bands) or (bands,H,W)
- Loading .npy (bands,H,W) or (H,W,bands)
- Saving a 2x2 figure and optionally individual PNGs.

Run:
    python wavelet_feature_viz.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pywt

try:
    import scipy.io as sio
except Exception:
    sio = None

# ===================== 用户配置区（运行前修改这里即可） =====================
DATA_TYPE = "mat"        # "mat" or "npy"

# ---- 数据路径 ----
MAT_PATH = "dataset/moffett_dataset.mat"
MAT_KEY = "Y"            # .mat 中的变量名
NPY_PATH = None          # 如果用 npy，这里写路径

# ---- 小波参数 ----
WAVELET = "bior3.3"
LEVEL = 1                # SWT level（目前脚本取 level=1 的系数）
CROP_BORDER = 6   # 建议 4~8，对 bior3.3 用 6 很合适

# ---- 聚合方式 ----
# 对 LL：建议 "mean" 或 "median"；对方向子带：建议 "mean" 或 "median"
AGG = "mean"             # "mean" | "median"

# ---- 可视化参数 ----
CLIP = 99.7              # 分位裁剪，99~99.9 常用
GAMMA_LL = 0.95          # LL增强（接近1即可）
GAMMA_DIR = 0.85         # 方向子带增强（0.6~0.9 常用，越小越突出弱响应）
USE_LOG_DIR = False      # 方向子带是否用 log 增强（一般 False 就够）

# ---- 色图设置 ----
CMAP_LL = "jet"      # LL：viridis/gray/jet
CMAP_DIR = "jet"     # 方向子带：建议 seismic / RdBu / coolwarm（发散色图）

# ---- 输出设置 ----
OUT_DIR = "SWT_image"
PREFIX = "samson"
SAVE_INDIVIDUAL = True   # 是否保存 LL/H/V/D 单张
DPI = 250
# ======================================================================


# ------------------------ 小工具：数据加载与形状处理 ------------------------ #

def crop_border(x2d, k):
    if k <= 0:
        return x2d
    return x2d[k:-k, k:-k]

def _to_bhw(arr: np.ndarray) -> np.ndarray:
    """Convert input to (bands, height, width) if possible."""
    arr = np.asarray(arr)
    if arr.ndim == 3:
        # (bands,H,W) or (H,W,bands)
        if arr.shape[0] < arr.shape[-1] and arr.shape[0] <= 512:
            return arr
        return np.transpose(arr, (2, 0, 1))

    if arr.ndim == 2:
        # (bands, pixels) or (pixels, bands)
        a, b = arr.shape
        if a < b:
            bands, pixels = a, b
            flat = arr
        else:
            bands, pixels = b, a
            flat = arr.T

        h = int(np.sqrt(pixels))
        w = pixels // h
        if h * w != pixels:
            raise ValueError(f"Cannot reshape pixels={pixels} into (H,W). Got h={h}, w={w}.")
        return flat.reshape(bands, h, w)

    raise ValueError(f"Unsupported array shape: {arr.shape}")


def load_hsi(mat_path=None, key=None, npy_path=None):
    """Load hyperspectral data and convert to (bands,H,W)."""
    if npy_path:
        arr = np.load(npy_path)
        return _to_bhw(arr)

    if mat_path:
        if sio is None:
            raise ImportError("scipy is required to load .mat files: pip install scipy")
        mat = sio.loadmat(mat_path)
        if key is None:
            key = next(k for k in mat.keys() if not k.startswith("__"))
        if key not in mat:
            raise KeyError(f"Key '{key}' not found in {mat_path}. Available keys: {list(mat.keys())}")
        arr = mat[key]
        return _to_bhw(arr)

    raise ValueError("Provide either --mat or --npy")


# ------------------------ 核心：SWT系数（保方向） ------------------------ #
def precompute_swt_coeffs(image: np.ndarray, wavelet: str = "bior3.3", level: int = 1):
    """
    返回 SWT 的原始子带系数（保留方向信息）：
    LL(approx), cH(horizontal), cV(vertical), cD(diagonal)
    形状均为 (bands, H, W)
    """
    bands, H, W = image.shape
    LL = np.zeros((bands, H, W), dtype=np.float32)
    cH = np.zeros((bands, H, W), dtype=np.float32)
    cV = np.zeros((bands, H, W), dtype=np.float32)
    cD = np.zeros((bands, H, W), dtype=np.float32)

    for b in range(bands):
        x = image[b].astype(np.float32)

        # SWT 对偶数尺寸更稳：奇数尺寸 padding
        oh, ow = x.shape
        padded = False
        if (oh % 2 != 0) or (ow % 2 != 0):
            x = np.pad(x, [(0, oh % 2), (0, ow % 2)], mode="edge")
            padded = True

        coeffs = pywt.swt2(x, wavelet, level=level)
        # 取最高层（level=1时就是 coeffs[0]）
        approx, (ch, cv, cd) = coeffs[-1]

        if padded:
            approx = approx[:oh, :ow]
            ch = ch[:oh, :ow]
            cv = cv[:oh, :ow]
            cd = cd[:oh, :ow]

        LL[b] = approx
        cH[b] = ch
        cV[b] = cv
        cD[b] = cd

    return LL, cH, cV, cD


# ------------------------ 聚合：把 (bands,H,W) -> (H,W) ------------------------ #
def aggregate_bands(x: np.ndarray, mode: str = "mean") -> np.ndarray:
    """
    x: (bands,H,W)
    输出: (H,W)
    """
    if mode == "mean":
        return np.mean(x, axis=0)
    if mode == "median":
        return np.median(x, axis=0)
    raise ValueError("AGG must be 'mean' or 'median'")


# ------------------------ 归一化：LL 与方向子带分别处理 ------------------------ #
def normalize_ll(x2d: np.ndarray, clip_percent: float = 99.7, gamma: float = 0.95, eps: float = 1e-8):
    """
    LL 结构图：非负显示更直观 -> [0,1]
    """
    x = x2d.astype(np.float32)
    # 用分位裁剪抑制极端值
    lo = np.percentile(x, 100 - clip_percent)
    hi = np.percentile(x, clip_percent)
    if hi - lo < eps:
        return np.zeros_like(x, dtype=np.float32)
    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo)
    x = np.power(x, gamma)
    return x


def normalize_signed_dir(x2d: np.ndarray, clip_percent: float = 99.7, gamma: float = 0.85, use_log: bool = False, eps: float = 1e-8):
    """
    方向子带：保留正负符号，归一化到 [-1,1]，用于发散色图显示。
    """
    x = x2d.astype(np.float32)

    s = np.percentile(np.abs(x), clip_percent)
    s = max(float(s), eps)
    x = np.clip(x, -s, s) / s  # [-1,1]

    # 增强弱响应（保符号）
    if use_log:
        # 对称 log：sign(x) * log(1+k|x|) / log(1+k)
        k = 10.0
        x = np.sign(x) * (np.log1p(k * np.abs(x)) / np.log1p(k))
    else:
        x = np.sign(x) * (np.abs(x) ** gamma)

    return x


# ------------------------ 绘图保存 ------------------------ #
def save_2x2(maps, titles, cmaps, out_path, dpi=250):
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), dpi=dpi)
    axes = axes.ravel()
    for ax, m, t, c in zip(axes, maps, titles, cmaps):
        if c in ["seismic", "RdBu", "coolwarm", "bwr"]:
            ax.imshow(m, cmap=c, vmin=-1.0, vmax=1.0)  # 方向图：对称范围
        else:
            ax.imshow(m, cmap=c, vmin=0.0, vmax=1.0)   # LL：0~1
        ax.set_title(t, fontsize=12)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    # 1) Load data
    if DATA_TYPE == "mat":
        hsi = load_hsi(mat_path=MAT_PATH, key=MAT_KEY)
    elif DATA_TYPE == "npy":
        hsi = load_hsi(npy_path=NPY_PATH)
    else:
        raise ValueError("DATA_TYPE must be 'mat' or 'npy'")

    bands, H, W = hsi.shape
    print(f"Loaded HSI: bands={bands}, H={H}, W={W}")

    # 2) SWT coefficients (direction-aware)
    LL_b, cH_b, cV_b, cD_b = precompute_swt_coeffs(hsi, wavelet=WAVELET, level=LEVEL)

    # 3) Aggregate across spectral bands -> 2D maps
    LL2 = crop_border(aggregate_bands(LL_b, mode=AGG), CROP_BORDER)
    H2 = crop_border(aggregate_bands(cH_b, mode=AGG), CROP_BORDER)
    V2 = crop_border(aggregate_bands(cV_b, mode=AGG), CROP_BORDER)
    D2 = crop_border(aggregate_bands(cD_b, mode=AGG), CROP_BORDER)

    # 4) Normalize
    LLn = normalize_ll(LL2, clip_percent=CLIP, gamma=GAMMA_LL)
    Hn  = normalize_signed_dir(H2, clip_percent=CLIP, gamma=GAMMA_DIR, use_log=USE_LOG_DIR)
    Vn  = normalize_signed_dir(V2, clip_percent=CLIP, gamma=GAMMA_DIR, use_log=USE_LOG_DIR)
    Dn  = normalize_signed_dir(D2, clip_percent=CLIP, gamma=GAMMA_DIR, use_log=USE_LOG_DIR)

    # 5) Save
    os.makedirs(OUT_DIR, exist_ok=True)

    out_path = os.path.join(OUT_DIR, f"{PREFIX}_LL_H_V_D.png")
    titles = [
        "Low-frequency Feature (LL)",
        "High-frequency Feature H (cH)",
        "High-frequency Feature V (cV)",
        "High-frequency Feature D (cD)",
    ]
    maps = [LLn, Hn, Vn, Dn]
    cmaps = [CMAP_LL, CMAP_DIR, CMAP_DIR, CMAP_DIR]

    save_2x2(maps, titles, cmaps, out_path, dpi=DPI)
    print(f"Saved 2x2 feature figure to: {out_path}")

    if SAVE_INDIVIDUAL:
        # LL
        p_ll = os.path.join(OUT_DIR, f"{PREFIX}_LL.png")
        plt.imsave(p_ll, LLn, cmap=CMAP_LL, vmin=0.0, vmax=1.0)
        print(f"Saved LL to: {p_ll}")

        # H/V/D (signed, diverging)
        for name, m in [("H", Hn), ("V", Vn), ("D", Dn)]:
            pth = os.path.join(OUT_DIR, f"{PREFIX}_{name}.png")
            plt.imsave(pth, m, cmap=CMAP_DIR, vmin=0.0, vmax=1.0)
            print(f"Saved {name} to: {pth}")


if __name__ == "__main__":
    main()

