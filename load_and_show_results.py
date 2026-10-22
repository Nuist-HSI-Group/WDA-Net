# -*- coding: utf-8 -*-
"""
加载训练好的 MSDCN 权重并直接显示结果图（不保存）
同时显示模型参数量、MACs 和 FLOPs

这个版本不需要命令行参数，直接在下面修改路径后运行即可。
"""

import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

import torch

from model import MSDCN
from pretrain import precompute_wavelet_subbands

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# =========================================================
# 直接在这里修改
# =========================================================
data = "urban"
CHECKPOINT_PATH = f"checkpoints/{data}.pth"
MAT_PATH = f"dataset/{data}_dataset.mat"   # 如果 checkpoint 里已有 mat_path，也建议这里直接写，最稳妥
DEVICE = "cuda"                            # "auto" / "cpu" / "cuda"
SHOW_PARAM_DETAIL = True                   # True 显示每一层参数；False 只显示总参数量
# =========================================================


def load_data(mat_path, h, w, device):
    mat_data = sio.loadmat(mat_path)
    Y = mat_data["Y"]  # (bands, H*W)
    mixed_spectra_np = Y.reshape(Y.shape[0], h, w)  # (bands, H, W)

    LL_np, LH_np, HL_np, HH_np = precompute_wavelet_subbands(mixed_spectra_np)

    LL = torch.tensor(LL_np, dtype=torch.float32, device=device).unsqueeze(0)
    LH = torch.tensor(LH_np, dtype=torch.float32, device=device).unsqueeze(0)
    HL = torch.tensor(HL_np, dtype=torch.float32, device=device).unsqueeze(0)
    HH = torch.tensor(HH_np, dtype=torch.float32, device=device).unsqueeze(0)
    original = torch.tensor(mixed_spectra_np, dtype=torch.float32, device=device).unsqueeze(0)

    true_endmembers = mat_data.get("M", None)
    true_abundances = mat_data.get("A", None)

    return LL, LH, HL, HH, original, true_endmembers, true_abundances


def normalize_curve(x):
    x = np.asarray(x, dtype=np.float32)
    xmax = np.max(x)
    if xmax <= 1e-8:
        return x
    return x / xmax


def format_number(num):
    """
    格式化大数字
    """
    if num >= 1e12:
        return f"{num / 1e12:.4f} T"
    elif num >= 1e9:
        return f"{num / 1e9:.4f} G"
    elif num >= 1e6:
        return f"{num / 1e6:.4f} M"
    elif num >= 1e3:
        return f"{num / 1e3:.4f} K"
    else:
        return f"{num:.0f}"


def print_model_parameters(model, show_detail=True):
    """
    展示模型参数量
    show_detail=True 时显示每一层参数名、shape 和参数数量
    """
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    non_trainable_params = total_params - trainable_params

    print("\n================ 模型参数统计 ================")
    print(f"Total params:         {total_params:,}  ({total_params / 1e6:.6f} M)")
    print(f"Trainable params:     {trainable_params:,}  ({trainable_params / 1e6:.6f} M)")
    print(f"Non-trainable params: {non_trainable_params:,}  ({non_trainable_params / 1e6:.6f} M)")

    if show_detail:
        print("\n---------------- 每层参数 ----------------")
        print(f"{'Parameter name':60s} {'Shape':25s} {'Param #':>15s}")
        print("-" * 105)

        for name, param in model.named_parameters():
            print(
                f"{name:60s} "
                f"{str(list(param.shape)):25s} "
                f"{param.numel():15,}"
            )

    print("==========================================\n")


def print_model_flops(model, original, LL, LH, HL, HH):
    """
    使用 thop 统计模型计算量

    注意：
    1. thop.profile 返回的是 MACs，不是严格 FLOPs。
    2. 常见论文/代码中经常用 FLOPs ≈ 2 * MACs。
    3. 如果你的论文中 FLOPs 定义为 MACs，则直接使用 MACs 那一行即可。
    """
    try:
        from thop import profile
    except ImportError:
        print("\n================ 模型计算量统计 ================")
        print("未安装 thop，无法统计 FLOPs / MACs。")
        print("请先运行：pip install thop")
        print("===============================================\n")
        return

    model.eval()

    with torch.no_grad():
        macs, params_from_thop = profile(
            model,
            inputs=(original, LL, LH, HL, HH),
            verbose=False
        )

    flops = 2 * macs

    print("\n================ 模型计算量统计 ================")
    print(f"Input original shape: {list(original.shape)}")
    print(f"Input LL shape:       {list(LL.shape)}")
    print(f"Input LH shape:       {list(LH.shape)}")
    print(f"Input HL shape:       {list(HL.shape)}")
    print(f"Input HH shape:       {list(HH.shape)}")
    print("-----------------------------------------------")
    print(f"MACs:   {macs:,.0f}  ({format_number(macs)}MACs)")
    print(f"FLOPs:  {flops:,.0f}  ({format_number(flops)}FLOPs)")
    print(f"Params: {params_from_thop:,.0f}  ({format_number(params_from_thop)})")
    print("===============================================\n")


def plot_abundance_maps(estimated_abundances, true_abundances=None, abundance_index=None):
    """
    estimated_abundances: [H, W, K]
    true_abundances: [H, W, K] or None
    abundance_index: 重排索引
    """
    _, _, k = estimated_abundances.shape

    if true_abundances is not None and abundance_index is not None:
        est = estimated_abundances[:, :, abundance_index]

        fig, axes = plt.subplots(2, k, figsize=(4 * k, 7))

        if k == 1:
            axes = np.array(axes).reshape(2, 1)

        for i in range(k):
            ax1 = axes[0, i]
            im1 = ax1.imshow(true_abundances[:, :, i], cmap="jet")
            ax1.set_title(f"True abundance {i + 1}")
            ax1.axis("off")
            fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

            ax2 = axes[1, i]
            im2 = ax2.imshow(est[:, :, i], cmap="jet")
            ax2.set_title(f"Estimated abundance {i + 1}")
            ax2.axis("off")
            fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        plt.suptitle("Abundance maps", fontsize=14)
        plt.tight_layout()
        plt.show()

    else:
        fig, axes = plt.subplots(1, k, figsize=(4 * k, 4))

        if k == 1:
            axes = [axes]

        for i in range(k):
            im = axes[i].imshow(estimated_abundances[:, :, i], cmap="jet")
            axes[i].set_title(f"Estimated abundance {i + 1}")
            axes[i].axis("off")
            fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

        plt.suptitle("Estimated abundance maps", fontsize=14)
        plt.tight_layout()
        plt.show()


def plot_endmember_curves(estimated_endmembers, true_endmembers=None, endmember_index=None):
    """
    estimated_endmembers: [bands, K]
    true_endmembers: [bands, K] or None
    endmember_index: 重排索引
    """
    _, k = estimated_endmembers.shape

    if true_endmembers is not None and endmember_index is not None:
        est = estimated_endmembers[:, endmember_index]

        fig, axes = plt.subplots(1, k, figsize=(4.5 * k, 4), squeeze=False)
        axes = axes[0]

        for i in range(k):
            axes[i].plot(normalize_curve(true_endmembers[:, i]), label="True")
            axes[i].plot(normalize_curve(est[:, i]), label="Estimated")
            axes[i].set_title(f"Endmember {i + 1}")
            axes[i].set_xlabel("Band")
            axes[i].set_ylabel("Normalized reflectance")
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()

        plt.suptitle("Endmember curves", fontsize=14)
        plt.tight_layout()
        plt.show()

    else:
        fig, axes = plt.subplots(1, k, figsize=(4.5 * k, 4), squeeze=False)
        axes = axes[0]

        for i in range(k):
            axes[i].plot(normalize_curve(estimated_endmembers[:, i]), label="Estimated")
            axes[i].set_title(f"Endmember {i + 1}")
            axes[i].set_xlabel("Band")
            axes[i].set_ylabel("Normalized reflectance")
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()

        plt.suptitle("Estimated endmember curves", fontsize=14)
        plt.tight_layout()
        plt.show()


def build_model_from_checkpoint(ckpt, device):
    config = ckpt["config"]

    model = MSDCN(
        bands=config["bands"],
        height=config["height"],
        width=config["width"],
        n_endmembers=config["n_endmembers"],
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    return model, config


def main():
    if DEVICE == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if DEVICE == "cuda" and not torch.cuda.is_available():
            print("警告: CUDA 不可用，自动切换到 CPU")
            device = torch.device("cpu")
        else:
            device = torch.device(DEVICE)

    print(f"使用设备: {device}")

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"找不到权重文件: {CHECKPOINT_PATH}")

    if not os.path.exists(MAT_PATH):
        raise FileNotFoundError(f"找不到数据文件: {MAT_PATH}")

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model, config = build_model_from_checkpoint(ckpt, device)

    print(f"加载权重: {CHECKPOINT_PATH}")
    print(f"加载数据: {MAT_PATH}")

    # 参数量统计
    print_model_parameters(model, show_detail=SHOW_PARAM_DETAIL)

    LL, LH, HL, HH, original, true_endmembers, true_abundances = load_data(
        MAT_PATH,
        config["height"],
        config["width"],
        device
    )

    # FLOPs / MACs 统计
    print_model_flops(model, original, LL, LH, HL, HH)

    with torch.no_grad():
        estimated_abundances, estimated_endmembers, _ = model(original, LL, LH, HL, HH)

    estimated_abundances_np = estimated_abundances.squeeze(0).detach().cpu().numpy()   # [K, H, W]
    estimated_abundances_reshaped = estimated_abundances_np.transpose(1, 2, 0)         # [H, W, K]
    estimated_endmembers_np = estimated_endmembers.squeeze(0).detach().cpu().numpy().T # [bands, K]

    if true_endmembers is not None and true_abundances is not None:
        from utility import order_abundance, order_endmembers

        h, w = config["height"], config["width"]
        bands = config["bands"]

        true_abundances_reshaped = true_abundances.T.reshape(h, w, -1)
        true_endmembers_t = true_endmembers if true_endmembers.shape[0] == bands else true_endmembers.T

        abundance_index, mse_values = order_abundance(
            estimated_abundances_reshaped,
            true_abundances_reshaped
        )

        endmember_index, sad_values = order_endmembers(
            estimated_endmembers_np,
            true_endmembers_t
        )

        print("\n各个端元的 SAD：")
        for i in range(config["n_endmembers"]):
            print(f"端元 {i + 1}: {float(sad_values[i]):.6f}")

        print("\n各个丰度的 MSE：")
        for i in range(config["n_endmembers"]):
            print(f"丰度 {i + 1}: {float(mse_values[i]):.6f}")

        print("\nabundance_index:", abundance_index)
        print("endmember_index:", endmember_index)

        plot_abundance_maps(
            estimated_abundances_reshaped,
            true_abundances=true_abundances_reshaped,
            abundance_index=abundance_index,
        )

        plot_endmember_curves(
            estimated_endmembers_np,
            true_endmembers=true_endmembers_t,
            endmember_index=endmember_index,
        )

    else:
        print("\n未检测到真值 M / A，仅显示估计结果。")

        plot_abundance_maps(estimated_abundances_reshaped)
        plot_endmember_curves(estimated_endmembers_np)


if __name__ == "__main__":
    main()
