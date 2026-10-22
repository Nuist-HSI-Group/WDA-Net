import numpy as np
import pywt
import torch.nn as nn

def precompute_wavelet_subbands(image, wavelet='bior3.3'):
    """预处理时计算小波四子带"""
    bands, height, width = image.shape
    LL = np.zeros_like(image)
    LH = np.zeros_like(image)
    HL = np.zeros_like(image)
    HH = np.zeros_like(image)

    for b in range(bands):
        signal = image[b]
        # 处理奇数尺寸
        is_padded = False
        original_shape = signal.shape
        if (original_shape[0] % 2 != 0) or (original_shape[1] % 2 != 0):
            pad_height = 1 if original_shape[0] % 2 != 0 else 0
            pad_width = 1 if original_shape[1] % 2 != 0 else 0
            pad_widths = [(0, pad_height), (0, pad_width)]
            signal = np.pad(signal, pad_widths, mode='edge')
            is_padded = True

        # 执行SWT
        coeffs = pywt.swt2(signal, wavelet, level=1)
        approx, (h1, v1, d1) = coeffs[0]

        # 重构四子带
        coeffs_ll = [(approx, (np.zeros_like(h1), np.zeros_like(v1), np.zeros_like(d1)))]
        LL_b = pywt.iswt2(coeffs_ll, wavelet)

        coeffs_lh = [(np.zeros_like(approx), (h1, np.zeros_like(v1), np.zeros_like(d1)))]
        LH_b = pywt.iswt2(coeffs_lh, wavelet)

        coeffs_hl = [(np.zeros_like(approx), (np.zeros_like(h1), v1, np.zeros_like(d1)))]
        HL_b = pywt.iswt2(coeffs_hl, wavelet)

        coeffs_hh = [(np.zeros_like(approx), (np.zeros_like(h1), np.zeros_like(v1), d1))]
        HH_b = pywt.iswt2(coeffs_hh, wavelet)

        # 恢复原始尺寸
        if is_padded:
            LL_b = LL_b[:original_shape[0], :original_shape[1]]
            LH_b = LH_b[:original_shape[0], :original_shape[1]]
            HL_b = HL_b[:original_shape[0], :original_shape[1]]
            HH_b = HH_b[:original_shape[0], :original_shape[1]]

        LL[b] = LL_b
        LH[b] = LH_b
        HL[b] = HL_b
        HH[b] = HH_b

    return LL, LH, HL, HH

# 示例：预处理单个样本
# image = np.load('hyperspectral_image.npy')  # 形状：(bands, height, width)
# LL, LH, HL, HH = precompute_wavelet_subbands(image)
# np.savez('wavelet_subbands.npz', LL=LL, LH=LH, HL=HL, HH=HH)
# def weights_init(net):
#     """针对MSDCN模型的精确卷积层初始化"""
#     # --------------------------
#     #  1. conv_LL 模块（注意：第4层后是LeakyReLU(0.1)，其他是0.2）
#     # --------------------------
#     # conv_LL[0]: 3×3卷积，后接LeakyReLU(0.2)
#     nn.init.kaiming_normal_(net.conv_LL[0].weight)
#     # conv_LL[4]: 3×3卷积，后接LeakyReLU(0.1)
#     nn.init.kaiming_normal_(net.conv_LL[4].weight)
#     # conv_LL[8]: 3×3卷积，直接输出（无激活，非线性设为linear）
#     nn.init.kaiming_normal_(net.conv_LL[8].weight)
#
#     # --------------------------
#     #  2. conv_LH 模块（所有卷积后接LeakyReLU(0.2)）
#     # --------------------------
#     nn.init.kaiming_normal_(net.conv_LH[0].weight)
#     nn.init.kaiming_normal_(net.conv_LH[4].weight)
#     nn.init.kaiming_normal_(net.conv_LH[8].weight)
#
#     # --------------------------
#     #  3. conv_HL 模块（所有卷积后接LeakyReLU(0.2)）
#     # --------------------------
#     nn.init.kaiming_normal_(net.conv_HL[0].weight)
#     nn.init.kaiming_normal_(net.conv_HL[4].weight)
#     nn.init.kaiming_normal_(net.conv_HL[8].weight)
#
#     # --------------------------
#     #  4. conv_HH 模块（所有卷积后接LeakyReLU(0.2)）
#     # --------------------------
#     nn.init.kaiming_normal_(net.conv_HH[0].weight)
#     nn.init.kaiming_normal_(net.conv_HH[4].weight)
#     nn.init.kaiming_normal_(net.conv_HH[8].weight)
#
#     # # --------------------------
#     # #  5. fusion 卷积（直接输出，无激活）
#     # # --------------------------
#     # nn.init.kaiming_normal_(net.fusion.weight, a=0, mode='fan_in', nonlinearity='linear')
#     # if net.fusion.bias is not None:
#     #     nn.init.constant_(net.fusion.bias, 0)  # 偏置初始化为0
#     #
#     # # --------------------------
#     # #  6. smooth 模块（所有卷积后无激活）
#     # # --------------------------
#     nn.init.kaiming_normal_(net.smooth[0].weight)
#     # nn.init.kaiming_normal_(net.smooth[1].weight, a=0, mode='fan_in', nonlinearity='linear')
#     # nn.init.kaiming_normal_(net.smooth[2].weight, a=0, mode='fan_in', nonlinearity='linear')

# import torch.nn as nn
#
# def _init_convblock(block):
#     # block 是 ConvBlock，真正的卷积层在 block.net 里
#     nn.init.kaiming_normal_(block.net[0].weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
#     nn.init.kaiming_normal_(block.net[4].weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
#     nn.init.kaiming_normal_(block.net[8].weight, a=0.0, mode='fan_in', nonlinearity='linear')
#
# def weights_init(net):
#     _init_convblock(net.conv_LL)
#     _init_convblock(net.conv_LH)
#     _init_convblock(net.conv_HL)
#     _init_convblock(net.conv_HH)
#
#     nn.init.kaiming_normal_(net.fusion.weight, a=0.0, mode='fan_in', nonlinearity='linear')
#     if net.fusion.bias is not None:
#         nn.init.constant_(net.fusion.bias, 0)
#
#     nn.init.kaiming_normal_(net.smooth[0].weight, a=0.0, mode='fan_in', nonlinearity='linear')
#     nn.init.kaiming_normal_(net.smooth[1].weight, a=0.0, mode='fan_in', nonlinearity='linear')
#
#     # 端元解码器初始化成正值，更适合后续端元学习
#     nn.init.uniform_(net.decoder.weight, a=0.01, b=0.1)
