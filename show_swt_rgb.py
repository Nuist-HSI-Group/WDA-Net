import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from pretrain import precompute_wavelet_subbands  # 替换为实际模块路径


def normalize_band(band):
    """将波段数据归一化到[0, 255]范围"""
    min_val = np.min(band)
    max_val = np.max(band)
    if max_val - min_val < 1e-6:  # 避免除零
        return np.zeros_like(band, dtype=np.uint8)
    return ((band - min_val) / (max_val - min_val) * 255).astype(np.uint8)


def wavelet_rgb_visualization(data, mat_path, rgb_bands, a_bands,subband_type='LL', data_key='hsi_data'):
    """
    从.mat文件加载高光谱数据并生成小波子带RGB合成图及红色波段的jet色图（无空白、无标题）
    """
    # 1. 加载.mat文件
    mat_data = sio.loadmat(mat_path)
    hsi_data = mat_data[data_key]  # 提取高光谱数据

    # 2. 解析数据形状
    num_bands, num_pixels = hsi_data.shape
    height = int(np.sqrt(num_pixels))
    width = num_pixels // height
    assert height * width == num_pixels, "像素数无法匹配有效的高宽尺寸"

    # 3. 重塑为(波段数, 高度, 宽度)
    hsi_image = hsi_data.reshape(num_bands, height, width)
    print(f"高光谱数据形状: {hsi_image.shape} (波段数, 高度, 宽度)")

    # 4. 计算小波四子带
    LL, LH, HL, HH = precompute_wavelet_subbands(hsi_image)

    # 5. 选择目标子带
    subband_mapping = {
        'LL': LL,
        'LH': LH,
        'HL': HL,
        'HH': HH
    }
    selected_subband = subband_mapping[subband_type]

    # 6. 提取并归一化指定RGB波段
    if len(rgb_bands) != 3:
        raise ValueError("rgb_bands必须包含3个波段索引")
    r_band = normalize_band(selected_subband[rgb_bands[0]])
    g_band = normalize_band(selected_subband[rgb_bands[1]])
    b_band = normalize_band(selected_subband[rgb_bands[2]])

    # 7. 合成RGB图像（形状：(高度, 宽度, 3)）
    rgb_image = np.stack([r_band, g_band, b_band], axis=-1)

    # 8. 生成RGB图像并保存
    dpi = 100  # 分辨率
    fig_width = width / dpi
    fig_height = height / dpi

    # 保存RGB图像
    plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    ax = plt.Axes(plt.gcf(), [0., 0., 1., 1.])
    ax.set_axis_off()
    plt.gcf().add_axes(ax)
    ax.imshow(rgb_image)
    ax.set_aspect('equal')
    rgb_save_path = f"SWT_image/{data}_{subband_type}_rgb.png"
    plt.savefig(rgb_save_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close()
    print(f"RGB图像已保存至: {rgb_save_path}")

    # 9. 生成红色波段的色图（修复版）
    # 提取红色波段原始数据
    red_band_raw = selected_subband[a_bands]
    # 打印数据统计（排查用）
    print(
        f"红色波段数据统计：min={np.min(red_band_raw):.2f}, max={np.max(red_band_raw):.2f}, mean={np.mean(red_band_raw):.2f}")
    # 归一化到[0,1]（必须步骤）
    # min_val = np.min(red_band_raw)
    # max_val = np.max(red_band_raw)
    # if max_val - min_val < 1e-6:
    #     red_band_norm = np.zeros_like(red_band_raw, dtype=np.float32)
    # else:
    #     red_band_norm = (red_band_raw - min_val) / (max_val - min_val)
    # 绘制色图
    plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    ax_jet = plt.Axes(plt.gcf(), [0., 0., 1., 1.])
    ax_jet.set_axis_off()
    plt.gcf().add_axes(ax_jet)
    # 使用更友好的色图（推荐viridis）
    cmap = 'jet'  # 可选：'viridis'、'cividis'、'plasma'、'inferno'
    im = ax_jet.imshow(red_band_raw, cmap=cmap, vmin=0.0, vmax=1.0)  # 强制映射范围
    ax_jet.set_aspect('equal')
    # 保存色图
    jet_save_path = f"SWT_image/{data}_{subband_type}_red_{cmap}.png"
    plt.savefig(jet_save_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close()
    print(f"红色波段色图已保存至: {jet_save_path}")

    return rgb_image, red_band_raw


# 示例用法
if __name__ == "__main__":
    data = 'samson'
    mat_path = f"{data}_dataset.mat"  # 替换为实际路径
    if data=='moffett':
        rgb_bands = [50, 30, 10]  # 示例波段索引（第一个为红色波段）
    elif data=='samson':
        rgb_bands = [70, 48, 16]
        a_bands = 100
    elif data=='sy30A':
        rgb_bands = [50, 30, 10]
    else:
        rgb_bands = [50, 30, 10]

    wavelet_rgb_visualization(
        data=data,
        mat_path=mat_path,
        rgb_bands=rgb_bands,
        a_bands=a_bands,
        subband_type='LH',
        data_key='Y'  # 替换为.mat文件中实际的变量名
    )