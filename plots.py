import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

import matplotlib
matplotlib.use("Agg")

import numpy as np
from matplotlib import pyplot as plt
import os


def plot_abundance(ground_truth, estimated,index,em, save_dir):
    plt.figure(figsize=(12, 6), dpi=150)
    import matplotlib
    print(matplotlib.get_backend())

    for i in range(em):
        plt.subplot(2, em, i + 1)
        plt.imshow(ground_truth[:, :, index[i]], cmap='jet',vmin=0, vmax=1)

    for i in range(em):
        plt.subplot(2, em, em + i + 1)
        plt.imshow(estimated[:, :, i], cmap='jet', vmin=0, vmax=1)
    # plt.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.95,
    #                     wspace=0.06, hspace=0.08)

    plt.savefig(save_dir + "abundance.png")



def plot_endmembers(target, pred,index,em, save_dir):
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    x_length = len(pred[:, 0])

    # 先保存单个端元图片
    for i in range(em):
        # 为每个端元创建新的figure
        plt.figure(figsize=(8, 6), dpi=150)

        # 绘制当前端元
        plt.plot(pred[:, i], label='_nolegend_')
        plt.plot(target[:, index[i]], label='_nolegend_')
        plt.yticks([0, 0.5, 1])

        plt.xticks(fontsize=19)  # x轴刻度字体
        plt.yticks(fontsize=19)  # y轴刻度字体
        # 设置字体大小
        plt.rc('xtick', labelsize=20)
        plt.rc('ytick', labelsize=20)

        # 设置坐标轴范围
        plt.xlim(left=0)
        plt.ylim(bottom=0)

        # 设置坐标轴标签
        plt.xlabel('Bands', fontsize=22)
        plt.ylabel('Normalized Reflectance', fontsize=22)

        # 设置纵横比
        plt.gca().set_aspect(x_length)

        # 保存单个端元图片
        save_path = os.path.join(save_dir, f"endmember_{i + 1}")
        plt.savefig(save_path + ".png", bbox_inches='tight', dpi=150, facecolor='white')
        print(f"已保存: {save_path}.png")
        # 关闭当前figure，避免内容叠加
        plt.close()


