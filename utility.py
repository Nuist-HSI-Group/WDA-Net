import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import torch.nn.functional as F
from mpl_toolkits.axes_grid1 import make_axes_locatable



def numpy_MSE(y_true, y_pred):  # 错写成MSE了，实际上是算RMSE，将错就错了
    num_cols = y_pred.shape[0]
    num_rows = y_pred.shape[1]
    diff = y_true - y_pred
    squared_diff = np.square(diff)
    mse = squared_diff.sum() / (num_rows * num_cols)
    rmse = np.sqrt(mse)
    return rmse

from scipy.optimize import linear_sum_assignment  # 匈牙利算法

def numpy_SAD(y_true, y_pred):
    cos = y_pred.dot(y_true) / (np.linalg.norm(y_true) * np.linalg.norm(y_pred))
    if cos > 1.0: cos = 1.0
    return np.arccos(cos)


def order_endmembers(endmembers, endmembersGT):
    num_endmembers = endmembers.shape[1]
    SAD_matrix = np.zeros((num_endmembers, num_endmembers))
    SAD_index = np.zeros(num_endmembers).astype(int)
    SAD_endmember = np.zeros(num_endmembers)

    # 复制数据，避免修改原始数据
    e = endmembers.copy()
    egt = endmembersGT.copy()
    # 计算SAD矩阵
    for i in range(0, num_endmembers):
        for j in range(0, num_endmembers):
            SAD_matrix[i, j] = numpy_SAD(e[:, i], egt[:, j])

    # 一对一匹配端元（修复：处理全NaN情况）
    for i in range(0, num_endmembers):
        # 检查当前行是否全为NaN
        if np.all(np.isnan(SAD_matrix[i, :])):
            print(f"警告：估计端元{i}与所有真实端元的SAD均为NaN，无法匹配")
            SAD_index[i] = -1  # 标记为无效匹配
            SAD_endmember[i] = np.inf
            continue

        # 找到最小SAD对应的真实端元索引
        SAD_index[i] = np.nanargmin(SAD_matrix[i, :])
        SAD_endmember[i] = np.nanmin(SAD_matrix[i, :])

        # 标记已匹配的真实端元（修复：列操作，而非行操作！）
        SAD_matrix[:, SAD_index[i]] = np.inf # 正确：端元矩阵是(bands, K)，每列是一个端元

    return SAD_index, SAD_endmember


def order_abundance(abundance, abundanceGT):
    num_endmembers = abundance.shape[2]
    abundance_matrix = np.zeros((num_endmembers, num_endmembers))
    abundance_index = np.zeros(num_endmembers).astype(int)
    MSE_abundance = np.zeros(num_endmembers)
    a = abundance.copy()
    agt = abundanceGT.copy()
    for i in range(0, num_endmembers):
        for j in range(0, num_endmembers):
            abundance_matrix[i, j] = numpy_MSE(a[:, :, i], agt[:, :, j])

        abundance_index[i] = np.nanargmin(abundance_matrix[i, :])
        MSE_abundance[i] = np.nanmin(abundance_matrix[i, :])
        agt[:, :, abundance_index[i]] = np.inf
    return abundance_index, MSE_abundance



def reconstruct(S, A):
    S = np.reshape(S, (S.shape[0] * S.shape[1], S.shape[2]))
    reconstructed = np.matmul(S, A)
    return reconstructed


# SAD loss of reconstruction
def reconstruction_SADloss(output, target):
    _, band, h, w = output.shape
    output = torch.reshape(output, (band, h * w))
    target = torch.reshape(target, (band, h * w))
    abundance_loss = torch.acos(torch.cosine_similarity(output, target, dim=0))
    abundance_loss = torch.mean(abundance_loss)

    return abundance_loss


class load_data(torch.utils.data.Dataset):
    def __init__(self, img, transform=None):
        self.img = img.float()
        self.transform = transform

    def __getitem__(self, idx):
        return self.img

    def __len__(self):
        return 1



def SAD(y_true, y_pred):
    y_true2 = torch.nn.functional.normalize(y_true, dim=1)
    y_pred2 = torch.nn.functional.normalize(y_pred, dim=1)
    A = torch.mean(y_true2 * y_pred2)
    sad = torch.acos(A)
    return sad