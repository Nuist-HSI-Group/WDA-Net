import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io as sio

from model import MSDCN
from pretrain import *
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
def sad_loss(original, reconstructed, eps=1e-8):
    """
    光谱角距离损失
    输入: [B, C, H, W]
    """
    original_flat = original.permute(0, 2, 3, 1).reshape(-1, original.shape[1])
    recon_flat = reconstructed.permute(0, 2, 3, 1).reshape(-1, reconstructed.shape[1])

    original_norm = torch.norm(original_flat, dim=1) + eps
    recon_norm = torch.norm(recon_flat, dim=1) + eps

    cos_sim = torch.sum(original_flat * recon_flat, dim=1) / (original_norm * recon_norm)
    cos_sim = torch.clamp(cos_sim, -1 + 1e-8, 1 - 1e-8)

    return torch.acos(cos_sim).mean()


def endmember_smooth_loss(endmembers):
    """
    二阶差分平滑损失
    endmembers: [K, bands]
    """
    d1 = endmembers[:, 1:] - endmembers[:, :-1]
    d2 = d1[:, 1:] - d1[:, :-1]
    return torch.mean(d2 ** 2)


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.sqrt(diff * diff + self.eps * self.eps).mean()


def load_data(mat_path, h, w, device):
    """
    加载高光谱数据并生成小波四子带
    不再读取/返回 VCA 初始端元 M1
    """
    mat_data = sio.loadmat(mat_path)
    Y = mat_data['Y']  # (bands, H*W)
    mixed_spectra_np = Y.reshape(Y.shape[0], h, w)  # (bands, H, W)

    LL_np, LH_np, HL_np, HH_np = precompute_wavelet_subbands(mixed_spectra_np)

    LL = torch.tensor(LL_np, dtype=torch.float32, device=device).unsqueeze(0)
    LH = torch.tensor(LH_np, dtype=torch.float32, device=device).unsqueeze(0)
    HL = torch.tensor(HL_np, dtype=torch.float32, device=device).unsqueeze(0)
    HH = torch.tensor(HH_np, dtype=torch.float32, device=device).unsqueeze(0)
    original = torch.tensor(mixed_spectra_np, dtype=torch.float32, device=device).unsqueeze(0)

    true_endmembers = mat_data.get('M', None)
    true_abundances = mat_data.get('A', None)
    return LL, LH, HL, HH, original, true_endmembers, true_abundances





def train_model(model, LL, LH, HL, HH, original,
                true_endmembers, true_abundances,
                optimizer, scheduler, config):
    model.train()
    epochs = config['epochs']

    charbonnier = CharbonnierLoss(eps=config.get('charb_eps', 0.1))
    alpha_sad = config.get('alpha_sad', 0.1)
    beta_smooth = config.get('beta_smooth', 0.015)

    h, w = config['height'], config['width']
    bands = config['bands']
    n_endmembers = config['n_endmembers']

    # 只在有真值时做每 epoch 指标输出
    has_gt = (true_endmembers is not None) and (true_abundances is not None)

    if has_gt:
        from utility import order_abundance, order_endmembers

        true_abundances_reshaped = true_abundances.T.reshape(h, w, -1)   # [H, W, K]
        true_endmembers_t = true_endmembers if true_endmembers.shape[0] == bands else true_endmembers.T  # [bands, K]

    best_loss = float('inf')

    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)

        abundances, endmembers, reconstructed = model(original, LL, LH, HL, HH)

        loss_charb = charbonnier(reconstructed, original)
        loss_sad = sad_loss(original, reconstructed)
        loss_smooth = endmember_smooth_loss(endmembers)

        loss = loss_charb + alpha_sad * loss_sad + beta_smooth * loss_smooth

        loss.backward()
        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()

        if has_gt:
            # 预测丰度: [B, K, H, W] -> [H, W, K]
            estimated_abundances_np = abundances.detach().squeeze(0).cpu().numpy().transpose(1, 2, 0)

            # 预测端元: [B, K, bands] -> [bands, K]
            estimated_endmembers_np = endmembers.detach().squeeze(0).cpu().numpy().T

            abundance_index, mse_values = order_abundance(
                estimated_abundances_np, true_abundances_reshaped
            )

            endmember_index, sad_values = order_endmembers(
                estimated_endmembers_np, true_endmembers_t
            )

            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"Endmember SAD mean={sad_values.mean():.6f} "
                f"Abundance RMSE mean={mse_values.mean():.6f}"
            )
        else:
            print(f"Epoch [{epoch + 1}/{epochs}]")

    print(f"Best training loss: {best_loss:.6f}")

def save_model_weights(model, data_name, config, save_dir='checkpoints'):
    data_name = data_name.rstrip('/\\')
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f"{data_name}.pth")

    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'bands': config['bands'],
        'n_endmembers': config['n_endmembers'],
        'height': config['height'],
        'width': config['width']
    }, filepath)

    print(f"\n模型权重已保存至：{filepath}")
    return filepath


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    data_name = 'samson'
    configs = {
        'moffett': {'mat_path': 'dataset/moffett_dataset.mat', 'height': 50, 'width': 50, 'n_endmembers': 3, 'epochs': 400, 'lr': 4e-3, 'alpha_sad': 1, 'beta_smooth': 0},
        'houston': {'mat_path': 'dataset/houston_dataset.mat', 'height': 105, 'width': 105, 'n_endmembers': 4, 'epochs': 1500, 'lr': 3e-3, 'alpha_sad':1, 'beta_smooth': 0},
        'samson': {'mat_path': 'dataset/samson_dataset.mat', 'height': 95, 'width': 95, 'n_endmembers': 3, 'epochs': 600, 'lr': 8e-3, 'alpha_sad': 25, 'beta_smooth': 1},
        'jasper': {'mat_path': 'dataset/jasper_dataset.mat', 'height': 100, 'width': 100, 'n_endmembers': 4, 'epochs': 1000, 'lr': 6e-3, 'alpha_sad': 15, 'beta_smooth':0},
        'sy30A': {'mat_path': 'dataset/sy30A_dataset.mat', 'height': 60, 'width': 60, 'n_endmembers': 5, 'epochs':1000, 'lr': 4e-3, 'alpha_sad': 10, 'beta_smooth': 1},
        'apex': {'mat_path': 'dataset/apex_dataset.mat', 'height': 110, 'width': 110, 'n_endmembers': 4, 'epochs': 800, 'lr': 6e-3, 'alpha_sad':5, 'beta_smooth': 0},
        'urban': {'mat_path': 'dataset/urban_dataset.mat', 'height': 307, 'width': 307, 'n_endmembers': 4, 'epochs': 250, 'lr': 6e-3, 'alpha_sad': 25, 'beta_smooth':0},
        'urban6': {'mat_path': 'dataset/urban_dataset.mat', 'height': 307, 'width': 307, 'n_endmembers': 6,'epochs': 400, 'lr': 7e-3, 'alpha_sad': 1, 'beta_smooth': 1},
        'DC3': {'mat_path': 'dataset/DC3_dataset.mat', 'height': 105, 'width': 105, 'n_endmembers': 6, 'epochs': 1000, 'lr': 5e-3, 'alpha_sad': 1, 'beta_smooth': 1},
        '1': {'mat_path': 'dataset/1_dataset.mat', 'height': 20, 'width': 20, 'n_endmembers': 3, 'epochs':600, 'lr': 2e-3, 'alpha_sad': 0.1, 'beta_smooth': 0},
        'cuprite': {'mat_path': 'dataset/cuprite_dataset.mat', 'height': 250, 'width': 190, 'n_endmembers': 12, 'epochs': 250, 'lr': 6e-3, 'alpha_sad': 15, 'beta_smooth': 1},
    }
    config = configs[data_name]

    LL, LH, HL, HH, original, true_endmembers, true_abundances = load_data(
        config['mat_path'], config['height'], config['width'], device
    )
    bands = original.shape[1]
    config['bands'] = bands

    model = MSDCN(
        bands=bands,
        height=config['height'],
        width=config['width'],
        n_endmembers=config['n_endmembers']
    ).to(device)

    # weights_init(model)

    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=4e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.95)  #0.95

    train_model(
        model=model,
        LL=LL, LH=LH, HL=HL, HH=HH,
        original=original,
        true_endmembers=true_endmembers,
        true_abundances=true_abundances,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config
    )

    save_model_weights(model, data_name, config)

    model.eval()
    with torch.no_grad():
        estimated_abundances, estimated_endmembers, reconstructed = model(original,LL, LH, HL, HH)

    final_sad = sad_loss(original, reconstructed)
    final_charb = CharbonnierLoss()(reconstructed, original)
    final_smooth = endmember_smooth_loss(estimated_endmembers)

    print("\n最终损失：")
    print(f"  SAD: {final_sad:.6f}")
    print(f"  Charbonnier: {final_charb:.6f}")
    print(f"  Endmember smooth: {final_smooth:.6f}")

    estimated_endmembers_np = estimated_endmembers.detach().cpu().numpy().T  # -> (bands, K)
    estimated_abundances_np = estimated_abundances.detach().squeeze(0).cpu().numpy()  # (K, H, W)
    estimated_abundances_reshaped = estimated_abundances_np.transpose(1, 2, 0)  # (H, W, K)

    if true_endmembers is not None and true_abundances is not None:
        h, w = config['height'], config['width']
        true_abundances_reshaped = true_abundances.T.reshape(h, w, -1)
        true_endmembers_t = true_endmembers if true_endmembers.shape[0] == bands else true_endmembers.T

        from utility import order_abundance, order_endmembers
        abundance_index, mse_values = order_abundance(estimated_abundances_reshaped, true_abundances_reshaped)
        print("丰度MSE值:", mse_values, '均值：', mse_values.mean())

        endmember_index, sad_values = order_endmembers(estimated_endmembers_np, true_endmembers_t)
        print("端元SAD值:", sad_values, '均值：', sad_values.mean())
        print(abundance_index, endmember_index)

        for i in range(config['n_endmembers']):
            estimated_endmembers_np[:, i] = estimated_endmembers_np[:, i] / (estimated_endmembers_np[:, i].max() + 1e-8)
            true_endmembers_t[:, i] = true_endmembers_t[:, i] / (true_endmembers_t[:, i].max() + 1e-8)

        print(abundance_index, endmember_index)
        print("before import plots")

        from plots import plot_abundance, plot_endmembers
        print("after import plots")

        print("before plot_abundance")
        plot_abundance(
            true_abundances_reshaped,
            estimated_abundances_reshaped,
            abundance_index,
            config['n_endmembers'],
            f"{data_name}/"
        )
        print("after plot_abundance")

        print("before plot_endmembers")
        plot_endmembers(
            true_endmembers_t,
            estimated_endmembers_np,
            endmember_index,
            config['n_endmembers'],
            f"{data_name}/"
        )
        print("after plot_endmembers")


if __name__ == "__main__":
    total_start_time = time.time()
    main()
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    print("=" * 80)
    print(f"Total program time: {total_time:.2f} s")

