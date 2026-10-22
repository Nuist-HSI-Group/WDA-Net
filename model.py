import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from Endmember_Gene_HybridRandom import EndmemberExtractor
from mamba_ssm import Mamba


class LiteMambaBlock(nn.Module):
    def __init__(self, dim, d_state=8, d_conv=3, expand=1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x):
        return x + self.mamba(self.norm(x))

class NoiseDecoder(nn.Module):
    """
    生成与高光谱重构同尺寸的噪声项
    输出范围严格限制在 (0, 0.005)
    """
    def __init__(self, in_channels, bands, max_noise=0.005):
        super().__init__()
        self.max_noise = max_noise
        self.proj = nn.Conv2d(in_channels, bands, kernel_size=1, bias=False, padding=1)

        # 初始噪声接近 0，而不是 0.0025
        nn.init.zeros_(self.proj.weight)
        #nn.init.constant_(self.proj.bias, -7.0)

    def forward(self, x):
        noise = self.max_noise * torch.sigmoid(self.proj(x))
        return noise



class LowFreqGuidedHighFreqSelectiveMambaFusion(nn.Module):
    """
    LFG-HFSM: Low-Frequency Guided High-Frequency Selective Mamba Fusion

    设计思想：
      1) LL 是低频主干，先用 spatial Mamba 建模全局 abundance 结构，得到 g_LL。
      2) LH/HL/HH 是方向高频细节，分别用方向一致的 Mamba 增强：
           LH: horizontal row scan
           HL: vertical column scan
           HH: diagonal scan
      3) 由 g_LL 与各高频分支共同生成选择性门控，决定哪些高频细节需要注入。
      4) 融合形式：
           fused = g_LL + gate_LH * f_LH + gate_HL * f_HL + gate_HH * f_HH
      5) 融合后再接 spatial Mamba + channel MLP refine，增强全局一致性。

    输入: 4 个分支特征，每个都是 [B, 8K, H, W]
    输出: abundance logits [B, K, H, W]
    """
    def __init__(
        self,
        channels,
        hidden_dim=32,
        d_state=8,
        d_conv=3,
        expand=1,
        gate_hidden=None,
        use_directional_hf=True,
        use_residual_high=True,
        gate_temperature=1.0,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_dim = hidden_dim
        self.use_directional_hf = use_directional_hf
        self.use_residual_high = use_residual_high
        self.gate_temperature = gate_temperature

        if gate_hidden is None:
            gate_hidden = max(hidden_dim // 2, 8)

        # 四个小波子带投影到统一隐空间
        self.proj_LL = nn.Conv2d(channels, hidden_dim, kernel_size=3, bias=False, padding=1)
        self.proj_LH = nn.Conv2d(channels, hidden_dim, kernel_size=3, bias=False, padding=1)
        self.proj_HL = nn.Conv2d(channels, hidden_dim, kernel_size=3, bias=False, padding=1)
        self.proj_HH = nn.Conv2d(channels, hidden_dim, kernel_size=3, bias=False, padding=1)

        # LL 主干：全局 spatial Mamba
        self.ll_spatial_mamba = LiteMambaBlock(
            dim=hidden_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

        # 高频方向增强：LH 横向，HL 纵向，HH 对角
        self.lh_mamba = LiteMambaBlock(hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.hl_mamba = LiteMambaBlock(hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.hh_mamba = LiteMambaBlock(hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand)

        # 如果不想使用方向扫描，也可以退化成共享/普通空间 Mamba
        self.shared_hf_mamba = LiteMambaBlock(hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand)

        # LL 引导的高频选择门控
        # gate_x = sigmoid(Conv([g_LL, f_x]))
        self.gate_LH = nn.Sequential(
            nn.Conv2d(2 * hidden_dim, gate_hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(gate_hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(gate_hidden, hidden_dim, kernel_size=3, padding=1, bias=True)
        )
        self.gate_HL = nn.Sequential(
            nn.Conv2d(2 * hidden_dim, gate_hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(gate_hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(gate_hidden, hidden_dim, kernel_size=3, padding=1, bias=True)
        )
        self.gate_HH = nn.Sequential(
            nn.Conv2d(2 * hidden_dim, gate_hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(gate_hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(gate_hidden, hidden_dim, kernel_size=3, padding=1, bias=True)
        )

        # 高频注入强度，初始值较小，避免训练初期高频噪声破坏 abundance 主结构
        self.alpha_LH = nn.Parameter(torch.tensor(0.3))
        self.alpha_HL = nn.Parameter(torch.tensor(0.3))
        self.alpha_HH = nn.Parameter(torch.tensor(0.3))

        # 融合后的全局 refine
        self.fuse_norm = nn.BatchNorm2d(hidden_dim)
        self.fuse_spatial_mamba = LiteMambaBlock(
            dim=hidden_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

        # 轻量 channel/spectral refine，建模 abundance/endmember 相关通道依赖
        self.channel_refine = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
        )

        self.refine_scale = nn.Parameter(torch.tensor(0.5))

        # 输出回 K 通道
        self.out_proj = nn.Conv2d(hidden_dim, int(channels / 8), kernel_size=3, bias=False, padding=1)

    def _scan_raster(self, x, block):
        """
        [B, D, H, W] -> [B, H*W, D] -> [B, D, H, W]
        """
        B, D, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2).contiguous()
        tokens = block(tokens)
        return tokens.transpose(1, 2).reshape(B, D, H, W).contiguous()

    def _scan_horizontal(self, x, block):
        """
        LH: 按行扫描，每一行作为一个序列:
        [B, D, H, W] -> [B*H, W, D]
        """
        B, D, H, W = x.shape
        tokens = x.permute(0, 2, 3, 1).reshape(B * H, W, D).contiguous()
        tokens = block(tokens)
        return tokens.reshape(B, H, W, D).permute(0, 3, 1, 2).contiguous()

    def _scan_vertical(self, x, block):
        """
        HL: 按列扫描，每一列作为一个序列:
        [B, D, H, W] -> [B*W, H, D]
        """
        B, D, H, W = x.shape
        tokens = x.permute(0, 3, 2, 1).reshape(B * W, H, D).contiguous()
        tokens = block(tokens)
        return tokens.reshape(B, W, H, D).permute(0, 3, 2, 1).contiguous()

    def _diag_indices(self, H, W, device):
        """
        主对角线方向扫描索引，对应 HH 的 135° 对角纹理。
        每条对角线单独作为一个序列，避免不同对角线之间状态串扰。
        """
        idx_list = []
        len_list = []
        for offset in range(-(W - 1), H):
            h0 = max(offset, 0)
            w0 = max(-offset, 0)
            length = min(H - h0, W - w0)
            h = h0 + torch.arange(length, device=device)
            w = w0 + torch.arange(length, device=device)
            idx_list.append(h * W + w)
            len_list.append(length)
        return idx_list, len_list

    def _scan_diagonal(self, x, block):
        """
        HH: 精确主对角线扫描。
        [B, D, H, W] -> 多条 [B, L_diag, D] -> [B, D, H, W]
        """
        B, D, H, W = x.shape
        flat = x.flatten(2).transpose(1, 2).contiguous()  # [B, H*W, D]
        idx_list, len_list = self._diag_indices(H, W, x.device)

        num_diag = len(idx_list)
        max_len = max(len_list)
        padded = flat.new_zeros(B * num_diag, max_len, D)

        for i, idx in enumerate(idx_list):
            length = len_list[i]
            padded[i * B:(i + 1) * B, :length, :] = flat.index_select(1, idx)

        padded = block(padded)

        out_flat = flat.new_empty(B, H * W, D)
        for i, idx in enumerate(idx_list):
            length = len_list[i]
            out_flat[:, idx, :] = padded[i * B:(i + 1) * B, :length, :]

        return out_flat.transpose(1, 2).reshape(B, D, H, W).contiguous()

    def _hf_enhance(self, f_LH, f_HL, f_HH):
        """
        高频方向增强。
        """
        if self.use_directional_hf:
            f_LH = self._scan_horizontal(f_LH, self.lh_mamba)
            f_HL = self._scan_vertical(f_HL, self.hl_mamba)
            f_HH = self._scan_diagonal(f_HH, self.hh_mamba)
        else:
            f_LH = self._scan_raster(f_LH, self.shared_hf_mamba)
            f_HL = self._scan_raster(f_HL, self.shared_hf_mamba)
            f_HH = self._scan_raster(f_HH, self.shared_hf_mamba)

        return f_LH, f_HL, f_HH

    def forward(self, feat_LL, feat_LH, feat_HL, feat_HH):
        B, _, H, W = feat_LL.shape

        # 1) 投影
        f_LL = self.proj_LL(feat_LL)
        f_LH = self.proj_LH(feat_LH)
        f_HL = self.proj_HL(feat_HL)
        f_HH = self.proj_HH(feat_HH)

        # 2) LL 低频主干上下文
        g_LL = self._scan_raster(f_LL, self.ll_spatial_mamba)

        # 3) 高频方向 Mamba 增强
        f_LH, f_HL, f_HH = self._hf_enhance(f_LH, f_HL, f_HH)

        # 4) LL 引导的高频选择门控
        gate_LH = torch.sigmoid(self.gate_LH(torch.cat([g_LL, f_LH], dim=1)) / self.gate_temperature)
        gate_HL = torch.sigmoid(self.gate_HL(torch.cat([g_LL, f_HL], dim=1)) / self.gate_temperature)
        gate_HH = torch.sigmoid(self.gate_HH(torch.cat([g_LL, f_HH], dim=1)) / self.gate_temperature)

        # 5) 选择性注入高频
        if self.use_residual_high:
            # 只注入相对于低频主干的高频差异，进一步减少冗余和噪声
            inj_LH = gate_LH * (f_LH - g_LL)
            inj_HL = gate_HL * (f_HL - g_LL)
            inj_HH = gate_HH * (f_HH - g_LL)
        else:
            inj_LH = gate_LH * f_LH
            inj_HL = gate_HL * f_HL
            inj_HH = gate_HH * f_HH

        fused = (
            g_LL
            + self.alpha_LH * inj_LH
            + self.alpha_HL * inj_HL
            + self.alpha_HH * inj_HH
        )

        # 6) 融合后 spatial refine
        fused = self.fuse_norm(fused)
        tokens_in = fused.flatten(2).transpose(1, 2).contiguous()
        tokens_out = self.fuse_spatial_mamba(tokens_in)
        fused = tokens_out.transpose(1, 2).reshape(B, self.hidden_dim, H, W).contiguous()

        # 7) channel/spectral refine
        fused = fused + self.refine_scale * self.channel_refine(fused)

        # 8) abundance logits
        out = self.out_proj(fused)
        return out



class ConvBlock(nn.Module):
    def __init__(self, in_ch, mid1, mid2, out_ch, k1, k2, k3, dropout=0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid1, kernel_size=k1, padding='same', bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(mid1),
            nn.Dropout2d(dropout),

            nn.Conv2d(mid1, mid2, kernel_size=k2, padding='same', bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(mid2),
            nn.Dropout2d(dropout),

            nn.Conv2d(mid2, out_ch, kernel_size=k3, padding='same', bias=False),
        )

    def forward(self, x):
        return self.net(x)


class DirectionalConv2d(nn.Module):
    """
    真正的方向卷积：
    只在指定方向的对角线上保留可学习参数，其余位置恒为 0

    direction='45'  -> 反对角线 / 方向
    direction='135' -> 主对角线 \ 方向
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=5,
                 stride=1,
                 padding=None,
                 dilation=1,
                 groups=1,
                 bias=True,
                 direction='45'):
        super().__init__()

        assert kernel_size % 2 == 1, "kernel_size 必须是奇数，建议 3/5/7"
        assert direction in ['45', '135']
        assert in_channels % groups == 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.direction = direction

        if padding is None:
            padding = (kernel_size // 2) * dilation
        self.padding = padding

        self.weight_1d = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size)
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_1d, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = (self.in_channels // self.groups) * self.kernel_size
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def build_kernel(self):
        k = self.kernel_size
        device = self.weight_1d.device
        dtype = self.weight_1d.dtype

        kernel = torch.zeros(
            self.out_channels,
            self.in_channels // self.groups,
            k,
            k,
            device=device,
            dtype=dtype
        )

        idx = torch.arange(k, device=device)

        if self.direction == '45':
            kernel[:, :, idx, k - 1 - idx] = self.weight_1d
        else:
            kernel[:, :, idx, idx] = self.weight_1d

        return kernel

    def forward(self, x):
        kernel = self.build_kernel()
        return F.conv2d(
            x,
            kernel,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )


class DiagonalConvBlock(nn.Module):
    """
    HH 分支专用：三层主对角线方向卷积
    """
    def __init__(self, in_ch, mid1, mid2, out_ch, k1, k2, k3, dropout=0.05):
        super().__init__()
        self.net = nn.Sequential(
            DirectionalConv2d(in_ch, mid1, kernel_size=k1, bias=False, direction='135'),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(mid1),
            nn.Dropout2d(dropout),

            DirectionalConv2d(mid1, mid2, kernel_size=k2, bias=False, direction='135'),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(mid2),
            nn.Dropout2d(dropout),

            DirectionalConv2d(mid2, out_ch, kernel_size=k3, bias=False, direction='135'),
        )

    def forward(self, x):
        return self.net(x)


class MSDCN(nn.Module):
    def __init__(
        self,
        bands,
        height,
        width,
        n_endmembers,
        wavelet='bior3.3',
        dropout=0.15,
        fusion_mode='weighted',
        em_feat_ch=128,
        em_hidden_dim=256,
        em_constrain='sigmoid'
    ):
        super().__init__()
        self.bands = bands
        self.n_endmembers = n_endmembers
        self.wavelet = wavelet
        self.height = height
        self.width = width

        K = n_endmembers

        self.conv_LL = ConvBlock(
            in_ch=bands, mid1=32 * K, mid2=8 * K, out_ch=K,
            k1=3, k2=3, k3=3, dropout=dropout
        )
        self.conv_LH = ConvBlock(
            in_ch=bands, mid1=32 * K, mid2=8 * K, out_ch=K,
            k1=(1, 3), k2=(1, 5), k3=(1, 7), dropout=dropout
        )
        self.conv_HL = ConvBlock(
            in_ch=bands, mid1=32 * K, mid2=8* K, out_ch=K,
            k1=(3, 1), k2=(5, 1), k3=(7, 1), dropout=dropout
        )

        # 这里只改 HH，其余不变
        self.conv_HH = DiagonalConvBlock(
            in_ch=bands, mid1=32 * K, mid2=8 * K, out_ch=K,
            k1=3, k2=5, k3=7, dropout=dropout
        )

        self.fusion_mode = fusion_mode

        if fusion_mode == 'conv':
            self.fusion = nn.Conv2d(32 * K, K, kernel_size=3, padding='same', bias=True)
        elif fusion_mode == 'weighted':
            self.init_logits = torch.log(torch.tensor([0.70, 0.025, 0.025, 0.25], dtype=torch.float32))
            #self.fusion_logits = nn.Parameter(init_logits)
        elif fusion_mode == 'mamba':
            self.fusion = LowFreqGuidedHighFreqSelectiveMambaFusion(
                channels=8*K,
                hidden_dim=16,  # 轻量，通常 16/32 就够了
                d_state=16*K,
                d_conv=3,
                expand=1
            )
        else:
            raise ValueError("fusion_mode must be 'weighted', 'conv' or 'mamba'")

        self.smooth = nn.Sequential(
            nn.Conv2d(K, K, kernel_size=3, padding='same', bias=False),
        )

        self.softmax = nn.Softmax(dim=1)

        self.endmember_generator = EndmemberExtractor(
            bands=bands,
            n_endmembers=n_endmembers,
            num_queries_times=n_endmembers*n_endmembers,
            dropout=0.01,
            constrain="sigmoid",
        )
        self.noise_decoder = NoiseDecoder(
            in_channels=bands,
            bands=bands,
            max_noise=0.0005
        )

        self._cached_endmembers = None
        self._cached_noise = None

        self._cached_endmembers = None
        self._cached_noise = None

    @property
    def endmembers(self):
        if self._cached_endmembers is None:
            raise RuntimeError("No cached endmembers. Run forward() first or call get_endmembers(LL).")
        return self._cached_endmembers

    def get_endmembers(self, LL=None):
        if LL is not None:
            return self.endmember_generator(LL)
        return self.endmembers

    def encode_abundance(self, LL, LH, HL, HH):
        feat_LL = self.conv_LL(LL)
        feat_LH = self.conv_LH(LH)
        feat_HL = self.conv_HL(HL)
        feat_HH = self.conv_HH(HH)

        if self.fusion_mode == 'conv':
            fused = self.fusion(torch.cat([feat_LL, feat_LH, feat_HL, feat_HH], dim=1))
        elif self.fusion_mode == 'weighted':
            #w = F.softmax(self.fusion_logits, dim=0)
            w=self.init_logits
            fused = (
                    w[0] * feat_LL +
                    w[1] * feat_LH +
                    w[2] * feat_HL +
                    w[3] * feat_HH
            )
        else:  # mamba
            fused = self.fusion(feat_LL, feat_LH, feat_HL, feat_HH)

        fused = self.smooth(fused)
        abundance = self.softmax(fused)

        return abundance, fused

    def decode(self, abundance, endmembers):
        reconstructed = torch.einsum('bkhw,bkc->bchw', abundance, endmembers)
        return reconstructed

    def forward(self, original, LL, LH, HL, HH):
        abundance, fused_feature = self.encode_abundance(LL, LH, HL, HH)
        endmembers = self.endmember_generator(original)

        reconstructed_linear = self.decode(abundance, endmembers)

        noise = 0.000005 * torch.randn_like(reconstructed_linear)
        noise = torch.abs(noise)
        noise = torch.clamp(noise, -0.000005, 0.000005)

        reconstructed = reconstructed_linear + 0*noise

        self._cached_endmembers = endmembers
        self._cached_noise = noise
        return abundance, endmembers, reconstructed