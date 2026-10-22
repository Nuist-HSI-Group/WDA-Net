import torch
import torch.nn as nn
import torch.nn.functional as F


class EndmemberExtractor(nn.Module):
    """
    Hybrid Spectral-Random Endmember Extractor

    目标：
      1) 保留输入高光谱图像的真实光谱分布信息；
      2) 不完全依赖候选像元或近似纯像元；
      3) 引入可训练随机光谱原型，让端元可以从“数据候选 + 随机原型 + 全局均值锚点”中联合学习。

    输出:
      endmembers: [B, n_endmembers, bands]
    """
    def __init__(
        self,
        bands,
        n_endmembers,
        num_queries_times=10,
        dropout=0.0,
        constrain="sigmoid",
        random_ratio=0.39,
        mean_ratio=0.01,
        training_jitter=0.0,
        temperature=1.0,
    ):
        super().__init__()
        self.bands = bands
        self.n_endmembers = n_endmembers
        self.num_queries_times = num_queries_times
        self.num_queries = n_endmembers * num_queries_times
        self.constrain = constrain
        self.training_jitter = training_jitter
        self.temperature = temperature

        self.pixel_norm = nn.LayerNorm(bands)

        # 数据候选分支：从真实像元光谱中提取候选端元表示
        # 这里学习的是 residual/refine，而不是完全替代原始光谱。
        self.spectral_refine = nn.Sequential(
            nn.Linear(bands, bands),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bands, bands),
        )

        # 随机原型分支：每个端元有一组可训练随机光谱原型
        # forward 中会以图像均值光谱为锚点，用 std 控制扰动范围，使其“随机但不脱离数据分布”。
        self.random_spectra_raw = nn.Parameter(
            torch.empty(n_endmembers, num_queries_times, bands)
        )
        self.random_refine = nn.Sequential(
            nn.LayerNorm(bands),
            nn.Linear(bands, bands),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bands, bands),
        )

        # 每个端元对应一组数据候选光谱，组内做加权融合
        self.data_weights = nn.Parameter(torch.ones(n_endmembers, num_queries_times))

        # 每个端元对应一组随机光谱原型，组内做加权融合
        self.random_weights = nn.Parameter(torch.ones(n_endmembers, num_queries_times))

        # 三个来源的融合权重：[data_candidate, random_proto, mean_anchor]
        # 初始时仍以数据候选为主，但随机分支占一定比例，避免没有纯像元时端元被混合像元锁死。
        random_ratio = float(random_ratio)
        mean_ratio = float(mean_ratio)
        if random_ratio + mean_ratio >= 0.95:
            random_ratio = 0.35
            mean_ratio = 0.10
        data_ratio = 1.0 - random_ratio - mean_ratio
        init_probs = torch.tensor([data_ratio, random_ratio, mean_ratio], dtype=torch.float32)
        self.source_logits = nn.Parameter(
            torch.log(init_probs).view(1, 3).repeat(n_endmembers, 1)
        )

        # 控制随机原型围绕图像均值光谱的扰动幅度
        self.random_scale = nn.Parameter(torch.tensor(0.5))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # 随机原型初始化。tanh 后落在 [-1, 1]，再乘以图像 std。
        nn.init.normal_(self.random_spectra_raw, mean=0.0, std=0.5)

        # 最后一层 residual 初始化小一点，避免训练初期破坏光谱形状。
        if isinstance(self.spectral_refine[-1], nn.Linear):
            nn.init.zeros_(self.spectral_refine[-1].weight)
            nn.init.zeros_(self.spectral_refine[-1].bias)

        if isinstance(self.random_refine[-1], nn.Linear):
            nn.init.zeros_(self.random_refine[-1].weight)
            nn.init.zeros_(self.random_refine[-1].bias)

    def _safe_logit(self, x, eps=1e-4):
        x = x.clamp(eps, 1.0 - eps)
        return torch.log(x / (1.0 - x))

    def _to_model_space(self, x):
        """
        如果最终用 sigmoid 约束端元到 [0, 1]，则内部用 logit 空间学习，
        这样 final sigmoid 后还能保留原始光谱数值，而不是把 0-1 输入整体压到 0.5 附近。
        """
        if self.constrain == "sigmoid":
            return self._safe_logit(x)
        return x

    def _apply_constraint(self, x):
        if self.constrain == "sigmoid":
            return torch.sigmoid(x)
        elif self.constrain == "softplus":
            return F.softplus(x)
        elif self.constrain is None:
            return x
        else:
            raise ValueError(f"Unsupported constrain mode: {self.constrain}")

    def _segment_pool(self, x, out_tokens):
        # x: [B, N, C]
        B, N, C = x.shape

        if out_tokens >= N:
            idx = torch.linspace(0, N - 1, steps=out_tokens, device=x.device)
            idx = idx.round().long().clamp(0, N - 1)
            return x[:, idx, :]

        boundaries = torch.linspace(0, N, steps=out_tokens + 1, device=x.device).round().long()
        boundaries[0] = 0
        boundaries[-1] = N

        pooled = []
        for i in range(out_tokens):
            s = int(boundaries[i].item())
            e = int(boundaries[i + 1].item())
            if e <= s:
                e = min(s + 1, N)
            pooled.append(x[:, s:e, :].mean(dim=1, keepdim=True))

        return torch.cat(pooled, dim=1)  # [B, out_tokens, C]

    def _build_data_candidates(self, spatial_tokens):
        """
        数据候选分支。
        spatial_tokens: [B, H*W, bands], usually in [0, 1]
        """
        base_tokens = self._to_model_space(spatial_tokens)

        # 用原始光谱做 norm/refine，学习 residual
        residual = self.spectral_refine(self.pixel_norm(spatial_tokens))
        refined_tokens = base_tokens + residual

        candidate_tokens = self._segment_pool(refined_tokens, self.num_queries)
        candidate_tokens = candidate_tokens.view(
            spatial_tokens.shape[0],
            self.n_endmembers,
            self.num_queries_times,
            self.bands
        )  # [B, K, q, bands]

        w_data = F.softmax(self.data_weights / self.temperature, dim=1)  # [K, q]
        data_endmembers = (
            candidate_tokens * w_data.unsqueeze(0).unsqueeze(-1)
        ).sum(dim=2)  # [B, K, bands]

        return data_endmembers

    def _build_random_prototypes(self, spatial_tokens):
        """
        随机原型分支。
        随机原型不是完全自由乱跑，而是围绕当前图像的均值光谱和方差生成：
          random_proto = mean_spectrum + scale * std_spectrum * trainable_random
        如果 constrain='sigmoid'，则在 logit 空间中构造。
        """
        B, N, C = spatial_tokens.shape

        mean_spec = spatial_tokens.mean(dim=1)  # [B, bands]
        std_spec = spatial_tokens.std(dim=1, unbiased=False).clamp_min(1e-6)  # [B, bands]

        if self.constrain == "sigmoid":
            anchor = self._safe_logit(mean_spec)
            # 粗略把 reflectance 空间的 std 映射到 logit 空间，避免扰动太小
            spread = std_spec / (mean_spec * (1.0 - mean_spec)).clamp_min(1e-3)
            spread = spread.clamp(max=5.0)
        else:
            anchor = mean_spec
            spread = std_spec

        rand = torch.tanh(self.random_spectra_raw).unsqueeze(0)  # [1, K, q, bands]
        scale = F.softplus(self.random_scale)

        random_tokens = anchor[:, None, None, :] + scale * spread[:, None, None, :] * rand

        # 随机原型也经过一个轻量 spectral refine，但最后一层零初始化，初期不破坏随机锚点。
        random_tokens = random_tokens + self.random_refine(random_tokens)

        w_rand = F.softmax(self.random_weights / self.temperature, dim=1)  # [K, q]
        random_endmembers = (
            random_tokens * w_rand.unsqueeze(0).unsqueeze(-1)
        ).sum(dim=2)  # [B, K, bands]

        # 全局均值锚点，作为稳定项
        mean_anchor = anchor.unsqueeze(1).expand(B, self.n_endmembers, C)

        return random_endmembers, mean_anchor, std_spec

    def get_source_weights(self):
        """
        查看每个端元的数据候选/随机原型/均值锚点占比。
        返回: [K, 3]，三列分别是 data, random, mean。
        """
        return F.softmax(self.source_logits, dim=-1).detach()
    def forward(self, x):
        # x: [B, bands, H, W]
        B, C, H, W = x.shape
        if C != self.bands:
            raise ValueError(f"Expected {self.bands} bands, but got {C}")

        # 由高光谱输入生成初始光谱序列
        spatial_tokens = x.flatten(2).transpose(1, 2).contiguous()  # [B, H*W, bands]

        # 1) 数据候选端元：保留真实光谱分布
        data_endmembers = self._build_data_candidates(spatial_tokens)

        # 2) 随机原型端元：避免完全依赖纯像元/候选像元
        random_endmembers, mean_anchor, std_spec = self._build_random_prototypes(spatial_tokens)

        # 3) 三源融合：data candidate + random prototype + mean anchor
        source_w = F.softmax(self.source_logits, dim=-1)  # [K, 3]
        sources = torch.stack(
            [data_endmembers, random_endmembers, mean_anchor],
            dim=2
        )  # [B, K, 3, bands]

        endmembers = (
            sources * source_w.unsqueeze(0).unsqueeze(-1)
        ).sum(dim=2)  # [B, K, bands]

        # 4) 训练时可选微小随机扰动，增强鲁棒性。默认 training_jitter=0.0，不影响稳定性。
        if self.training and self.training_jitter > 0:
            noise = torch.randn_like(endmembers) * std_spec[:, None, :] * self.training_jitter
            endmembers = endmembers + noise

        endmembers = self._apply_constraint(endmembers)
        return endmembers
