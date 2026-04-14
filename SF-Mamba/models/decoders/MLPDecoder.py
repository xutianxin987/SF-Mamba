import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # (N, HW, C)
        x = self.proj(x)
        return x

class DecoderHead(nn.Module):
    def __init__(self,
                 in_channels=[96, 192, 384, 768],
                 num_classes=40,
                 dropout_ratio=0.1,
                 norm_layer=nn.BatchNorm2d,
                 embed_dim=768,
                 align_corners=False,
                 context_mode: str = "both"):  # 默认使用 both+gating
        super().__init__()
        self.num_classes = num_classes
        self.align_corners = align_corners
        self.context_mode = context_mode.lower()
        assert self.context_mode in {"both", "avg", "max", "none"}

        c1_in, c2_in, c3_in, c4_in = in_channels

        # Projection layers
        self.linear_c1 = MLP(c1_in, embed_dim)
        self.linear_c2 = MLP(c2_in, embed_dim)
        self.linear_c3 = MLP(c3_in, embed_dim)
        self.linear_c4 = MLP(c4_in, embed_dim)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, kernel_size=1),
            norm_layer(embed_dim),
            nn.ReLU(inplace=True)
        )

        # gating 参数 (avg, max)
        self.gate = nn.Parameter(torch.zeros(2))  # learnable logits

        # DW conv 保留
        pool_in_ch = embed_dim * 2   # feat + f_ctx
        self.pool_fuse = nn.Sequential(
            nn.Conv2d(pool_in_ch, embed_dim, kernel_size=1, groups=embed_dim),  # ✅ DW conv
            norm_layer(embed_dim),
            nn.ReLU(inplace=True)
        )

        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        n = c4.size(0)

        def project_and_up(x, proj, target_hw):
            H, W = x.shape[2], x.shape[3]
            x = proj(x).permute(0, 2, 1).reshape(n, -1, H, W)
            return F.interpolate(x, size=target_hw, mode='bilinear', align_corners=self.align_corners)

        target_hw = c1.shape[2:]
        _c4 = project_and_up(c4, self.linear_c4, target_hw)
        _c3 = project_and_up(c3, self.linear_c3, target_hw)
        _c2 = project_and_up(c2, self.linear_c2, target_hw)
        H1, W1 = target_hw
        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, H1, W1)

        # Multi-scale fusion
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        # Global pooling
        avg = F.adaptive_avg_pool2d(_c, 1)
        max_ = F.adaptive_max_pool2d(_c, 1)

        # gating with softmax
        gate_w = torch.softmax(self.gate, dim=0)  # (2,)
        f_ctx = gate_w[0] * avg + gate_w[1] * max_
        f_ctx = f_ctx.expand_as(_c)

        # concat + DW conv
        ctx_in = torch.cat([_c, f_ctx], dim=1)
        fused = self.pool_fuse(ctx_in)

        out = self.dropout(fused)
        out = self.linear_pred(out)
        return out
