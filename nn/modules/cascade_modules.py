import torch
import torch.nn as nn
import torch.nn.functional as F
from .block import C2f, Bottleneck

# ==========================================
# 创新点一：可微注视变换模块 (Differentiable Gaze Shift)
# ==========================================
class DifferentiableGazeShift(nn.Module):
    def __init__(self, out_size=(640, 640)):
        """
        args:
            out_size: 细视网络需要的输入尺寸 (H, W)
        """
        super().__init__()
        self.out_size = out_size

    def forward(self, x, crop_params):
        """
        x: 输入的全图 Tensor [B, C, H_in, W_in]
        crop_params: 粗检测器输出的裁剪参数 [B, 3] -> (tx, ty, scale)
                     tx, ty 范围在 [-1, 1], scale 范围 (0, 1]
        """
        B, C, H, W = x.shape
        
        # 1. 构建仿射变换矩阵 theta [B, 2, 3]
        #    [ sx, 0, tx ]
        #    [ 0, sy, ty ]
        theta = torch.zeros(B, 2, 3, device=x.device, dtype=x.dtype)
        
        # 缩放因子 (scale)。注意：STN中 scale越小，视野越小(放大倍数越大)
        # 这里假设传入的 scale 是 "保留区域的比例"，例如 0.5 代表取一半长宽
        s = crop_params[:, 2] 
        tx = crop_params[:, 0]
        ty = crop_params[:, 1]

        theta[:, 0, 0] = s
        theta[:, 1, 1] = s
        theta[:, 0, 2] = tx
        theta[:, 1, 2] = ty

        # 2. 生成采样网格 (Affine Grid)
        # 注意：size 需要是 (B, C, H_out, W_out)
        grid = F.affine_grid(theta, torch.Size((B, C, self.out_size[0], self.out_size[1])), align_corners=False)

        # 3. 可微采样 (Differentiable Sampling / Bilinear Interpolation)
        # 这就是公式 V_out(x,y) 的代码实现
        x_cropped = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

        return x_cropped

# ==========================================
# 创新点二：全局-局部上下文纠缠模块 (GL-Context Entanglement)
# ==========================================
class GL_ContextBlock(nn.Module):
    def __init__(self, c_local, c_global, c_out, nhead=4):
        """
        c_local: 细视特征通道数 (Query)
        c_global: 粗视特征通道数 (Key/Value)
        c_out: 输出通道数
        """
        super().__init__()
        self.norm_local = nn.LayerNorm(c_local)
        self.norm_global = nn.LayerNorm(c_global)
        
        # 这里的 Cross Attention 可以使用 PyTorch 自带的，也可以手写以更好地控制
        # 为了方便集成，我们使用 nn.MultiheadAttention
        # 注意：Transformer通常主要在 dim 维度操作，Conv层需要 permute
        self.cross_attn = nn.MultiheadAttention(embed_dim=c_local, kdim=c_global, vdim=c_global, num_heads=nhead, batch_first=True)
        
        self.proj = nn.Conv2d(c_local, c_out, 1) if c_local != c_out else nn.Identity()

    def forward(self, x_local, x_global):
        """
        x_local: [B, C_l, H_l, W_l] (Fine Feature)
        x_global: [B, C_g, H_g, W_g] (Coarse Feature)
        """
        B, C_l, H_l, W_l = x_local.shape
        B, C_g, H_g, W_g = x_global.shape

        # 1. 对齐空间特征 (Flatten)
        # [B, H_l*W_l, C_l]
        q = x_local.flatten(2).permute(0, 2, 1)
        # [B, H_g*W_g, C_g]
        k = x_global.flatten(2).permute(0, 2, 1)
        v = x_global.flatten(2).permute(0, 2, 1)

        # 归一化 (LayerNorm)
        q = self.norm_local(q)
        k = self.norm_global(k)
        v = self.norm_global(v)

        # 2. Cross Attention: Query=Local, Key/Value=Global
        # 公式: Softmax(Q * K.T / sqrt(d)) * V
        attn_out, _ = self.cross_attn(query=q, key=k, value=v)

        # 3. 残差连接 + 形状还原
        # Fused = LayerNorm(Q + Attention) ... 这里简单实现为直接相加后输出
        out = q + attn_out
        
        # [B, L, C] -> [B, C, L] -> [B, C, H, W]
        out = out.permute(0, 2, 1).view(B, C_l, H_l, W_l)
        
        return self.proj(out)