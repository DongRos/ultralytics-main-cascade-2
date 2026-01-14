import torch
import torch.nn as nn
from ultralytics import YOLO
from thop import profile
# 引入你的自定义模块 (确保路径正确，或者把类定义复制过来)
from ultralytics.nn.modules.cascade_modules import DifferentiableGazeShift, GL_ContextBlock

def compute_flops():
    # 1. 加载你的新模型
    print("正在加载模型...")
    model = YOLO('/home/liuyadong/ultralytics-main-cascade-2/yolo12s-cascade-EALF.yaml') # 替换为你修改后的yaml路径
    model_pt = model.model # 获取 pytorch 模型实例
    
    # 2. 创建一个伪造的输入 (Batch=1, RGB, 640x640)
    input_tensor = torch.randn(1, 3, 640, 640)

    # 3. 定义自定义规则 (Custom Handlers)
    # 告诉 thop 如何计算你那些奇怪模块的计算量
    custom_ops = {
        # STN 主要是网格生成和采样，计算量相对卷积很小，可以按像素数估算
        DifferentiableGazeShift: count_stn_flops,
        # Attention 主要是 QK^T * V，计算量是 O(N^2*D)
        GL_ContextBlock: count_attention_flops
    }

    print("开始计算 FLOPs...")
    # macs = Multiply-Accumulate Operations (通常 1 MAC = 2 FLOPs)
    macs, params = profile(model_pt, inputs=(input_tensor, ), custom_ops=custom_ops, verbose=False)

    gflops = macs * 2 / 1e9  # 转换为 GFLOPs
    print(f"\n================ Results ================")
    print(f"Model: YOLO12s-cascade-EALF")
    print(f"Input Size: 640x640")
    print(f"Parameters: {params / 1e6:.2f} M (与训练日志一致即正确)")
    print(f"GFLOPs:     {gflops:.2f} G")
    print(f"=========================================")

# --- 自定义计算规则 ---
def count_stn_flops(m, x, y):
    # x[0] 是输入的特征图 list [P2, P3, P4]
    # STN 对特征图进行 grid_sample，每个像素约涉及 8 次乘加运算 (双线性插值)
    total_ops = 0
    # 注意：这里假设 x 是个 list，如果不是则直接计算
    inputs = x[0] if isinstance(x, tuple) else x
    if isinstance(inputs, list):
        for feat in inputs:
            B, C, H, W = feat.shape
            # 生成 grid (H*W*2) + 采样 (H*W*C*8)
            total_ops += (H * W * 2) + (H * W * C * 8)
    else:
        B, C, H, W = inputs.shape
        total_ops += (H * W * C * 8)
    
    m.total_ops += torch.DoubleTensor([total_ops])

def count_attention_flops(m, x, y):
    # GL_ContextBlock 包含 LayerNorm, Linear(Proj), Attention
    # x[0]: local, x[1]: global
    # 这里主要估算 Attention 部分: Q(L, D) * K(S, D)^T -> (L, S) * V(S, D) -> (L, D)
    # 简化计算：2 * L * S * D (两次矩阵乘法)
    
    # 此时 x 可能是 list 或 tuple
    input_local = x[0]
    input_global = x[1]
    
    B, C_l, H_l, W_l = input_local.shape
    B, C_g, H_g, W_g = input_global.shape
    
    L = H_l * W_l # Query 序列长度
    S = H_g * W_g # Key/Value 序列长度
    D = C_l       # Embedding 维度 (假设 proj 后维度一致)
    
    # Attention FLOPs
    attn_flops = 2 * L * S * D 
    
    # 加上线性层 (Proj) FLOPs: L * D * D_out
    proj_flops = L * D * D 
    
    m.total_ops += torch.DoubleTensor([attn_flops + proj_flops])

if __name__ == "__main__":
    compute_flops()