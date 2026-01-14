import torch
import torch.nn as nn
from ultralytics.nn.tasks import CascadeDetectionModel

def diagnose_weights(model_cfg, weights_path):
    print(f"🔍 正在诊断权重文件: {weights_path}")
    
    # 1. 初始化模型
    model = CascadeDetectionModel(cfg=model_cfg, nc=10) # 记得 nc=10
    
    # 2. 加载权重
    try:
        ckpt = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(ckpt['model'].float().state_dict())
        print("✅ 权重加载过程无报错")
    except Exception as e:
        print(f"❌ 权重加载失败: {e}")
        return

    # 3. 寻找 CoarseDetect 层
    # 我们需要找到生成 gaze_params 的那一层。
    # 通常它在 CoarseDetect 内部，或者是一个独立的 MLP。
    # 假设你的 CoarseDetect 是第 21 层
    coarse_head = model.model[21]
    
    print("\n📊 [CoarseDetect 权重检查]")
    print(f"层类型: {type(coarse_head)}")
    
    # 尝试打印该层的一些关键权重的统计信息
    # 我们遍历它的所有子模块，看有没有 weights 接近全 0 的情况
    has_learnable_params = False
    for name, param in coarse_head.named_parameters():
        has_learnable_params = True
        mean_val = param.data.mean().item()
        std_val = param.data.std().item()
        max_val = param.data.max().item()
        
        print(f"  - 参数: {name}")
        print(f"    均值: {mean_val:.6f} | 标准差: {std_val:.6f} | 最大值: {max_val:.6f}")
        
        if std_val < 1e-6:
            print(f"    ⚠️ 警告: 该参数几乎没有变化（方差接近0），可能未经过有效训练！")
    
    if not has_learnable_params:
        print("❌ 错误：在 CoarseDetect 层中找不到可学习的参数！请检查代码定义。")

    # 4. 模拟一次推理看看 Raw Output
    print("\n🧪 [模拟推理测试]")
    dummy_input = torch.randn(1, 3, 640, 640)
    model.eval()
    with torch.no_grad():
        # 我们只运行到 CoarseDetect
        # 这里需要一点技巧，我们利用 forward hook 截取
        outputs = {}
        def hook(module, input, output):
            outputs['out'] = output
        
        coarse_head.register_forward_hook(hook)
        try:
            model(dummy_input)
        except:
            pass # 后面的层报错没关系，我们只要 CoarseDetect 的输出
            
        raw_out = outputs.get('out')
        if raw_out is not None:
            # 假设 raw_out 是 [Saliency, Uncertainty] 或者直接是 GazeParams
            # 这取决于你的 CoarseDetect 具体怎么写的
            print(f"  CoarseDetect 输出类型: {type(raw_out)}")
            if isinstance(raw_out, torch.Tensor):
                print(f"  输出形状: {raw_out.shape}")
                print(f"  输出数值(前10个): {raw_out.flatten()[:10].tolist()}")
            elif isinstance(raw_out, list):
                print(f"  输出是一个列表，长度: {len(raw_out)}")
                # 打印第一个张量的统计
                t = raw_out[0]
                if isinstance(t, torch.Tensor):
                    print(f"  列表第一个张量均值: {t.float().mean().item():.4f}")
        else:
            print("❌ 无法截获输出")

if __name__ == '__main__':
    diagnose_weights(
        model_cfg='/home/liuyadong/ultralytics-main-cascade/yolo12s-cascade.yaml',
        weights_path='/home/liuyadong/ultralytics-main-cascade/runs/train/yolo12s-cascade/weights/best.pt'
    )