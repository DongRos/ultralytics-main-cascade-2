import torch
import cv2
import numpy as np
import os
from ultralytics.nn.tasks import CascadeDetectionModel
from ultralytics.utils.torch_utils import select_device

def visualize_coarse_maps(img_path, model_cfg, weights_path, device='0'):
    # 1. 初始化环境
    device = select_device(device)
    output_dir = 'runs/coarse_analysis'
    os.makedirs(output_dir, exist_ok=True)

    # 2. 加载模型
    print(f"正在加载模型: {model_cfg}")
    # [修复] 强制指定 nc=10 (适配 VisDrone 训练出的权重)
    # 如果你不确定是多少类，可以先加载 checkpoint 读取 'model' 里的 nc，但手动指定最快
    model = CascadeDetectionModel(cfg=model_cfg, nc=10)
    if weights_path and os.path.exists(weights_path):
        ckpt = torch.load(weights_path, map_location=device)
        model.load_state_dict(ckpt['model'].float().state_dict())
        print("✅ 权重加载成功")
    model.to(device).eval()

    # 3. 图像预处理
    ori_img = cv2.imread(img_path)
    h, w = ori_img.shape[:2]
    img = cv2.resize(ori_img, (640, 640))
    img = img[:, :, ::-1].transpose(2, 0, 1) # BGR to RGB, HWC to CHW
    img = np.ascontiguousarray(img)
    img_tensor = torch.from_numpy(img).to(device).float() / 255.0
    img_tensor = img_tensor[None] # Add batch dimension [1, 3, 640, 640]

    # 4. 钩子函数 (Hook) 提取中间层输出
    # 我们需要拦截 CoarseDetect (Layer 21) 的输出
    coarse_output = None
    def hook_fn(module, input, output):
        nonlocal coarse_output
        coarse_output = output

    # 绑定 Hook 到 CoarseDetect 层
    # 注意：根据你的 YAML，CoarseDetect 是第 21 层
    target_layer = model.model[21]
    handle = target_layer.register_forward_hook(hook_fn)

    # 5. 推理
    with torch.no_grad():
        model(img_tensor)
    handle.remove() # 移除 Hook

    if coarse_output is None:
        print("❌ 未捕获到 CoarseDetect 输出，请检查层索引！")
        return

    # 6. 解析并保存图层
    # 假设 coarse_output 是 [P3_out, P4_out, P5_out] 的列表
    # 我们以分辨率最高的 P3 (通常在索引 0) 为例
    # P3_out shape: [1, 2, 80, 80] -> 2个通道分别是 Saliency 和 Uncertainty
    feat = coarse_output[0][0].detach().cpu().numpy() # [2, 80, 80]
    
    maps = {
        'Saliency_Map': feat[0],     # 通道 0
        'Uncertainty_Map': feat[1]   # 通道 1
    }

    for name, m in maps.items():
        # 归一化到 0-255
        m_norm = (m - m.min()) / (m.max() - m.min() + 1e-8)
        m_img = (m_norm * 255).astype(np.uint8)
        
        # 调整大小回原图尺寸，方便对比
        m_img = cv2.resize(m_img, (w, h))
        
        # 应用热力图
        color_map = cv2.applyColorMap(m_img, cv2.COLORMAP_JET)
        
        # 与原图叠加 (Alpha 混合) 以便查看对应关系
        fusion = cv2.addWeighted(ori_img, 0.6, color_map, 0.4, 0)
        
        # 保存
        cv2.imwrite(f"{output_dir}/{name}.jpg", color_map)
        cv2.imwrite(f"{output_dir}/{name}_overlay.jpg", fusion)
        print(f"  -> 已保存: {output_dir}/{name}_overlay.jpg")

if __name__ == '__main__':
    # 修改以下路径进行运行
    visualize_coarse_maps(
        img_path='/home/liuyadong/ultralytics-main-cascade/图片素材/9999953_00000_d_0000025.jpg',
        model_cfg='yolo12s-cascade.yaml',
        weights_path='/home/liuyadong/ultralytics-main-cascade/runs/train/yolo12s-cascade/weights/best.pt'
    )