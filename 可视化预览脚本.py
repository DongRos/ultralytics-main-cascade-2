import torch
from ultralytics.nn.tasks import CascadeDetectionModel
from ultralytics.utils.torch_utils import select_device
import cv2
import os
import numpy as np

def run_visualization():
    # 1. 设置
    model_yaml = '/home/liuyadong/ultralytics-main-cascade/yolo12s-cascade.yaml'
    # weights_path = 'runs/train/yolo12s-cascade12/weights/best.pt' # 训练好的权重
    weights_path ='/home/liuyadong/ultralytics-main-cascade/runs/train/yolo12s-cascade2-多尺度加多处P2/weights/best.pt'# 如果还没有训练好的权重，设为 None，只看随机初始化的效果
    
    # 找一张你的数据集里的图片
    img_path = '/home/liuyadong/ultralytics-main-cascade/图片素材/0000146_01678_d_0000066.jpg'
    
    device = select_device('0')
    
    # 2. 加载模型
    print("正在加载模型...")
    model = CascadeDetectionModel(cfg=model_yaml, verbose=True)
    if weights_path and os.path.exists(weights_path):
        model.load(torch.load(weights_path))
    model.to(device)
    model.eval() # 开启验证模式

    # 3. 预处理图片
    # YOLO 需要简单的预处理：读取 -> Resize -> Transpose -> Normalize
    original_img = cv2.imread(img_path)
    if original_img is None:
        print(f"❌ 找不到图片: {img_path}")
        return

    # Resize 到 640x640 (简单缩放，不带 Letterbox，仅作演示)
    img = cv2.resize(original_img, (640, 640))
    # BGR -> RGB
    img = img[:, :, ::-1].transpose(2, 0, 1) # HWC -> CHW
    img = np.ascontiguousarray(img)
    img_tensor = torch.from_numpy(img).to(device).float()
    img_tensor /= 255.0 # 0-255 -> 0.0-1.0
    if len(img_tensor.shape) == 3:
        img_tensor = img_tensor[None] # 增加 Batch 维度: [1, 3, 640, 640]

    # 4. 运行推理 (开启 visualize=True)
    print("开始推理并生成可视化图...")
    # 清理旧的 debug 文件夹
    if os.path.exists('runs/debug_vis'):
        import shutil
        shutil.rmtree('runs/debug_vis')
        
    with torch.no_grad():
        # 这里传入 visualize=True 会触发我们在 _predict_once 里写的 save_debug_visualization
        model.predict(img_tensor, visualize=True) 

    print("\n✅ 可视化完成！请查看 runs/debug_vis 文件夹。")
    print("  - Heatmap: 粗检测关注的区域")
    print("  - STN_Cropped: 经过 STN 变换后的特征图(平均激活)")

if __name__ == '__main__':
    run_visualization()