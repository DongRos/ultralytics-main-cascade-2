import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
from ultralytics.nn.tasks import CascadeDetectionModel
from ultralytics.utils.torch_utils import select_device

def apply_stn_to_rgb(img_tensor, gaze_params):
    """
    手动对 RGB 原图应用 STN 变换
    """
    B, C, H, W = img_tensor.size()
    tx = gaze_params[:, 0]
    ty = gaze_params[:, 1]
    s  = gaze_params[:, 2]

    tx_trans = (tx - 0.5) * 2
    ty_trans = (ty - 0.5) * 2
    
    theta = torch.zeros(B, 2, 3, device=img_tensor.device)
    theta[:, 0, 0] = s
    theta[:, 1, 1] = s
    theta[:, 0, 2] = tx_trans
    theta[:, 1, 2] = ty_trans

    grid = F.affine_grid(theta, img_tensor.size(), align_corners=False)
    warped_rgb = F.grid_sample(img_tensor, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    return warped_rgb

def visualize_stn(img_path, model_cfg, weights_path, device='0'):
    # 1. 准备环境
    device = select_device(device)
    save_dir = 'runs/stn_visualization'
    if os.path.exists(save_dir):
        import shutil
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    
    print(f"🚀 开始 STN 可视化...")
    
    # 2. 加载模型
    model = CascadeDetectionModel(cfg=model_cfg, nc=10, verbose=False)
    if weights_path and os.path.exists(weights_path):
        ckpt = torch.load(weights_path, map_location=device)
        model.load_state_dict(ckpt['model'].float().state_dict())
        print("✅ 权重加载成功")
    else:
        print("⚠️ 未找到权重，使用随机参数")
    
    model.to(device).eval()

    # --- [关键修复] 自动查找层索引 ---
    stn_layer_idx = -1
    coarse_layer_idx = -1
    
    print("\n🔍 正在自动定位模块层索引...")
    for i, m in enumerate(model.model):
        name = m.__class__.__name__
        if name == 'DifferentiableGazeShift':
            stn_layer_idx = i
            print(f"  -> 找到 STN (DifferentiableGazeShift): Layer {i}")
        elif name == 'CoarseDetect':
            coarse_layer_idx = i
            print(f"  -> 找到 CoarseDetect: Layer {i}")
            
    if stn_layer_idx == -1 or coarse_layer_idx == -1:
        print("❌ 错误：无法在模型中找到 CoarseDetect 或 STN 模块！请检查 YAML 配置。")
        return
    # -------------------------------

    # 3. 注册 Hook (STN)
    captured_data = {'params': None, 'features': []}

    def hook_fn(module, input, output):
        # [防御性检查] 防止 input[1] 越界
        if len(input) > 1:
            captured_data['params'] = input[1].detach()
        else:
            print(f"⚠️ Warning: STN Layer {stn_layer_idx} 仅接收到 1 个输入。tasks.py 逻辑可能未生效。")
            # 尝试造一个假参数防止脚本崩溃
            captured_data['params'] = torch.tensor([[0.5, 0.5, 1.0]], device=input[0].device)
            
        captured_data['features'] = output
    
    model.model[stn_layer_idx].register_forward_hook(hook_fn)

    # 4. 图像预处理
    original_cv_img = cv2.imread(img_path)
    if original_cv_img is None:
        raise FileNotFoundError(f"找不到图片: {img_path}")
        
    h0, w0 = original_cv_img.shape[:2]
    img = cv2.resize(original_cv_img, (640, 640))
    img_in = img[:, :, ::-1].transpose(2, 0, 1)
    img_in = np.ascontiguousarray(img_in)
    img_tensor = torch.from_numpy(img_in).to(device).float() / 255.0
    img_tensor = img_tensor[None]

    # 5. 运行推理
    with torch.no_grad():
        model(img_tensor)

    # 6. 可视化参数 & RGB 切片
    params = captured_data['params']
    if params is None: return

    tx, ty, s = params[0].tolist()
    print(f"\n🔍 [CoarseDetect 决策结果]")
    print(f"   - 中心点: ({tx:.4f}, {ty:.4f})")
    print(f"   - 缩放因子: {s:.4f}")
    print(f"   => 视线坐标: ({tx*w0:.0f}, {ty*h0:.0f})")

    rgb_crop_tensor = apply_stn_to_rgb(img_tensor, params)
    rgb_crop = rgb_crop_tensor[0].cpu().numpy().transpose(1, 2, 0)
    rgb_crop = (rgb_crop * 255).astype(np.uint8)
    rgb_crop = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(f"{save_dir}/1_RGB_Real_Crop.jpg", rgb_crop)
    display_img = np.hstack([img, rgb_crop])
    cv2.imwrite(f"{save_dir}/0_Comparison_RGB.jpg", display_img)
    print(f"✅ 保存 RGB 对比图: {save_dir}/0_Comparison_RGB.jpg")

    # 7. 可视化 Saliency Map (第二次推理 Hook CoarseDetect)
    coarse_data = {'out': None}
    def coarse_hook(module, input, output):
        coarse_data['out'] = output
    
    # 移除旧 Hook，注册新 Hook
    # model.model[stn_layer_idx].remove_hook() # PyTorch 旧版本可能不支持，这里重新推理一次无妨
    model.model[coarse_layer_idx].register_forward_hook(coarse_hook)
    
    with torch.no_grad():
        model(img_tensor)
        
    if coarse_data['out'] is not None:
        # coarse_outputs[0] 是分辨率最高的特征 (P2 或 P3)
        raw_feat = coarse_data['out'][0] 
        saliency = raw_feat[0, 0].cpu().numpy()
        
        saliency_norm = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        heatmap = cv2.applyColorMap((saliency_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (w0, h0))
        
        overlay = cv2.addWeighted(original_cv_img, 0.6, heatmap, 0.4, 0)
        cv2.imwrite(f"{save_dir}/-1_Saliency_Map_Overlay.jpg", overlay)
        print(f"🔥 保存热力图: {save_dir}/-1_Saliency_Map_Overlay.jpg")

    # 8. 可视化特征图切片
    feature_maps = captured_data['features']
    if isinstance(feature_maps, list):
        for i, feat in enumerate(feature_maps):
            heatmap = torch.mean(feat[0], dim=0).cpu().numpy()
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap_view = cv2.resize(heatmap_color, (320, 320), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(f"{save_dir}/2_Feature_Crop_Level_{i}.jpg", heatmap_view)

    print("\n🎉 可视化完成！")

if __name__ == '__main__':
    # 替换路径
    img_path = '/home/liuyadong/ultralytics-main-cascade/图片素材/0000146_01678_d_0000066.jpg'
    
    # 请确保 yaml 文件和你训练权重是对应的 (P2版用P2 yaml)
    visualize_stn(
        img_path=img_path,
        model_cfg='/home/liuyadong/ultralytics-main-cascade/yolo12s-cascade.yaml', 
        weights_path='/home/liuyadong/ultralytics-main-cascade/runs/train/yolo12s-cascade2-多尺度加多处P2/weights/best.pt'
    )