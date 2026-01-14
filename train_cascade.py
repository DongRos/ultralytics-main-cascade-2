import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import os
import torch
import logging
# 【引入级联模型所需的组件】
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import CascadeDetectionModel
from ultralytics.utils.loss import CascadeLoss
from ultralytics.utils import LOGGER 

# ==============================================================================
# ----------------- 显卡自动检测与设置 -----------------
# ==============================================================================
# 建议：不要在代码里硬编码 os.environ['CUDA_VISIBLE_DEVICES']，这容易导致找不到显卡。
# 如果你想指定显卡，建议在运行命令前加，例如: CUDA_VISIBLE_DEVICES=1 python train_cascade.py

def check_gpu_availability():
    """检查并打印当前可用的GPU信息"""
    print(f"\n[GPU 检测] PyTorch版本: {torch.__version__}")
    if torch.cuda.is_available():
        cnt = torch.cuda.device_count()
        print(f"[GPU 检测] 发现 {cnt} 个可用 GPU:")
        for i in range(cnt):
            print(f"  - index {i}: {torch.cuda.get_device_name(i)}")
        return True
    else:
        print("[GPU 检测] ❌ 未发现可用 GPU，将使用 CPU 训练 (速度会很慢)")
        return False

# ==============================================================================
# ----------------- 自定义 CascadeTrainer 类 -----------------
# ==============================================================================
class CascadeTrainer(DetectionTrainer):
    """
    自定义训练器，用于支持 CascadeDetectionModel 和 CascadeLoss
    """
    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        重写 get_model，强制使用 CascadeDetectionModel
        """
        # 创建 CascadeDetectionModel 实例
        model = CascadeDetectionModel(cfg, nc=self.data["nc"], verbose=verbose)
        if weights:
            model.load(weights)
        return model

    def get_loss(self):
        """
        重写 get_loss，使用自定义的 CascadeLoss
        """
        return CascadeLoss(self.model)

# ==============================================================================
# ----------------- 日志与路径工具函数 -----------------
# ==============================================================================
def get_unique_dir(project_dir, base_name):
    """确保获取一个唯一的文件夹路径"""
    run_path = os.path.join(project_dir, base_name)
    if not os.path.exists(run_path):
        return run_path
    i = 2
    while True:
        unique_name = f"{base_name}{i}"
        run_path = os.path.join(project_dir, unique_name)
        if not os.path.exists(run_path):
            return run_path
        i += 1

def run_training_task(model_yaml_path, data_yaml_path, device_id='0', train_params=None):
    if train_params is None:
        train_params = {}

    project_dir = 'runs/train/'
    base_run_name = os.path.splitext(os.path.basename(model_yaml_path))[0]
    final_save_dir = get_unique_dir(project_dir, base_run_name)
    
    os.makedirs(final_save_dir, exist_ok=True)
    log_file_path = os.path.join(final_save_dir, 'train_log.txt')
    
    # 配置文件日志处理器
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    LOGGER.addHandler(file_handler)

    try:
        LOGGER.info(f"============================================================")
        LOGGER.info(f"🚀 [级联任务开始] 准备训练模型: {base_run_name}")
        LOGGER.info(f"   - 结果将保存在: {final_save_dir}")
        LOGGER.info(f"   - 完整日志将写入: {log_file_path}")
        LOGGER.info(f"   - 请求使用设备: {device_id}")
        LOGGER.info(f"============================================================")
        
        # 1. 组装参数
        args = {
            'model': model_yaml_path,
            'data': data_yaml_path,
            'project': os.path.dirname(final_save_dir),
            'name': os.path.basename(final_save_dir),
            'device': device_id,
            'exist_ok': True,
            **train_params 
        }

        # 2. 实例化自定义训练器
        trainer = CascadeTrainer(overrides=args)

        # 3. 开始训练
        trainer.train()
        
        LOGGER.info(f"\n✅ [任务完成] 模型 {base_run_name} 训练结束！")

    except Exception as e:
        LOGGER.error(f"\n❌ [训练出错] 任务 '{base_run_name}' 发生错误: {e}", exc_info=True)

    finally:
        LOGGER.removeHandler(file_handler)
        file_handler.close()
        print(f"📄 任务 '{base_run_name}' 的完整日志已保存至: {log_file_path}")
        print(f"============================================================\n")


# ==============================================================================
#                             主程序入口
# ==============================================================================
if __name__ == '__main__':
    # 1. 先检查显卡
    check_gpu_availability()

    DATASET_CONFIG = '/home/liuyadong/ultralytics-main-cascade-2/ultralytics/cfg/datasets/VisDrone.yaml'
    
    # 【修改这里】
    # 建议使用 '0'，ultralytics 会自动找到第一块可用的显卡。
    # 如果你有多个显卡想用第二块，可以在运行脚本时在命令行指定：
    # CUDA_VISIBLE_DEVICES=1 python train_cascade.py
    # 此时代码里的 GPU_DEVICE 依然填 '0' 即可（因为对于程序来说它是第0块可见的卡）
    GPU_DEVICE = '0' 

    model_config_1 = '/home/liuyadong/ultralytics-main-cascade-2/yolo12s-cascade-EALF.yaml'
    
    params_for_task_1 = {
        'imgsz': 640, 
        'epochs': 300, 
        'batch': 16, 
        'workers': 4,
        'optimizer': 'SGD', 
        'cache': False, 
        'close_mosaic': 0,
        'seed': 42,
        'lr0': 0.01,
        'cos_lr': True
    }
    
    run_training_task(
        model_yaml_path=model_config_1, data_yaml_path=DATASET_CONFIG,
        device_id=GPU_DEVICE, train_params=params_for_task_1
    )