import torch
import os

class Config:
    # --- 路径配置 ---
    # 你的项目根目录
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 原始数据路径 (解压后的位置)
    # 脚本会递归扫描这个目录下的所有 .pcap 文件
    RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
    
    # 预处理结果保存路径
    PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
    
    # 确保输出目录存在
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # --- 数据特征参数 ---
    # 1. 载荷特征: 截取前 784 字节 (28x28, 类似 MNIST 图片大小)
    MAX_PAYLOAD_LEN = 784 
    
    # 2. 序列特征: 截取前 20 个包的 (方向, 大小, 时间间隔)
    # 我们不仅提取大小和时间，还可以提取方向(进/出)
    MAX_SEQ_LEN = 20 
    
    # --- 训练超参数 ---
    SEED = 42
    BATCH_SIZE = 128
    EPOCHS_PRETRAIN = 50   # 第一阶段训练轮数
    EPOCHS_FINETUNE = 20   # 第二阶段训练轮数
    
    # 学习率
    LR_PRETRAIN = 1e-3
    LR_FINETUNE = 1e-4
    
    # 温度系数 (对比损失用)
    TEMP = 0.1
    
    # 设备配置 (自动检测显卡)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"配置已加载: 使用设备 {DEVICE}")