import torch
from torch.utils.data import Dataset
from src.augmentations import TrafficAugmentations

class ContrastiveDataset(Dataset):
    """
    第一阶段：多视图对比学习 Dataset
    特点：对于每一个样本，返回 4 个经过不同策略增强的视图
    """
    def __init__(self, data_list):
        self.data = data_list
        
        # --- 定义 4 种混淆增强器 ---
        # 1. 包长填充
        self.aug_pad = TrafficAugmentations(p_padding=0.8, p_jitter=0.1, p_drop=0.1, p_mask=0.1)
        
        # 2. 时间抖动
        self.aug_jitter = TrafficAugmentations(p_padding=0.1, p_jitter=0.8, p_drop=0.1, p_mask=0.1)
        
        # 3. 随机丢包
        self.aug_drop = TrafficAugmentations(p_padding=0.1, p_jitter=0.1, p_drop=0.8, p_mask=0.1)
        
        # 4. 载荷掩码

        self.aug_mask = TrafficAugmentations(p_padding=0.1, p_jitter=0.1, p_drop=0.1, p_mask=0.8)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raw_sample = self.data[idx]
        
        base_data = {
            'payload': raw_sample['payload'],
            'sequence': raw_sample['sequence'],
            'label': raw_sample['label']
        }
        
        # 生成 4 个视图
        view1 = self.aug_pad(base_data)    # 视图A: 填充为主
        view2 = self.aug_jitter(base_data) # 视图B: 抖动为主
        view3 = self.aug_drop(base_data)   # 视图C: 丢包为主
        view4 = self.aug_mask(base_data)   # 视图D: 掩码为主
        
        return view1, view2, view3, view4

class FinetuneDataset(Dataset):
    """
    第二阶段：微调专用 Dataset 
    """
    def __init__(self, data_list, augment=False):
        self.data = data_list
        self.augment = augment
        # 微调时使用温和的混合增强
        #每个的超参数较小，加一些微小的干扰。
        self.augmentor = TrafficAugmentations(p_padding=0.2, p_jitter=0.2, p_drop=0.0, p_mask=0.0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # 如果是训练时 (augment=True)，增加难度。
        # 如果是测试时 (augment=False)，去掉干扰
        if self.augment:
            sample = self.augmentor(sample)
        return {
            'payload': sample['payload'],
            'sequence': sample['sequence'],
            'label': sample['label']
        }