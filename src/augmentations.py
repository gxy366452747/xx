import torch
import numpy as np

class TrafficAugmentations:
    """
    混淆增强模块
    包含四种增强策略
    
    输入数据格式:
        sample['payload']: Tensor [784]
        sample['sequence']: Tensor [20, 2]  (dim 0: Packet Length, dim 1: IAT)
    """
    
    def __init__(self, p_padding=0.5, p_jitter=0.5, p_drop=0.4, p_mask=0.4):
        # 概率超参数：控制每种策略触发的可能性
        self.p_padding = p_padding  # 策略1: 包长填充
        self.p_jitter = p_jitter    # 策略2: 时间抖动
        self.p_drop = p_drop        # 策略3: 随机丢包
        self.p_mask = p_mask        # 策略4: 载荷掩码

    def __call__(self, sample):
        """
        前向传播：对输入样本应用随机增强
        """
        # 深拷贝，避免修改原始数据
        aug_payload = sample['payload'].clone()
        aug_seq = sample['sequence'].clone() # Shape: [Seq_Len, 2]

        # ==========================================
        # 策略 1: 包长填充 
        # 向数据包尾部添加随机字节
        if np.random.rand() < self.p_padding:
            # 随机选择 40% 的包进行填充
            mask = torch.rand(aug_seq.shape[0]) < 0.4
            # 制造噪音：增加 0.01~0.1 的长度
            noise = torch.rand(aug_seq.shape[0]) * 0.1 + 0.01
            # 被选中的地方加噪音，没选中的保持原样
            aug_seq[:, 0] = torch.where(mask, aug_seq[:, 0] + noise, aug_seq[:, 0])
            # 截断，保证不超过 1.0 (归一化上限)
            aug_seq[:, 0] = torch.clamp(aug_seq[:, 0], 0.0, 1.0)

        # ==========================================
        # 策略 2: 时间抖动 (Timing Jitter)
        # ==========================================
        # 模拟网络延迟抖动或人为引入的发送延迟，破坏基于包到达间隔时间(IAT)的时序特征。
        if np.random.rand() < self.p_jitter:
            # 生成高斯噪声，均值0，标准差0.03(有正有负)
            jitter = torch.randn(aug_seq.shape[0]) * 0.03
            # 把噪声加到“时间间隔”特征上
            aug_seq[:, 1] = aug_seq[:, 1] + jitter
            
            aug_seq[:, 1] = torch.abs(aug_seq[:, 1])

        # ==========================================
        # 策略 3: 随机丢包 
        # ==========================================
        #  随机丢弃序列中的部分数据包，模拟不稳定的网络环境或中间设备的过滤行为。
        # 增强模型在缺失局部信息情况下的鲁棒性。
        if np.random.rand() < self.p_drop:
            seq_len = aug_seq.shape[0]
            # 随机选择 1-3 个包进行丢弃
            num_drop = np.random.randint(1, 4)
            drop_indices = np.random.choice(seq_len, num_drop, replace=False)
            
            # 我们不改变 Tensor 形状 (否则 Batch 训练会报错)
            # 而是将丢弃的包特征置为 0 
            for idx in drop_indices:
                aug_seq[idx, :] = 0.0

        # ==========================================
        # 策略 4: 载荷掩码 (Payload Masking) 
        # ==========================================
        # 随机遮盖原始载荷中的连续片段。
        # 强迫模型利用剩余的字节上下文来重建特征表示。
        if np.random.rand() < self.p_mask:
            payload_len = aug_payload.shape[0]
            # 遮盖长度: 总长度的 10% - 20%
            mask_len = int(payload_len * np.random.uniform(0.1, 0.2))
            # 随机起始位置
            start = np.random.randint(0, payload_len - mask_len)
            
            # 把这一段全部涂成 0
            aug_payload[start : start+mask_len] = 0.0

        return {'payload': aug_payload, 'sequence': aug_seq, 'label': sample['label']}

# --- 单元测试 ---
if __name__ == "__main__":
    
    print("正在测试混淆增强模块...")
    
    # 模拟一条数据
    dummy_data = {
        'payload': torch.ones(784),      # 全1载荷
        'sequence': torch.ones(20, 2),   # 全1序列
        'label': 0
    }
    
    # 强制开启所有增强
    augmentor = TrafficAugmentations(p_padding=1.0, p_jitter=1.0, p_drop=1.0, p_mask=1.0)
    aug_data = augmentor(dummy_data)
    
    print("\n[原始序列前5行]:\n", dummy_data['sequence'][:5])
    print("\n[增强后序列前5行] (注意数值变化和0值):")
    print(aug_data['sequence'][:5])
    
    print("\n[载荷掩码测试] 0值数量:", (aug_data['payload'] == 0).sum().item())
    print("测试完成！")