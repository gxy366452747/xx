#极端不平衡实验 (Listen to Minority)
#少样本学习实验 (Few-Shot Learning)
# 只用 1% 的恶意样本训练，测试在 99% 良性样本上的性能。
# 对比有无预训练的差异。
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import random
import numpy as np
from sklearn.metrics import classification_report

from src.config import Config
from src.models import TrafficEncoder
from src.augmentations import TrafficAugmentations

# --- 定义特定的 Dataset ---
class FewShotDataset(Dataset):
    def __init__(self, data_list, augment=False):
        self.data = data_list
        self.augment = augment
        self.augmentor = TrafficAugmentations(p_padding=0.2, p_jitter=0.2, p_drop=0.0, p_mask=0.0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.augment:
            sample = self.augmentor(sample)
        return {
            'payload': sample['payload'],
            'sequence': sample['sequence'],
            'label': sample['label']
        }

class TrafficClassifier(nn.Module):
    def __init__(self, encoder, num_classes=2):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, payload, sequence):
        features = self.encoder(payload, sequence)
        return self.classifier(features)

def train_and_evaluate(train_loader, test_loader, use_pretrain=False, description=""):
    print(f"\n⚡ 开始实验: {description}")
    
    # 初始化模型
    encoder = TrafficEncoder(feature_dim=128)
    if use_pretrain:
        checkpoint_path = "checkpoints/pretrain/encoder_epoch_50.pth"
        if os.path.exists(checkpoint_path):
            encoder.load_state_dict(torch.load(checkpoint_path))
            print("   ✅ [有预训练] 权重加载成功")
        else:
            print("   ⚠️ 未找到预训练权重！")
    else:
        print("   📉 [无预训练] 使用随机初始化")
        
    model = TrafficClassifier(encoder).to(Config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss() # 这里用普通 CE 即可

    # 训练 10 轮
    EPOCHS = 10
    best_f1 = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            p = batch['payload'].to(Config.DEVICE)
            s = batch['sequence'].to(Config.DEVICE)
            y = batch['label'].to(Config.DEVICE)
            
            optimizer.zero_grad()
            output = model(p, s)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
    # 测试
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            p = batch['payload'].to(Config.DEVICE)
            s = batch['sequence'].to(Config.DEVICE)
            y = batch['label'].to(Config.DEVICE)
            output = model(p, s)
            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    malware_f1 = report['1']['f1-score']
    accuracy = report['accuracy']
    
    print(f"   🏁 结果 ({description}): Accuracy = {accuracy:.4f} | Malware F1 = {malware_f1:.4f}")
    return accuracy, malware_f1

def main():
    print(f"🧪 [Listen to Minority] 极端不平衡实验 | 设备: {Config.DEVICE}")

    # 1. 加载数据
    data_path = os.path.join(Config.PROCESSED_DATA_DIR, "all_traffic_data.pt")
    all_data = torch.load(data_path)
    labeled_data = [d for d in all_data if d['label'].item() != -1]
    
    # 2. 手动构造极端不平衡数据集
    # 分离良性和恶意
    benign_data = [d for d in labeled_data if d['label'].item() == 0]
    malware_data = [d for d in labeled_data if d['label'].item() == 1]
    
    print(f"原始分布: 良性 {len(benign_data)} | 恶意 {len(malware_data)}")
    
    # ----------------------------------------------------
    # 🌟 核心操作：保留 1% 的恶意样本用于训练 (Few-shot)
    # ----------------------------------------------------
    MALWARE_KEEP_RATIO = 0.01  # 只留 1% 的恶意样本！
    num_malware_train = int(len(malware_data) * 0.8 * MALWARE_KEEP_RATIO)
    
    # 随机打乱
    random.shuffle(benign_data)
    random.shuffle(malware_data)
    
    # 划分训练/测试 (测试集保持原样，以反映真实评估)
    # 训练集：80%良性 + 1%恶意
    # 测试集：20%良性 + 20%恶意 (测试集要大，才能测得准)
    
    # 切分索引
    b_split = int(len(benign_data) * 0.8)
    m_split = int(len(malware_data) * 0.8)
    
    train_benign = benign_data[:b_split]
    test_benign = benign_data[b_split:]
    
    # 训练集里的恶意样本，我们只取极少一部分！
    train_malware_full = malware_data[:m_split]
    train_malware_few = train_malware_full[:num_malware_train] # 只有几十个样本
    
    test_malware = malware_data[m_split:]
    
    print(f"🔥 构造极端训练集:")
    print(f"   良性样本: {len(train_benign)} (海量)")
    print(f"   恶意样本: {len(train_malware_few)} (极少! 只有原始的 {MALWARE_KEEP_RATIO*100}%)")
    print(f"   不平衡比例: 1 : {len(train_benign)/len(train_malware_few):.1f}")

    # 构造 DataLoader
    train_set_few = FewShotDataset(train_benign + train_malware_few, augment=True)
    test_set = FewShotDataset(test_benign + test_malware, augment=False)
    
    train_loader = DataLoader(train_set_few, batch_size=64, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, drop_last=True)

    print("="*40)
    
    # 3. 运行对比实验
    # 实验 A: 无预训练
    acc_base, f1_base = train_and_evaluate(train_loader, test_loader, use_pretrain=False, description="Baseline (No Pretrain)")
    
    # 实验 B: 有预训练
    acc_ours, f1_ours = train_and_evaluate(train_loader, test_loader, use_pretrain=True, description="Ours (With Pretrain)")
    
    print("="*40)
    print("📊 最终对比报告:")
    print(f"Baseline F1: {f1_base:.4f}")
    print(f"Ours F1    : {f1_ours:.4f}")
    
    improve = f1_ours - f1_base
    print(f"🚀 提升幅度: {improve*100:.2f}%")
    
    if improve > 0.05:
        print("✅ 结论: 在极度缺乏恶意样本的情况下，你的方法完爆 Baseline！(Listen to Minority)")
    else:
        print("🤔 结论: 还需要进一步调整参数。")

if __name__ == "__main__":
    main()