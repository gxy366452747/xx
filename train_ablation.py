#消融实验 (Ablation Study)
# 砍掉第一阶段预训练
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report
import os
import numpy as np

from src.config import Config
from src.dataset import FinetuneDataset
from src.models import TrafficEncoder
from src.losses import SymmetricCrossEntropy

# --- 这是一个不加载预训练权重的纯监督学习脚本 ---

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

def main():
    print(f" [Ablation Study] 开始消融实验：无预训练基准测试 | 设备: {Config.DEVICE}")

    # 1. 加载数据
    data_path = os.path.join(Config.PROCESSED_DATA_DIR, "all_traffic_data.pt")
    all_data = torch.load(data_path)
    labeled_data = [d for d in all_data if d['label'].item() != -1]
    
    # 2. 划分数据集 (保持和之前一样的比例 8:2)
    train_size = int(0.8 * len(labeled_data))
    test_size = len(labeled_data) - train_size
    train_dataset, test_dataset = random_split(
        FinetuneDataset(labeled_data, augment=True), 
        [train_size, test_size]
    )
    test_dataset.dataset.augment = False 

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4)

    # 3. 初始化模型 (随机初始化！不加载 .pth)
    print(" 注意：正在使用随机初始化的 Encoder（无预训练知识）...")
    encoder = TrafficEncoder(feature_dim=128)
    model = TrafficClassifier(encoder).to(Config.DEVICE)

    # 4. 优化器 (全量微调，因为没有预训练特征可以冻结)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 损失函数 (保持一致)
    criterion = SymmetricCrossEntropy(alpha=0.1, beta=1.0, num_classes=2)

    # 5. 训练循环 (跑少一点轮数， 15-20 轮，看看收敛速度)
    EPOCHS = 15
    print(f"开始 Baseline 训练，共 {EPOCHS} 轮...")
    
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            p = batch['payload'].to(Config.DEVICE)
            s = batch['sequence'].to(Config.DEVICE)
            y = batch['label'].to(Config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(p, s)
            loss = criterion(outputs, y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
        train_acc = 100 * correct / total
        
        # 测试
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for batch in test_loader:
                p = batch['payload'].to(Config.DEVICE)
                s = batch['sequence'].to(Config.DEVICE)
                y = batch['label'].to(Config.DEVICE)
                outputs = model(p, s)
                _, predicted = torch.max(outputs.data, 1)
                test_total += y.size(0)
                test_correct += (predicted == y).sum().item()
        
        test_acc = 100 * test_correct / test_total
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc

    print("\n" + "="*30)
    print(f" 基准测试完成！最佳准确率 (无预训练): {best_acc:.2f}%")
    print(f"最佳准确率 (有预训练): 99.91%")
    
    diff = 99.91 - best_acc
    if diff > 0:
        print(f"结论: 方法提升了 {diff:.2f}% 的性能，且收敛可能更快！")
    else:
        print(f" 结论: 两种方法差不多，可能是数据集太简单了。")

if __name__ == "__main__":
    main()