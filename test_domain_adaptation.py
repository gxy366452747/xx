#泛化实验（第二次）
#跨领域小样本自适应实验 (Few-Shot Domain Adaptation)
#加载 USTC 的模型，然后用 10% 的 数据快速微调分类器，用剩下 90% 的 数据进行测试。

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
import os

from src.config import Config
from src.dataset import FinetuneDataset
from src.models import TrafficEncoder
from src.losses import SymmetricCrossEntropy

# --- 复用分类器 ---
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
    print(f"开始跨领域小样本自适应实验 (Few-Shot Domain Adaptation) | 设备: {Config.DEVICE}")
    print("目标：用 10% 的新环境数据校准分类边界，测试剩余 90% 数据。")
    print("="*60)

    # 1. 加载 数据
    adpt_data_path = "./DeepTraffic-CL/data/processed/cicids2017_traffic_data.pt"
    if not os.path.exists(adpt_data_path):
        print(" 找不到数据！")
        return
    adpt_data = torch.load(adpt_data_path)
    
    # 2. 划分 10% 适应集 (Adaptation) 和 90% 测试集 (Test)
    
    adapt_size = int(0.1 * len(adpt_data))
    test_size = len(adpt_data) - adapt_size
    adapt_dataset, test_dataset = random_split(
        FinetuneDataset(adpt_data, augment=True), # 适应集依然开启轻微增强
        [adapt_size, test_size]
    )
    test_dataset.dataset.augment = False # 测试集绝不增强
    
    print(f"数据划分: 10% 适应集 ({adapt_size}个) | 90% 测试集 ({test_size}个)")

    adapt_loader = DataLoader(adapt_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # 3. 加载 USTC 训练好的
    encoder = TrafficEncoder(feature_dim=128)
    model = TrafficClassifier(encoder).to(Config.DEVICE)
    model_path = "./DeepTraffic-CL/checkpoints/finetune/best_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
    
    # 4. 冻结Encoder，训练Classifier
    print("策略: 冻结预训练特征提取器，仅微调分类头使其适应新领域。")
    for param in model.encoder.parameters():
        param.requires_grad = False
        
    # 重置分类器参数（重新适应 ）
    for layer in model.classifier:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)
    criterion = SymmetricCrossEntropy(alpha=0.1, beta=1.0, num_classes=2)

    # 5. 快速适应训练 (只跑 10 个 Epoch)
    print(" 开始快速环境适应 (Adaptation)...")
    for epoch in range(10):
        model.train()
        total_loss = 0
        for batch in adapt_loader:
            p = batch['payload'].to(Config.DEVICE)
            s = batch['sequence'].to(Config.DEVICE)
            y = batch['label'].to(Config.DEVICE)
            
            optimizer.zero_grad()
            loss = criterion(model(p, s), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # 简单打印一下
        if (epoch+1) % 5 == 0:
            print(f"   [Epoch {epoch+1}/10] 适应中... Loss: {total_loss/len(adapt_loader):.4f}")

    # 6. 重新在 90% 的新数据上测试
    print("="*60)
    print(" 适应完毕！开始对剩余 90% 的新数据进行最终测试...")
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            p = batch['payload'].to(Config.DEVICE)
            s = batch['sequence'].to(Config.DEVICE)
            y = batch['label'].to(Config.DEVICE)
            
            outputs = model(p, s)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # 7. 出报告
    print("\n" + "="*60)
    print("领域自适应后泛化能力评估报告 (Post-Adaptation)")
    print("="*60)
    print(classification_report(all_labels, all_preds, target_names=['Benign(良性)', 'Malware(恶意)'], digits=4))
    
    cm = confusion_matrix(all_labels, all_preds)
    print("\n混淆矩阵 (Confusion Matrix):")
    print(f"真正例(TN): {cm[0][0]:<6} | 假正例(FP/误报): {cm[0][1]:<6}")
    print(f"假反例(FN/漏报): {cm[1][0]:<6} | 真反例(TP): {cm[1][1]:<6}")

if __name__ == "__main__":
    main()