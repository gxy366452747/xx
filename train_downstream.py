import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

from src.config import Config
from src.dataset import FinetuneDataset
from src.models import TrafficEncoder
from src.losses import SymmetricCrossEntropy

# --- 定义下游分类模型 ---
class TrafficClassifier(nn.Module):
    def __init__(self, encoder, num_classes=2):
        super().__init__()
        self.encoder = encoder
        # 定义分类头：输入 128 维特征 -> 输出 2 类 (良性/恶意)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2), # 防止过拟合
            nn.Linear(64, num_classes)
        )

    def forward(self, payload, sequence):
        # 1. 提取特征 (此时 Encoder 是预训练好的)
        features = self.encoder(payload, sequence)
        # 2. 分类
        logits = self.classifier(features)
        return logits

def main():
    print(f"[Phase 2] 开始下游任务微调 (Fine-tuning) | 设备: {Config.DEVICE}")

    # 1. 加载数据
    data_path = os.path.join(Config.PROCESSED_DATA_DIR, "all_traffic_data.pt")
    if not os.path.exists(data_path):
        print(" 找不到数据文件！")
        return
    
    all_data = torch.load(data_path)
    
    # 过滤掉无标签数据 (Label = -1 的)
   
    labeled_data = [d for d in all_data if d['label'].item() != -1]
    print(f"原始数据: {len(all_data)} -> 有标签数据: {len(labeled_data)}")
    
    if len(labeled_data) == 0:
        print("警告：没有找到有标签数据(label 0或1)。请检查 preprocess.py 的打标逻辑。")
        return

    # 2. 划分训练集和测试集 (8:2)
    train_size = int(0.8 * len(labeled_data))
    test_size = len(labeled_data) - train_size
    train_dataset, test_dataset = random_split(
        FinetuneDataset(labeled_data, augment=True), # 训练集开启弱增强
        [train_size, test_size]
    )
    # 测试集不增强
    test_dataset.dataset.augment = False 

    print(f"训练集: {len(train_dataset)} | 测试集: {len(test_dataset)}")

    # 3. DataLoader
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4)

    # 4. 加载预训练模型
    print("正在加载预训练权重...")
    encoder = TrafficEncoder(feature_dim=128)
    
    checkpoint_path = r"C:\Users\MR\OneDrive\Desktop\gxy\DeepTraffic-CL\checkpoints\pretrain\encoder_epoch_50.pth"
    if os.path.exists(checkpoint_path):
        encoder.load_state_dict(torch.load(checkpoint_path))
        print("成功加载预训练权重！")
    else:
        print("未找到预训练权重，将使用随机初始化训练 (效果会变差)")

    # 5. 构建分类模型
    model = TrafficClassifier(encoder).to(Config.DEVICE)

    # 冻结 Encoder，只训练 Classifier (始终冻结，不进行解冻)
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    
    # 使用鲁棒损失函数 SCE (Alpha=0.1, Beta=1.0)
    criterion = SymmetricCrossEntropy(alpha=0.1, beta=1.0, num_classes=2)

    # 创建保存目录
    os.makedirs("checkpoints/finetune", exist_ok=True)

    # 6. 训练循环
    best_acc = 0.0
    
    for epoch in range(Config.EPOCHS_FINETUNE):
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
        print(f"Epoch [{epoch+1}/{Config.EPOCHS_FINETUNE}] Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%")

        # --- 验证/测试 ---
        model.eval()
        test_correct = 0
        test_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                p = batch['payload'].to(Config.DEVICE)
                s = batch['sequence'].to(Config.DEVICE)
                y = batch['label'].to(Config.DEVICE)
                
                outputs = model(p, s)
                _, predicted = torch.max(outputs.data, 1)
                
                test_total += y.size(0)
                test_correct += (predicted == y).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        test_acc = 100 * test_correct / test_total
        print(f"   >>> Test Acc: {test_acc:.2f}%")
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "checkpoints/finetune/best_model.pth")

    # 7. 最终评估报告
    print("\n" + "="*30)
    print(f"训练完成！最佳准确率: {best_acc:.2f}%")
    print("最终测试集详细报告:")
    print(classification_report(all_labels, all_preds, target_names=['Benign', 'Malware'], digits=4))
    print("混淆矩阵:")
    print(confusion_matrix(all_labels, all_preds))

if __name__ == "__main__":
    main()