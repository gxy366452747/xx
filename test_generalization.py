#跨数据集泛化实验 (OOD Generalization)
# 测试ADPT2020 数据集，验证模型泛化能力。
#直接使用USTC-TFC2016 的模型进行测试 
import torch
import os
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from src.config import Config
from src.dataset import FinetuneDataset
from src.models import TrafficEncoder
import torch.nn as nn

# --- 复用我们之前定义的分类模型结构 ---
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
        logits = self.classifier(features)
        return logits

def main():
    print(f"开始跨数据集泛化实验 (OOD Generalization Test) | 设备: {Config.DEVICE}")
    print(" 测试数据集: ADPT2020")
    print(" 使用模型: USTC-TFC2016 预训练并微调后的最佳模型")
    print("="*60)

    # 1. 加载全新的 ADPT 数据
    adpt_data_path = "./data/processed/adpt_traffic_data.pt"
    if not os.path.exists(adpt_data_path):
        print(" 找不到 ADPT 数据，请先运行 preprocess_adpt.py")
        return
    
    adpt_data = torch.load(adpt_data_path)
    print(f"成功加载 ADPT 测试数据: 共 {len(adpt_data)} 条流。")
    
    # 2. 包装成 Dataset 和 DataLoader (无需训练，所以 augment=False)
    test_dataset = FinetuneDataset(adpt_data, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # 3. 加载我们训练好的最强模型
    # 注意：这里加载的是 checkpoints/finetune 下的完整模型！
    encoder = TrafficEncoder(feature_dim=128)
    model = TrafficClassifier(encoder).to(Config.DEVICE)
    
    model_path = "checkpoints/finetune/best_model.pth"
    if not os.path.exists(model_path):
        print(f"找不到预训练模型 {model_path}，请确保已经跑过 train_downstream.py")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
    model.eval()
    print(" 成功加载目标检测模型。")
    print("="*60)

    # 4. 开始纯测试
    all_preds = []
    all_labels = []
    
    print("🚀 正在对 ADPT2020 数据进行检测预测...")
    with torch.no_grad():
        for batch in test_loader:
            p = batch['payload'].to(Config.DEVICE)
            s = batch['sequence'].to(Config.DEVICE)
            y = batch['label'].to(Config.DEVICE)
            
            outputs = model(p, s)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
    # 5. 出具评估报告
    print("\n" + "="*60)
    print(" 泛化能力最终评估报告 (Cross-Dataset Evaluation)")
    print("="*60)
    print(classification_report(all_labels, all_preds, target_names=['Benign(良性)', 'Malware(恶意)'], digits=4))
    
    print("\n 混淆矩阵 (Confusion Matrix):")
    cm = confusion_matrix(all_labels, all_preds)
    print(f"真正例(TN): {cm[0][0]:<6} | 假正例(FP/误报): {cm[0][1]:<6}")
    print(f"假反例(FN/漏报): {cm[1][0]:<6} | 真反例(TP): {cm[1][1]:<6}")

if __name__ == "__main__":
    main()