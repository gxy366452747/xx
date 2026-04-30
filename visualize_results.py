import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import random
import matplotlib

# 设置无图形界面后端 (防止服务器报错)
matplotlib.use('Agg')

from src.config import Config
from src.models import TrafficEncoder

def visualize_features():
    print(f"🎨 [Visualization] 开始生成 t-SNE 特征分布图 | 设备: {Config.DEVICE}")
    
    # 1. 加载数据
    data_path = os.path.join(Config.PROCESSED_DATA_DIR, "all_traffic_data.pt")
    if not os.path.exists(data_path):
        print("❌ 找不到数据文件！")
        return
        
    all_data = torch.load(data_path)
    # 只取有标签数据 (0: Benign, 1: Malware)
    labeled_data = [d for d in all_data if d['label'].item() != -1]
    
    print(f"数据加载完成，共有标签样本 {len(labeled_data)} 个。")

    # 2. 随机抽样
    # t-SNE 比较慢，我们抽取 2000 个样本来画图足够了
    SAMPLE_SIZE = 2000
    if len(labeled_data) > SAMPLE_SIZE:
        sampled_data = random.sample(labeled_data, SAMPLE_SIZE)
    else:
        sampled_data = labeled_data
        
    print(f"随机抽样 {len(sampled_data)} 个点进行绘图...")

    # 3. 准备模型
    encoder = TrafficEncoder(feature_dim=128).to(Config.DEVICE)
    
    # 加载第二阶段微调后的最佳模型
    finetuned_path = "checkpoints/finetune/best_model.pth"
    
    if os.path.exists(finetuned_path):
        print(f"正在加载微调权重: {finetuned_path}")
        state_dict = torch.load(finetuned_path)
        
        # 提取 Encoder 部分的权重
        # 因为微调保存的是 Classifier (包含 encoder 和 classifier 头)
        # 所以键名是 "encoder.payload_enc..." 我们要去掉前缀 "encoder."
        encoder_dict = {}
        for k, v in state_dict.items():
            if k.startswith("encoder."):
                encoder_dict[k.replace("encoder.", "")] = v
        
        encoder.load_state_dict(encoder_dict)
    else:
        print("⚠️ 未找到微调权重！将尝试加载预训练权重...")
        pretrain_path = "checkpoints/pretrain/encoder_epoch_50.pth"
        if os.path.exists(pretrain_path):
            encoder.load_state_dict(torch.load(pretrain_path))
            print("✅ 已加载预训练权重 (Pre-trained)")
        else:
            print("❌ 没有任何权重文件，使用的是随机初始化模型（图会很乱）")

    encoder.eval()

    # 4. 提取特征
    features = []
    labels = []
    
    print("正在提取高维特征...")
    with torch.no_grad():
        for i, sample in enumerate(sampled_data):
            # 增加维度 [784] -> [1, 784]
            p = sample['payload'].unsqueeze(0).to(Config.DEVICE)
            s = sample['sequence'].unsqueeze(0).to(Config.DEVICE)
            y = sample['label'].item()
            
            # 通过编码器得到 128维 向量
            emb = encoder(p, s) 
            
            features.append(emb.cpu().numpy()[0])
            labels.append(y)

    features = np.array(features)
    labels = np.array(labels)

    # 5. t-SNE 降维 (128维 -> 2维)
    print("正在运行 t-SNE 降维 (这可能需要 1-2 分钟)...")
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(features)

    # 6. 绘图
    print("正在保存图片...")
    plt.figure(figsize=(12, 10))
    
    # 绘制良性点 (蓝色)
    benign_mask = (labels == 0)
    plt.scatter(X_embedded[benign_mask, 0], X_embedded[benign_mask, 1], 
                c='dodgerblue', label='Benign (Normal)', alpha=0.6, s=15, edgecolors='none')
    
    # 绘制恶意点 (红色)
    malware_mask = (labels == 1)
    plt.scatter(X_embedded[malware_mask, 0], X_embedded[malware_mask, 1], 
                c='crimson', label='Malware (Encrypted)', alpha=0.6, s=15, edgecolors='none')

    plt.title("t-SNE Visualization: Traffic Features Distribution", fontsize=16)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, alpha=0.2, linestyle='--')
    
    # 保存
    save_file = "tsne_result.png"
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    
    print(f"\n🎉 成功！图片已保存为: {os.path.abspath(save_file)}")
    print("请在 Cursor 左侧文件列表找到 'tsne_result.png' 并打开查看。")

if __name__ == "__main__":
    visualize_features()
