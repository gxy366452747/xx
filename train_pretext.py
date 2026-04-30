import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import os
from tqdm import tqdm

from src.config import Config
from src.dataset import ContrastiveDataset
from src.models import TrafficEncoder, SimCLR
# 注意：这里引用的是我们刚才更新过的支持多视图的 Loss
from src.losses import MultiViewNTXentLoss

def main():
    print(f" [Phase 1] 开始多视图(4-View)对比学习预训练 | 设备: {Config.DEVICE}")
    
    # -----------------------------------------------------------
    # 1. 加载数据
    # -----------------------------------------------------------
    data_path = os.path.join(Config.PROCESSED_DATA_DIR, "all_traffic_data.pt")
    if not os.path.exists(data_path):
        print(f"错误：找不到数据文件 {data_path}，请先运行 preprocess.py")
        return

    # 加载预处理好的 Tensor 数据
    all_data = torch.load(data_path)
    print(f"数据加载完成，共 {len(all_data)} 条样本。")
    
    # -----------------------------------------------------------
    # 2. 准备 Dataset
    # -----------------------------------------------------------
    # 实例化数据集，它现在会针对每个样本返回 4 个不同的视图
    dataset = ContrastiveDataset(all_data)
    
    # 显存警告：因为变成了 4 个视图，相当于 Batch Size 翻倍了。
    # 如果显存不够，请去 config.py 调小 BATCH_SIZE (例如 64 -> 32)
    dataloader = DataLoader(
        dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    # -----------------------------------------------------------
    # 3. 模型初始化
    # -----------------------------------------------------------
    # 骨干网络 (CNN+LSTM)
    encoder = TrafficEncoder(feature_dim=128)
    # 加上投影头的 SimCLR 框架
    model = SimCLR(encoder).to(Config.DEVICE)
    
    # -----------------------------------------------------------
    # 4. 优化器与损失
    # -----------------------------------------------------------
    optimizer = optim.Adam(model.parameters(), lr=Config.LR_PRETRAIN)
    
    # 使用刚才更新的、支持多视图的 Loss 函数
    criterion = MultiViewNTXentLoss(temperature=Config.TEMP)

    # 创建保存文件夹
    os.makedirs("checkpoints/pretrain", exist_ok=True)

    # -----------------------------------------------------------
    # 5. 训练循环
    # -----------------------------------------------------------
    print(f"开始训练，共 {Config.EPOCHS_PRETRAIN} 个 Epoch...")
    
    for epoch in range(Config.EPOCHS_PRETRAIN):
        model.train()
        total_loss = 0
        
        # 进度条描述
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{Config.EPOCHS_PRETRAIN}")
        
        # --- [修改] 这里现在解包出 4 个视图 ---
        for view1, view2, view3, view4 in pbar:
            
            # 1. 数据搬运到 GPU
            # 为了代码整洁，我们把 4 个视图的数据分别取出来
            # 这里的效率优化：我们可以把它们拼起来一次性过模型
            
            # 收集所有的 payload: [Batch, 784] -> [4*Batch, 784]
            payloads = torch.cat([
                view1['payload'], view2['payload'], 
                view3['payload'], view4['payload']
            ], dim=0).to(Config.DEVICE)
            
            # 收集所有的 sequence: [Batch, 20, 2] -> [4*Batch, 20, 2]
            sequences = torch.cat([
                view1['sequence'], view2['sequence'], 
                view3['sequence'], view4['sequence']
            ], dim=0).to(Config.DEVICE)
            
            optimizer.zero_grad()
            
            # 2. 前向传播 (一次性处理所有视图)
            # z_all 的形状是 [4*Batch, 64]
            _, z_all = model(payloads, sequences)
            
            # 3. 拆分特征
            # 为了喂给 Loss 函数，我们需要把大张量拆回 4 个 [Batch, 64] 的列表
            # torch.chunk 将张量切成 4 块
            z_list = torch.chunk(z_all, 4, dim=0)
            
            # 4. 计算多视图对比损失
            # 这里的 z_list 就是 [z1, z2, z3, z4]
            loss = criterion(z_list)
            
            # 5. 反向传播与优化
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        # 打印本轮平均 Loss
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}] 完成. 平均 Loss: {avg_loss:.4f}")
        
        # 6. 定期保存模型
        if (epoch + 1) % 5 == 0:
            save_path = f"checkpoints/pretrain/encoder_epoch_{epoch+1}.pth"
            # 依然只保存 encoder 的参数，因为 projector 只是为了训练用的
            torch.save(model.encoder.state_dict(), save_path)
            print(f"权重已保存: {save_path}")

    print("多视图预训练结束！模型已保存。")

if __name__ == "__main__":
    main()