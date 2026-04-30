import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import Config
#神经网络结构

class PayloadEncoder(nn.Module):
    """处理 原始载荷 (784字节) 的 1D CNN"""
    def __init__(self, input_len=Config.MAX_PAYLOAD_LEN, embed_dim=128):
        super().__init__()
        # Input: [Batch, 1, 784]
        # 卷积层1
        # 输入通道数为1，输出通道数为16，卷积核大小为5，步长为2，填充为2
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2) # -> 392
        self.bn1 = nn.BatchNorm1d(16)
        # 卷积层2
        # 输入通道数为16，输出通道数为32，卷积核大小为5，步长为2，填充为2
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2) # -> 196
        # 自动整理工具
        self.bn2 = nn.BatchNorm1d(32)
        
        # Global Average Pooling
        #全局平均池化最后每个通道只保留 1 个平均值。
        
        self.pool = nn.AdaptiveAvgPool1d(1) 
        
        # Linear (全连接层)
        # 将32维特征映射到128维统一空间
        self.fc = nn.Linear(32, embed_dim)
        
    def forward(self, x):
        # x: [Batch, Length] -> [Batch, 1, Length]
        # 在第 2 维度（垂直方向）上添加一个维度，变成 [Batch, 1, Length]
        x = x.unsqueeze(1)
        #第一轮加工，长度减半，通道数加宽，激活函数去掉负数
        x = F.relu(self.bn1(self.conv1(x)))
        #第二轮加工，长度再减半（步长是2），通道数再翻倍，激活函数再去掉负数
        x = F.relu(self.bn2(self.conv2(x)))
        #全局平均池化,长度归一，
        x = self.pool(x)
        # 最后变成[64,128]
        x = x.view(x.size(0), -1) # Flatten
        return self.fc(x)

class SequenceEncoder(nn.Module):
    """处理 包长序列(包长+间隔) 的 LSTM（长短时记忆网络）"""
    def __init__(self, input_dim=2, hidden_dim=64, embed_dim=128):
        super().__init__()
        # input_dim=2: 因为每个包只有 [大小, 时间间隔] 这两个信息。
        # num_layers=2: 双层 LSTM，理解能力更强。
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2)
        # 将LSTM的输出映射到128维统一空间
        self.fc = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        # x: [Batch, Seq_Len, 2]
        #自动循环二十次，每次处理一个包，处理完一个包后，把结果和下一个包的结果一起处理，直到处理完所有包。
        out, (h_n, c_n) = self.lstm(x)
        # 取最后一个时间步的输出，作为整个序列的特征
        return self.fc(out[:, -1, :])

class TrafficEncoder(nn.Module):
    """主编码器：融合 Payload 和 Sequence 特征"""
    def __init__(self, feature_dim=128):
        super().__init__()
        self.payload_enc = PayloadEncoder(embed_dim=feature_dim)
        self.seq_enc = SequenceEncoder(embed_dim=feature_dim)
        
        # 融合层
        # 输入是 128+128=256，输出变回 128。
        # 作用是把“载荷信息”和“时序信息”融合
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, payload, sequence):
        #调用载荷编码器和时序编码器，得到载荷特征和时序特征
        f1 = self.payload_enc(payload)#载荷
        f2 = self.seq_enc(sequence)#时序
        concat = torch.cat((f1, f2), dim=1)#物理合并，128+128=256
        embedding = self.fusion(concat)#全连接层，256->128
        return embedding

class SimCLR(nn.Module):
    """第一阶段：对比学习"""
    def __init__(self, encoder, projection_dim=64):
        super().__init__()
        #调用主编码器，得到128维特征向量
        self.encoder = encoder
        #加了一个专门用于训练的投影层，128->128->64
        self.projector = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, projection_dim)
        )

    def forward(self, payload, sequence):
        h = self.encoder(payload, sequence)
        # 学术界发现：直接用 h 算 Loss 效果不好，因为 h 保留了太多细节；
        # 把 h 经过投影头变成 z，再在 z 上算 Loss，能迫使 h 学到更好的特征。
        z = self.projector(h)
        # 训练时：我们要用到 z 来算 Loss。
        # 训练后：我们扔掉 z，只保留 h
        return h, z