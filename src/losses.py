import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiViewNTXentLoss(nn.Module):
    """
    多视图 (Multi-View) 的 InfoNCE 损失函数。
    
    目标：
    最大化所有正样本对（同一原始样本生成的任意两个视图）的一致性。
    比如有 4 个视图 (A, B, C, D)，那么 (A,B), (A,C), (A,D), (B,C)... 都是正样本对，都要拉近。
    """
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, views):
        """
        输入 views: 一个包含 K 个 Tensor 的列表。
        例如: [z1, z2, z3, z4]，每个 z 的形状都是 [Batch_Size, Dim]
        """
        # K 是视图数量 (这里是 4)
        K = len(views)
        batch_size = views[0].shape[0]
        device = views[0].device
        
        # 1. 拼接所有视图
        # 形状变化: [Batch, Dim] x 4 -> [4*Batch, Dim]
        # 顺序: [Batch1, Batch2, Batch3, Batch4]
        features = torch.cat(views, dim=0)
        
        # 2. 计算相似度矩阵
        # 先做归一化，这样矩阵乘法就等于计算余弦相似度
        features = F.normalize(features, dim=1)
        # sim_matrix[i, j] 表示第 i 个特征和第 j 个特征的相似度
        # 形状: [4*Batch, 4*Batch]
        similarity_matrix = torch.matmul(features, features.T)
        
        # 3. 构建标签 (Mask)
        # 我们需要知道谁和谁是正样本。
        # 比如 Batch=2, K=2。索引 0 的正样本是 2。索引 1 的正样本是 3。
        # labels 矩阵会标记出所有的正样本对位置。
        labels = torch.cat([torch.arange(batch_size) for _ in range(K)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(device)
        
        # 4. 移除对角线
        # mask[i, i] = 1 (自己跟自己肯定像，但不能算进 Loss)
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
        # 把对角线上的正样本标记抹掉
        labels = labels * ~mask
        
        # 5. 计算 Loss (广义 InfoNCE)
        # 公式: Loss = - log ( sum(exp(pos)) / sum(exp(all)) )
        
        # 缩放温度
        logits = similarity_matrix / self.temperature
        
        # 数值稳定性处理 (减去最大值防止 exp 溢出)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        
        # 分母: sum(exp(all)) - exp(self)
        # 这里利用 mask 把自己排除掉
        exp_logits = torch.exp(logits) * ~mask 
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # 分子: 只取正样本对的 log_prob
        # labels 矩阵里为 1 的位置就是正样本
        # sum(1) 是对每一行求和，计算有多少个正样本
        
        mean_log_prob_pos = (labels * log_prob).sum(1) / labels.sum(1)
        
        loss = -mean_log_prob_pos
        loss = loss.mean()
        
        return loss

class SymmetricCrossEntropy(nn.Module):
    """
    第二阶段：鲁棒损失函数
    用于分类微调，对抗噪声标签。
    防止模型过拟合那些错误的标签
    """
    def __init__(self, alpha=0.1, beta=1.0, num_classes=2):
        super().__init__()
        # alpha: 控制 CE (标准学习) 的权重。通常小一点 (0.1)，防止模型学太死。
        self.alpha = alpha
        # beta:  控制 RCE (抗噪能力) 的权重。通常大一点 (1.0)，强调鲁棒性。
        self.beta = beta
        # 分类只有 2 类 (良性/恶意)
        self.num_classes = num_classes

    def forward(self, pred, labels):
        # 1. CE Loss (标准交叉熵)
        ce = F.cross_entropy(pred, labels)

        # 2. RCE Loss (反向交叉熵 - 抗噪核心)
        #计算预测概率
        #softmax: 将预测值转换为概率分布
        #把模型输出的分数 (比如 [2.0, -1.0]) 变成概率 (比如 [0.95, 0.05])。
        pred_prob = F.softmax(pred, dim=1)
        #数据截断
        # 因为后面要算 log(x)。如果概率是 0，log(0) = 负无穷大，程序直接报错 NaN (Not a Number)。
        # 所以我们强制规定：最小不能小于 1e-7，最大不能超过 1.0。
        pred_prob = torch.clamp(pred_prob, min=1e-7, max=1.0)
        # 标签独热编码 (One-Hot Encoding)
        #  把数字标签变成向量。
        # 如果 label = 0 (良性) -> 变成 [1, 0]
        # 如果 label = 1 (恶意) -> 变成 [0, 1]
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        #  计算 RCE 核心公式
        # 公式：Sum( 预测概率 * log(标签概率) )
        rce = -1 * torch.sum(pred_prob * torch.log(label_one_hot), dim=1)
        # 第三部分：加权求和
        # 最后把 CE 标准交叉熵和 RCE 反向交叉熵 加权求和，得到总损失。
        return self.alpha * ce + self.beta * rce.mean()