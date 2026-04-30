基于对比学习的恶意加密流量鲁棒检测系统

1. 项目简介

本项目旨在利用对比学习（SimCLR框架）解决加密流量检测中的类不平衡和抗混淆问题。通过“预训练+微调”的两阶段模式，实现对恶意流量的高精度检测。

2. 代码结构与阅读指南

A. 基础配置

src/config.py: 全局配置文件。

定义了数据路径、特征提取长度 (Payload=784, Seq=20)、训练超参数 (BatchSize, LR)。

B. 数据工程 

src/preprocess.py (预处理):

功能: 将原始 PCAP 文件流式解析为 PyTorch Tensor。

核心: 五元组流聚合、双模态特征提取 (载荷+时序)。

输出: data/processed/all_traffic_data.pt

src/augmentations.py (混淆增强):

功能: 定义对抗性混淆策略，模拟恶意流量的变异。

策略:

Packet Padding (包长填充)

Timing Jitter (时间抖动)

Packet Dropping (随机丢包)

Payload Masking (载荷掩码)

src/dataset.py (数据加载):

功能: 将预处理数据与混淆策略结合。

ContrastiveDataset: 实时生成两个增强视图 (View1, View2) 用于预训练。

FinetuneDataset: 生成带标签的单视图用于微调。

C. 模型架构

src/models.py:

TrafficEncoder: 双模态编码器 (1D-CNN 处理载荷 + LSTM 处理序列)。

SimCLR: 封装编码器和投影头 (Projection Head)。

src/losses.py:

NTXentLoss: 对比损失函数 (用于第一阶段)。

SymmetricCrossEntropy: 鲁棒交叉熵损失 (用于第二阶段)。

D. 训练流程

train_pretext.py (第一阶段):

无监督预训练。让模型学习如何区分不同的流，提取抗混淆特征。

train_downstream.py (第二阶段):

下游任务微调。冻结编码器，训练分类头，输出准确率。

train_few_shot.py (验证实验):

模拟极少恶意样本 (1%) 的情况，验证模型的少样本学习能力（复现 Listen to Minority 核心思想）。

E. 可视化

visualize_results.py:

使用 t-SNE 将高维特征降维展示。验证良性与恶意流量在特征空间的可分性。

3. 运行顺序

python src/preprocess.py (数据准备)

python train_pretext.py (预训练)

python train_downstream.py (分类检测)