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

ContrastiveDataset: 实时生成四个增强视图 用于预训练。

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



E. 可视化

visualize_results.py:

使用 t-SNE 将高维特征降维展示。验证良性与恶意流量在特征空间的可分性。

3. 运行顺序

python src/preprocess.py (数据准备)

python train_pretext.py (预训练)

python train_downstream.py (分类检测)


数据集说明及获取方式

⚠️ 提请注意：
由于原始网络流量包（PCAP）包含大量用户隐私且文件体积极其庞大（高达数十 GB），本 GitHub 仓库不包含任何原始数据集。
本项目代码内置了“防爆内存与截断机制（Early Stop）”，对于每个庞大的开源数据集，我们仅提取其中极具代表性的 1000~2000 个流样本用于模型论证与轻量级测试验证，以适配个人计算机的本地运行。

如需复现本项目，请前往以下官方地址下载对应的数据集，并将其放入 data/raw/ 目录中。

1. USTC-TFC2016 (主实验与训练集)

简介：由中国科学技术大学发布的经典加密流量数据集，包含 10 类正常应用流量与 10 类真实的恶意软件加密流量。

下载链接：GitHub - USTC-TFC2016

本项目使用量：作为预训练与基础实验的基石，抽取了其中部分核心 PCAP 文件，最终提取了约 21000+ 条均衡流数据用于骨干网络训练。

2. ADPT2020 (泛化数据集 1)

简介：包含针对物联网（IoT）设备及高级持续性威胁（APT）的网络攻击流量，网络拓扑与 USTC 数据集完全不同。

下载链接：由于该数据集属于内部或特定学术渠道公开，请参考相关 APT 攻击流量开源仓库（如 Stratosphere IPS）获取等效替换数据。

本项目使用量：仅用于跨域泛化测试。提取了约 2000 个流样本（包含正常背景与各类 APT 攻击）。

3. CIC-IDS2017 (泛化数据集 2)

简介：加拿大网络安全研究所（CIC）发布的殿堂级基准数据集。包含了极其真实的现代企业级背景流量，以及多种高阶网络攻击（如 DDoS, Brute Force 等）。

下载链接：CIC-IDS2017 Official Webpage

下载建议：无需下载全部几十GB数据！ 仅需下载 MachineLearningCSV.zip（需替换为包含IP的版本 GeneratedLabelledFlows.zip）和 Tuesday-WorkingHours.pcap。

本项目使用量：为防止本地内存溢出，代码中设定了动态截断机制。实际仅从 Tuesday.pcap 中提取了 1000 条良性样本与 1000 条恶意攻击样本用于跨洋网络环境泛化测试。

4. UNSW-NB15 (泛化数据集 3)

简介：澳大利亚新南威尔士大学发布的国际权威现代网络攻击数据集。包含海量的正常背景流量以及 9 种现代复杂攻击（如 Fuzzers, Analysis, Backdoors 等）。

下载链接：UNSW-NB15 Official Webpage

下载建议：为了极简复现，请仅下载全局通缉令文件 NUSW-NB15_GT.csv，以及任意一个流量包卷宗（如 1.pcap）。

本项目使用量：基于全局 Ground Truth 映射策略，从单一 pcap 文件中动态提取了 1000 条良性样本 与 745 条现代混合攻击样本 进行终极泛化测试。

📊 核心实验与结果

本项目的所有评估脚本均位于项目根目录，主要包含以下三大核心实验：

实验一：消融实验 (Ablation Study)

实验目的：验证第一阶段“多视图自监督对比预训练”对模型稳定性的绝对贡献。

实验设定：比较了从零开始随机初始化的基线模型（Baseline）与经过对比预训练的模型（Ours）在小样本微调下的表现。

结论：在极少标签数据下，Baseline 会发生严重过拟合，F1 分数发生断崖式震荡（~65%）；而 Ours 凭借无监督先验知识，稳定达到 99.9% 的超高分类精度。

实验二：终极抗混淆鲁棒性对比实验 (Comparative Study)

实验目的：模拟真实工业界标签稀缺，且黑客极力伪装的恶劣网络对抗环境。

实验设定：仅提供 1000 个带标签样本供分类器学习，并在测试集中加入 80% 概率的强混淆攻击（掩码、填充、时序抖动、随机丢包）。

对比算法：

传统机器学习：Random Forest

普通深度学习：Baseline CNN

专职抗混淆基线：Adv-CNN (对抗训练模型)

本文方法：Ours (DeepTraffic-CL)

结论：Baseline CNN 遇到混淆直接崩溃（准确率跌至 65%）；Adv-CNN 因标签不足无法收敛（F1 分数约 81%）；而 Ours 模型凭借无监督阶段学到的抗混淆表征，取得了 91%+ 的碾压级性能优势。

实验三：跨域领域自适应泛化实验 (OOD Generalization Study)

实验目的：验证模型在部署到完全未知的网络拓扑和环境时，克服“水土不服”的能力。

实验设定：采用小样本自适应策略（Few-Shot Domain Adaptation），冻结主干网络，仅使用目标域中 10% 的极小样本校准分类头，并在剩余 90% 的未知数据上进行盲测。

泛化结果：

泛化集 1 (ADPT2020)：精准识别跨域 APT 攻击，微调后测试集准确率飙升至 89.4%。

泛化集 2 (CIC-IDS2017)：在复杂的加拿大企业级混合背景下，以仅 200 个样本的适应代价，在 1800 个测试集上实现了 97.0% 的极高准确率。

泛化集 3 (UNSW-NB15)：面对澳洲网络环境下的 9 大类现代复杂攻击，以 174 个样本校准后，取得了 95.5% 的准确率与 95.4% 的宏平均 F1 分数。

🚀 快速启动

环境配置：

pip install torch torchvision scikit-learn pandas scapy tqdm numpy


(注意：建议使用 numpy<2.0 以保证与 PyTorch 1.x/2.x 兼容)

数据预处理（需先自行下载 PCAP）：

python preprocess_cicids.py    # 提取泛化集2特征
python preprocess_unswnb15.py  # 提取泛化集3特征


复现核心实验：

python run_comparison.py       # 运行终极抗混淆多算法对比实验
python test_domain_adaptation.py # 运行跨域自适应泛化实验


📝 致谢

感谢中科大（USTC）、加拿大网络安全研究所（CIC）以及澳洲新南威尔士大学（UNSW）为学术界无私提供的优质网络流量基准数据集。