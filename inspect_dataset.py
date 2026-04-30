import os
import torch
import glob
from collections import Counter

def inspect_adpt_data(data_path):
    """
    侦察 ADPT2020 数据集的结构
    """
    print("="*50)
    print(f"🔍 开始探查数据集: {data_path}")
    print("="*50)

    if not os.path.exists(data_path):
        print(f"❌ 找不到路径: {data_path}")
        print("请检查路径是否正确！")
        return

    # 1. 检查是不是 pcap 文件目录
    pcap_files = glob.glob(os.path.join(data_path, "**/*.pcap"), recursive=True)
    pcapng_files = glob.glob(os.path.join(data_path, "**/*.pcapng"), recursive=True)
    all_pcaps = pcap_files + pcapng_files

    if len(all_pcaps) > 0:
        print(f"✅ 发现 {len(all_pcaps)} 个 PCAP 原始流量文件。")
        print("💡 结论：如果是 PCAP 文件，我们可以直接复用现有的 `src/preprocess.py` 进行处理！只需改一下读取路径即可。")
        print(f"   示例文件: {all_pcaps[0]}")
        return

    # 2. 检查是不是已经处理好的 .pt 张量文件
    pt_files = glob.glob(os.path.join(data_path, "*.pt"))
    if len(pt_files) > 0:
        print(f"✅ 发现 {len(pt_files)} 个 .pt PyTorch数据文件。")
        for pt_file in pt_files[:2]:  # 只看前两个
            try:
                data = torch.load(pt_file)
                print(f"\n📄 文件: {pt_file}")
                print(f"   数据类型: {type(data)}")
                if isinstance(data, list) and len(data) > 0:
                    print(f"   包含样本数: {len(data)}")
                    sample = data[0]
                    print(f"   单个样本的 Keys: {sample.keys() if isinstance(sample, dict) else 'Not a dict'}")
                    if isinstance(sample, dict):
                        for k, v in sample.items():
                            if torch.is_tensor(v):
                                print(f"     - {k}: Tensor shape {v.shape}, dtype {v.dtype}")
                            else:
                                print(f"     - {k}: {type(v)} = {v}")
                    
                    # 统计标签分布
                    if isinstance(sample, dict) and 'label' in sample:
                        labels = [d['label'].item() if torch.is_tensor(d['label']) else d['label'] for d in data]
                        print(f"   🏷️ 标签分布: {dict(Counter(labels))}")
            except Exception as e:
                print(f"   读取出错: {e}")
        return

    # 3. 检查是不是 CSV 特征文件
    csv_files = glob.glob(os.path.join(data_path, "*.csv"))
    if len(csv_files) > 0:
        print(f"✅ 发现 {len(csv_files)} 个 CSV 特征文件。")
        print("💡 结论：这说明原作者已经把特征提好了（可能是统计特征），我们可能需要写一个新的 Dataset 类来加载。")
        print(f"   示例文件: {csv_files[0]}")
        return

    print("⚠️ 未发现常见的流量数据格式 (pcap, pt, csv)。请检查该目录下到底是什么文件。")
    files = os.listdir(data_path)
    print(f"该目录下的文件/文件夹有: {files[:10]} ...")

if __name__ == "__main__":
    # ⚠️ 请把这里换成你服务器上存放 ADPT2020 数据集的真实路径！
    # 比如: adpt_path = "/home/gxy/datasets/ADPT2020"
    adpt_path = "/home/gxy/DeepTraffic-CL/data/raw/ADPT2020" 
    inspect_adpt_data(adpt_path)