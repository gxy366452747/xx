import os
import pandas as pd
from scapy.all import PcapReader, IP, TCP, UDP
import torch
from tqdm import tqdm

# --- 配置路径 ---
# 直接指向你刚才截图里的第 4 号文件夹
BINETFLOW_PATH = "./data/raw/CTU-13/CTU-13-Dataset/4/capture20110815.binetflow"
PCAP_PATH = "./data/raw/CTU-13/CTU-13-Dataset/4/botnet-capture-20110815-rbot-dos.pcap"
OUTPUT_PATH = "./data/processed/ctu13_traffic_data.pt"

# --- 参数配置 ---
MAX_PACKETS_TO_SCAN = 5000000  # CTU13的包比较密集，扫500万个就够了
MAX_SAMPLES_PER_CLASS = 1000   # 提取上限

def get_5tuple(pkt):
    """提取五元组字符串，忽略方向"""
    if IP in pkt:
        src_ip, dst_ip = pkt[IP].src, pkt[IP].dst
        proto = pkt[IP].proto
        src_port, dst_port = 0, 0
        if TCP in pkt:
            src_port, dst_port = pkt[TCP].sport, pkt[TCP].dport
        elif UDP in pkt:
            src_port, dst_port = pkt[UDP].sport, pkt[UDP].dport
            
        # 统一格式：小IP:端口_大IP:端口_协议
        ep1 = f"{src_ip}:{src_port}"
        ep2 = f"{dst_ip}:{dst_port}"
        if ep1 > ep2:
            ep1, ep2 = ep2, ep1
        return f"{ep1}_{ep2}_{proto}"
    return None

def main():
    print("🚀 开始 CTU-13 (僵尸网络) 本地预处理")
    print("=" * 60)
    
    if not os.path.exists(BINETFLOW_PATH) or not os.path.exists(PCAP_PATH):
        print("❌ 找不到文件！请检查路径是否正确。")
        return

    # 1. 解析 .binetflow (相当于 CSV)
    print(f"  [1/2] 解析 binetflow 标签: {os.path.basename(BINETFLOW_PATH)}")
    # binetflow 是逗号分隔的
    df = pd.read_csv(BINETFLOW_PATH)
    
    flow_labels = {}
    benign_count = 0
    malware_count = 0
    
    for _, row in df.iterrows():
        try:
            # CTU-13 的标签通常包含 'Botnet' 或 'Background'/'LEGITIMATE'
            label_str = str(row['Label']).lower()
            if 'botnet' in label_str:
                label = 1
                malware_count += 1
            # 在 CTU13 公开版中，纯正的 LEGITIMATE 很少，我们把 Background 也当做背景正常流量来凑数
            elif 'background' in label_str or 'legitimate' in label_str:
                label = 0
                benign_count += 1
            else:
                continue

            src_ip, dst_ip = row['SrcAddr'], row['DstAddr']
            src_port, dst_port = int(row.get('Sport', 0) if pd.notna(row.get('Sport')) else 0), int(row.get('Dport', 0) if pd.notna(row.get('Dport')) else 0)
            
            # 协议号转换 (TCP=6, UDP=17, ICMP=1 等)
            proto_str = str(row['Proto']).lower()
            if 'tcp' in proto_str: proto = 6
            elif 'udp' in proto_str: proto = 17
            elif 'icmp' in proto_str: proto = 1
            else: proto = 0 # 兜底

            ep1 = f"{src_ip}:{src_port}"
            ep2 = f"{dst_ip}:{dst_port}"
            if ep1 > ep2:
                ep1, ep2 = ep2, ep1
            flow_key = f"{ep1}_{ep2}_{proto}"
            flow_labels[flow_key] = label
        except Exception:
            continue
            
    print(f"        -> 成功提取通缉令: 背景正常 {benign_count} | 僵尸网络恶意 {malware_count}")

    # 2. 提取 PCAP
    print(f"  [2/2] 提取 PCAP 特征: {os.path.basename(PCAP_PATH)}")
    
    dataset = []
    benign_collected = 0
    malware_collected = 0
    pkt_count = 0
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    try:
        with PcapReader(PCAP_PATH) as pcap_reader:
            for pkt in pcap_reader:
                pkt_count += 1
                
                # 每扫 50万个包报一次平安
                if pkt_count % 500000 == 0:
                    print(f"        -> [进度报告] 已扫描 {pkt_count} 个包... 当前战况: 良性 {benign_collected}/{MAX_SAMPLES_PER_CLASS} | 恶意 {malware_collected}/{MAX_SAMPLES_PER_CLASS}")
                
                if pkt_count > MAX_PACKETS_TO_SCAN:
                    print(f"        -> 🛑 达到安全解析包数上限 ({MAX_PACKETS_TO_SCAN})，提前退出。")
                    break
                    
                flow_key = get_5tuple(pkt)
                if not flow_key or flow_key not in flow_labels:
                    continue
                    
                label = flow_labels[flow_key]
                
                # 满员拦截
                if label == 0 and benign_collected >= MAX_SAMPLES_PER_CLASS:
                    continue
                if label == 1 and malware_collected >= MAX_SAMPLES_PER_CLASS:
                    continue
                    
                payload = bytes(pkt[TCP].payload) if TCP in pkt else (bytes(pkt[UDP].payload) if UDP in pkt else b"")
                if len(payload) < 10:
                    continue
                    
                payload_padded = list(payload[:128].ljust(128, b'\x00'))
                
                dataset.append({
                    "payload": torch.tensor(payload_padded, dtype=torch.float32) / 255.0,
                    "sequence": torch.zeros(10, 3), # 简化的流序列
                    "label": torch.tensor(label, dtype=torch.long)
                })
                
                if label == 0: benign_collected += 1
                if label == 1: malware_collected += 1
                
                # 如果收集满了，直接退出
                if benign_collected >= MAX_SAMPLES_PER_CLASS and malware_collected >= MAX_SAMPLES_PER_CLASS:
                    print(f"\n        -> 🛑 触发保护机制：已收集足够的均衡样本 ({benign_collected}良性/{malware_collected}恶意)，提前安全退出！")
                    break

    except Exception as e:
        print(f"⚠️ 解析 PCAP 出现问题: {e}")

    print("=" * 60)
    print(f"🎉 预处理完成！")
    print(f"📊 最终分布: 良性 (0) = {benign_collected} | 恶意 (1) = {malware_collected}")
    
    if len(dataset) > 0:
        torch.save(dataset, OUTPUT_PATH)
        print(f"💾 数据已保存至: {OUTPUT_PATH}")
        print("💡 下一步：修改 test_domain_adaptation.py 里的路径为 ctu13_traffic_data.pt 进行泛化测试！")
    else:
        print("⚠️ 未提取到有效样本，请检查 PCAP 流量。")

if __name__ == "__main__":
    main()