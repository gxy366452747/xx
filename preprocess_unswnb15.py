import os
import pandas as pd
import torch
from scapy.all import PcapReader, IP, TCP, UDP
import numpy as np

# --- 极简配置 ---
GT_CSV_PATH = "./data/raw/UNSW-NB15/NUSW-NB15_GT.csv"
PCAP_PATH = "./data/raw/UNSW-NB15/1.pcap" 
OUTPUT_FILE = "./data/processed/unswnb15_traffic_data.pt"

MAX_PAYLOAD_LEN = 784
MAX_SEQ_LEN = 20
MAX_SAMPLES_PER_CLASS = 1000

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

def get_flow_key(ip1, port1, ip2, port2, proto):
    # 忽略方向，统一格式
    if ip1 > ip2:
        ip1, ip2 = ip2, ip1
        port1, port2 = port2, port1
    return f"{ip1}:{port1}-{ip2}:{port2}-{proto}"

def build_malware_set(csv_path):
    print(f"  [1/2] 解析 (Ground Truth): {os.path.basename(csv_path)}")
    malware_keys = set()
    try:
        df = pd.read_csv(csv_path, low_memory=False)
        # 统一转小写，去掉首尾空格
        df.columns = df.columns.astype(str).str.strip().str.lower()
        cols = list(df.columns)
        
        # 智能匹配：兼容全拼和缩写
        try:
            src_ip_col = [c for c in cols if ('src' in c or 'source' in c) and 'ip' in c][0]
            dst_ip_col = [c for c in cols if ('dst' in c or 'dest' in c) and 'ip' in c][0]
            src_port_col = [c for c in cols if 'sport' in c or (('src' in c or 'source' in c) and 'port' in c)][0]
            dst_port_col = [c for c in cols if 'dport' in c or (('dst' in c or 'dest' in c) and 'port' in c)][0]
            proto_col = [c for c in cols if 'proto' in c][0]
        except IndexError:
            #  如果匹配失败，直接把 CSV 真实的列名打印出来！
            print(f"  匹配列名失败！作者起的列名是: {cols}")
            return set()
            
        for _, row in df.iterrows():
            src_ip = str(row[src_ip_col]).strip()
            dst_ip = str(row[dst_ip_col]).strip()
            src_port = str(row[src_port_col]).strip()
            dst_port = str(row[dst_port_col]).strip()
            proto_str = str(row[proto_col]).strip().lower()
            
            # 转为数字协议
            proto = 6 if proto_str == 'tcp' else (17 if proto_str == 'udp' else 0)
            if proto == 0: continue
            
            key = get_flow_key(src_ip, src_port, dst_ip, dst_port, proto)
            malware_keys.add(key)
            
        print(f"        -> 成功载入黑名单！共记录 {len(malware_keys)} 条独特恶意流特征。")
        return malware_keys
    except Exception as e:
        print(f"   读取 GT CSV 异常: {e}")
        return set()

def main():
    print(f" 开始 UNSW-NB15 终极预处理")
    print("="*60)
    
    if not os.path.exists(GT_CSV_PATH) or not os.path.exists(PCAP_PATH):
        print(f" 找不到文件！请确保 {GT_CSV_PATH} 和 {PCAP_PATH} 存在。")
        return

    # 第一步：
    malware_keys = build_malware_set(GT_CSV_PATH)
    if len(malware_keys) == 0: return

    print(f"  [2/2] 提取 PCAP 特征: {os.path.basename(PCAP_PATH)}")
    
    flow_data = {}
    extracted_samples = []
    pkt_count = 0
    benign_collected = 0
    malware_collected = 0
    
    try:
        with PcapReader(PCAP_PATH) as pcap_reader:
            for pkt in pcap_reader:
                pkt_count += 1
                
                # 每 50 万包播报一次
                if pkt_count % 500000 == 0:
                    print(f"        -> [进度报告] 已快进 {pkt_count//10000} 万个包... 当前战况: 良性 {benign_collected}/1000 | 恶意 {malware_collected}/1000")
                
                if benign_collected >= MAX_SAMPLES_PER_CLASS and malware_collected >= MAX_SAMPLES_PER_CLASS:
                    print(f"        ->  触发保护机制：已收集足够的均衡样本，提前安全退出！")
                    break
                
                if not pkt.haslayer(IP): continue
                ip_layer = pkt[IP]
                proto = ip_layer.proto
                
                if proto == 6 and pkt.haslayer(TCP):
                    src_port, dst_port = pkt[TCP].sport, pkt[TCP].dport
                    payload = bytes(pkt[TCP].payload)
                elif proto == 17 and pkt.haslayer(UDP):
                    src_port, dst_port = pkt[UDP].sport, pkt[UDP].dport
                    payload = bytes(pkt[UDP].payload)
                else: continue
                
                key = get_flow_key(ip_layer.src, src_port, ip_layer.dst, dst_port, proto)
                
                # 核心逻辑：就是恶意 (1)，否则就是良性 (0)
                label = 1 if key in malware_keys else 0
                
                if label == 0 and benign_collected >= MAX_SAMPLES_PER_CLASS: continue
                if label == 1 and malware_collected >= MAX_SAMPLES_PER_CLASS: continue
                
                if key not in flow_data:
                    flow_data[key] = {'payload': b'', 'seq_lengths': [], 'seq_times': [], 'label': label, 'closed': False}
                
                flow = flow_data[key]
                if flow['closed']: continue
                
                if len(flow['payload']) < MAX_PAYLOAD_LEN:
                    flow['payload'] += payload
                
                if len(flow['seq_lengths']) < MAX_SEQ_LEN:
                    flow['seq_lengths'].append(len(pkt))
                    flow['seq_times'].append(float(pkt.time))
                
                if len(flow['payload']) >= MAX_PAYLOAD_LEN and len(flow['seq_lengths']) >= MAX_SEQ_LEN:
                    final_payload = np.frombuffer(flow['payload'][:MAX_PAYLOAD_LEN].ljust(MAX_PAYLOAD_LEN, b'\x00'), dtype=np.uint8)
                    times = np.array(flow['seq_times'])
                    iats = np.zeros(MAX_SEQ_LEN)
                    if len(times) > 1: iats[1:] = times[1:] - times[:-1]
                    
                    seq_features = np.column_stack((np.array(flow['seq_lengths']), iats))
                    final_payload = final_payload.astype(np.float32) / 255.0
                    seq_features[:, 0] = seq_features[:, 0] / 1500.0
                    seq_features[:, 1] = np.clip(seq_features[:, 1], 0, 1.0)
                    
                    extracted_samples.append({
                        'payload': torch.tensor(final_payload, dtype=torch.float32),
                        'sequence': torch.tensor(seq_features, dtype=torch.float32),
                        'label': torch.tensor(label, dtype=torch.long)
                    })
                    
                    if label == 0: benign_collected += 1
                    else: malware_collected += 1
                        
                    flow['closed'] = True
                        
    except Exception as e:
        print(f"  [X] PCAP读取中断: {e}")
        
    print("\n" + "="*60)
    print(f" 预处理安全完成！共提取 {len(extracted_samples)} 个样本。")
    if len(extracted_samples) > 0:
        torch.save(extracted_samples, OUTPUT_FILE)
        labels = [s['label'].item() for s in extracted_samples]
        print(f"最终分布: 良性 (0) = {labels.count(0)} | 恶意 (1) = {labels.count(1)}")
        print(f" 数据已轻量级保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()