#预处理ADPT2020数据集

import os
import glob
import pandas as pd
import torch
from scapy.all import PcapReader, IP, TCP, UDP
import numpy as np

# --- 配置参数 ---
ADPT_DATA_DIR = "./data/raw/ADPT2020"  
OUTPUT_DIR = "./data/processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "adpt_traffic_data.pt")

MAX_PAYLOAD_LEN = 784
MAX_SEQ_LEN = 20
# 允许每个文件读取 20万个包
MAX_PKTS_PER_FILE = 200000

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_flow_key(ip1, port1, ip2, port2, proto):
    if ip1 > ip2:
        ip1, ip2 = ip2, ip1
        port1, port2 = port2, port1
    return f"{ip1}:{port1}-{ip2}:{port2}-{proto}"

#读取 CSV 文件。对于每一行，它抽出 IP 和端口，组合成一个字符串，
# 然后看最后那列的活动字段，如果是良性，则标记为0，如果是恶意，则标记为1
def build_label_dict(csv_path):
    print(f"  [1/2] 解析 CSV 标签: {os.path.basename(csv_path)}")
    try:
        df = pd.read_csv(csv_path, skipinitialspace=True) 
        df.columns = df.columns.str.strip()
        
        label_dict = {}
        malware_count = 0
        benign_count = 0
        
        for _, row in df.iterrows():
            try:
                src_ip = str(row['Src IP']).strip()
                dst_ip = str(row['Dst IP']).strip()
                src_port = int(row['Src Port'])
                dst_port = int(row['Dst Port'])
                proto = int(row['Protocol']) 
                
                # 获取标签字符串
                activity = str(row.get('Activity', 'benign')).lower().strip()
                stage = str(row.get('Stage', 'benign')).lower().strip()
                
               
                # 把 benign 和 normal 都算作良性
                benign_keywords = ['benign', 'normal', 'nan', '']
                
                if activity in benign_keywords and stage in benign_keywords:
                    is_malware = 0
                    benign_count += 1
                else:
                    is_malware = 1
                    malware_count += 1
                #防伪机制，函数强制把 IP 地址小的排在前面（if ip1 > ip2: 交换），这样无论是去是回都是同一个流
                key = get_flow_key(src_ip, src_port, dst_ip, dst_port, proto)
                label_dict[key] = is_malware
            except Exception as e:
                continue
                
        print(f"        -> 成功提取标签: 良性 {benign_count} | 恶意 {malware_count}")
        return label_dict
    except Exception as e:
        print(f"  [X] 读取 CSV 失败: {e}")
        return {}
#查找数据包和csv文件中匹配的流，提取特征
def process_single_pcap(pcap_path, label_dict):
    print(f"  [2/2] 提取 PCAP 特征: {os.path.basename(pcap_path)}")
    flow_data = {}
    extracted_samples = []
    
    pkt_count = 0
    match_count = 0
    
    try:
        with PcapReader(pcap_path) as pcap_reader:
            for pkt in pcap_reader:
                pkt_count += 1
                if pkt_count > MAX_PKTS_PER_FILE:
                    print(f"        -> 达到最大解析包数限制 ({MAX_PKTS_PER_FILE})，提前结束。")
                    break
                
                if not pkt.haslayer(IP): continue
                ip_layer = pkt[IP]
                proto = ip_layer.proto
                
                if proto == 6 and pkt.haslayer(TCP):
                    src_port = pkt[TCP].sport
                    dst_port = pkt[TCP].dport
                    payload = bytes(pkt[TCP].payload)
                elif proto == 17 and pkt.haslayer(UDP):
                    src_port = pkt[UDP].sport
                    dst_port = pkt[UDP].dport
                    payload = bytes(pkt[UDP].payload)
                else: continue
                
                key = get_flow_key(ip_layer.src, src_port, ip_layer.dst, dst_port, proto)
                #如果流在标签字典中，则提取特征，打上标签，载荷，序列，标签，closed
                if key in label_dict:
                    label = label_dict[key]
                    if key not in flow_data:
                        flow_data[key] = {'payload': b'', 'seq_lengths': [], 'seq_times': [], 'label': label, 'closed': False}
                    
                    flow = flow_data[key]
                    if flow['closed']: continue
                    
                    if len(flow['payload']) < MAX_PAYLOAD_LEN:
                        flow['payload'] += payload
                    
                    if len(flow['seq_lengths']) < MAX_SEQ_LEN:
                        flow['seq_lengths'].append(len(pkt))
                        flow['seq_times'].append(float(pkt.time))
                    
                    # 收集完毕，打包
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
                        flow['closed'] = True
                        match_count += 1
                        
    except Exception as e:
        pass
        
    print(f"        -> 扫描数据包 {pkt_count} 个，成功提取流样本: {match_count} 个")
    return extracted_samples

def main():
    print(f" 开始 ADPT2020 泛化数据集预处理")
    csv_dir = os.path.join(ADPT_DATA_DIR, "csv")
    pcap_dir = os.path.join(ADPT_DATA_DIR, "pcap-data")
    
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    all_extracted_data = []
    
    for csv_file in csv_files:
        base_name = os.path.basename(csv_file).replace("_Flow.csv", "")
        pcap_file = os.path.join(pcap_dir, base_name)
        
        if os.path.exists(pcap_file):
            print("\n" + "-"*50)
            print(f"发现配对文件: {base_name}")
            label_dict = build_label_dict(csv_file)
            if len(label_dict) == 0: continue
            
            samples = process_single_pcap(pcap_file, label_dict)
            all_extracted_data.extend(samples)

    print("="*50)
    print(f" 预处理完成！共提取 {len(all_extracted_data)} 个样本。")
    if len(all_extracted_data) > 0:
        torch.save(all_extracted_data, OUTPUT_FILE)
        labels = [s['label'].item() for s in all_extracted_data]
        print(f" 分布: 良性 (0) = {labels.count(0)} | 恶意 (1) = {labels.count(1)}")

if __name__ == "__main__":
    main()