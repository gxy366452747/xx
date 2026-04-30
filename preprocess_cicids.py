import os
import glob
import pandas as pd
import torch
from scapy.all import PcapReader, IP, TCP, UDP
import numpy as np

# --- 配置参数 ---
CIC_DATA_DIR = "./data/raw/CIC-IDS2017"  
CSV_DIR = os.path.join(CIC_DATA_DIR, "MachineLearningCVE")
PCAP_DIR = os.path.join(CIC_DATA_DIR, "PCAPs") 

OUTPUT_DIR = "./data/processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "cicids2017_traffic_data.pt")

MAX_PAYLOAD_LEN = 784
MAX_SEQ_LEN = 20

# 【本地电脑保护机制】
MAX_SAMPLES_PER_CLASS = 1000 
# 直到抓到 1000 个恶意包为止！
MAX_PKTS_PER_FILE = float('inf') 

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_flow_key(ip1, port1, ip2, port2, proto):
    if ip1 > ip2:
        ip1, ip2 = ip2, ip1
        port1, port2 = port2, port1
    return f"{ip1}:{port1}-{ip2}:{port2}-{proto}"

def build_label_dict(csv_path):
    print(f"  [1/2] 解析 CSV 标签: {os.path.basename(csv_path)}")
    try:
        df = pd.read_csv(csv_path, encoding='cp1252', skipinitialspace=True) 
        df.columns = df.columns.str.strip() 
        
        label_dict = {}
        benign_count, malware_count = 0, 0
        
        required_cols = ['Source IP', 'Destination IP', 'Source Port', 'Destination Port', 'Protocol', 'Label']
        for col in required_cols:
            if col not in df.columns:
                alt_col = col.replace("Source", "Src").replace("Destination", "Dst")
                if alt_col in df.columns:
                    df.rename(columns={alt_col: col}, inplace=True)
                else:
                    print(f"\n  [致命错误]：CSV中缺少列 '{col}'！")
                    return {}

        for _, row in df.iterrows():
            try:
                src_ip = str(row['Source IP']).strip()
                dst_ip = str(row['Destination IP']).strip()
                src_port = int(row['Source Port'])
                dst_port = int(row['Destination Port'])
                proto = int(row['Protocol']) 
                label_str = str(row['Label']).upper().strip()
                
                is_malware = 0 if 'BENIGN' in label_str else 1
                key = get_flow_key(src_ip, src_port, dst_ip, dst_port, proto)
                
                if key in label_dict and label_dict[key] == 1: continue
                label_dict[key] = is_malware
                
                if is_malware: malware_count += 1
                else: benign_count += 1
            except: continue
                
        print(f"        -> 成功提取通缉令: 良性 {benign_count} | 恶意 {malware_count}")
        return label_dict
    except Exception as e: 
        print(f"   读取CSV异常: {e}")
        return {}

def process_single_pcap(pcap_path, label_dict):
    print(f"  [2/2] 提取 PCAP 特征: {os.path.basename(pcap_path)}")
    flow_data = {}
    extracted_samples = []
    
    pkt_count = 0
    benign_collected = 0
    malware_collected = 0
    
    try:
        with PcapReader(pcap_path) as pcap_reader:
            for pkt in pcap_reader:
                pkt_count += 1
                
                # --- 进度播报 ---
                if pkt_count % 1000000 == 0:
                    print(f"        -> [进度报告] 已快进 {pkt_count//1000000}00 万个包... 当前战况: 良性 {benign_collected}/1000 | 恶意 {malware_collected}/1000 (黑客估计还没起床，继续扫！)")
                
                # 满载退出机制
                if benign_collected >= MAX_SAMPLES_PER_CLASS and malware_collected >= MAX_SAMPLES_PER_CLASS:
                    print(f"        -> 触发保护机制：已收集足够的均衡样本 ({MAX_SAMPLES_PER_CLASS}良性/{MAX_SAMPLES_PER_CLASS}恶意)，提前安全退出！")
                    break
                
                if pkt_count > MAX_PKTS_PER_FILE:
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
                
                if key in label_dict:
                    label = label_dict[key]
                    
                    #  垃圾过滤器：一旦该类型收集满了，后面的全当垃圾丢掉，绝不放进内存！
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
        
    print(f"        -> 扫描了 {pkt_count} 个包，提取良性: {benign_collected}，提取恶意: {malware_collected}")
    return extracted_samples

def main():
    print(f"开始 CIC-IDS2017 本地预处理")
    
    csv_files = glob.glob(os.path.join(CSV_DIR, "*.csv"))
    all_extracted_data = []
    
    for csv_file in csv_files:
        base_name = os.path.basename(csv_file).replace("_ISCX.csv", "").replace("_Flow.csv", "").replace(".csv", "")
        if not base_name.endswith(".pcap"):
            base_name += ".pcap"
            
        pcap_file = os.path.join(PCAP_DIR, base_name)
        
        if os.path.exists(pcap_file):
            print("\n" + "="*60)
            print(f"发现配对文件: {base_name}")
            label_dict = build_label_dict(csv_file)
            if len(label_dict) == 0: continue
            
            samples = process_single_pcap(pcap_file, label_dict)
            all_extracted_data.extend(samples)
            
            if len(all_extracted_data) >= MAX_SAMPLES_PER_CLASS * 2:
                break

    print("\n" + "="*60)
    print(f"预处理安全完成！共提取 {len(all_extracted_data)} 个样本。")
    if len(all_extracted_data) > 0:
        torch.save(all_extracted_data, OUTPUT_FILE)
        labels = [s['label'].item() for s in all_extracted_data]
        print(f" 最终分布: 良性 (0) = {labels.count(0)} | 恶意 (1) = {labels.count(1)}")
        print(f" 数据已轻量级保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()