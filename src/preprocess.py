import os
import glob
import numpy as np
import torch
from scapy.all import PcapReader, IP, TCP, UDP, Raw
from tqdm import tqdm
from config import Config
import logging

# 关闭 Scapy 警告
logging.getLogger("scapy.runtime").setLevel(logging.ERROR)

# --- 限制配置 ---
# 防止 10GB 的大文件卡死程序
# 每个文件最多读取5万包就强行停止
MAX_PKTS_PER_FILE = 50000 

def process_pcap_fast(pcap_path):
    """
    流式解析 PCAP
    """
    flows = {}      
    flow_counts = {} 
    
    total_pkts_read = 0 # 计数器
    
    try:
        # 使用流式读取
        with PcapReader(pcap_path) as reader:
            for pkt in reader:
                total_pkts_read += 1
                
                # 刹车
                if total_pkts_read > MAX_PKTS_PER_FILE:
                    # 默默地停止，不打印太多信息干扰进度条
                    break
                
                if IP not in pkt: continue
                
                # 协议提取
                if TCP in pkt:
                    proto = 6
                    sport = pkt[TCP].sport
                    dport = pkt[TCP].dport
                elif UDP in pkt:
                    proto = 17
                    sport = pkt[UDP].sport
                    dport = pkt[UDP].dport
                else:
                    continue 

                src_ip = pkt[IP].src
                dst_ip = pkt[IP].dst

                # 生成唯一 Key: (IP1, Port1, IP2, Port2, Proto)
                # sorted() 保证了 A发给B 和 B发给A 算作同一条流
                key = tuple(sorted([(src_ip, sport), (dst_ip, dport)]) + [proto])

                if key not in flow_counts:
                    flow_counts[key] = 0
                    flows[key] = []


                # 只保留前 MAX_SEQ_LEN(20) 个包
                # 把我们需要的信息提取出来，打包成一个小字典
                if flow_counts[key] < Config.MAX_SEQ_LEN:
                    pkt_data = {
                        'time': float(pkt.time),# 时间戳 (为了算时间间隔)
                        'len': len(pkt),# 包大小
                        'payload': bytes(pkt[Raw].load) if Raw in pkt else b''# 载荷 (除去包头的数据)
                    }
                    flows[key].append(pkt_data)
                    flow_counts[key] += 1
                
    except Exception as e:
        print(f"\n[Error] 读取出错 {pcap_path}: {e}")
        return []

    processed_flows = []
    
    # 简单的标签规则
    path_lower = pcap_path.lower()
    # 根据文件夹名字打标
    if 'benign' in path_lower:
        label = 0
    elif 'malware' in path_lower:
        label = 1
    else:
        label = -1 # 无标签/未知

    # 后处理：提取特征
    for key, pkts in flows.items():
        if len(pkts) < 2: continue 

        # 按时间排序，保证包的顺序是正确的
        pkts.sort(key=lambda x: x['time'])

        # --- 1. 包长序列 ---
        seq_features = []
        prev_time = pkts[0]['time']
        
        for i, p in enumerate(pkts[:Config.MAX_SEQ_LEN]):
            # 包大小归一化
            # 网络包最大一般是 1500 字节 (MTU)。
            pkt_len = p['len'] / 1500.0
            #计算时间间隔
            cur_time = p['time']
            iat = cur_time - prev_time if i > 0 else 0.0
            prev_time = cur_time
            # 截断：如果间隔超过1秒，就记为1秒。防止某些异常大的数值影响模型。
            iat = min(iat, 1.0)
            seq_features.append([pkt_len, iat])

        # Padding 
        #超过截断，不够补齐，我们选用的是20个包
        if len(seq_features) < Config.MAX_SEQ_LEN:
            pad_len = Config.MAX_SEQ_LEN - len(seq_features)
            seq_features.extend([[0.0, 0.0]] * pad_len)

        # --- 2. 载荷特征 ---
        raw_payload = b''
        #  把这条流里所有包的 payload (除去包头的数据) 拼接到一起 
        # 超过784截断
        for p in pkts:
            raw_payload += p['payload']
            if len(raw_payload) >= Config.MAX_PAYLOAD_LEN:
                break
        #把二进制字节流 转换成数字数组 
        payload_arr = np.frombuffer(raw_payload, dtype=np.uint8)

        #补齐和截断
        if len(payload_arr) < Config.MAX_PAYLOAD_LEN:
            payload_arr = np.pad(payload_arr, (0, Config.MAX_PAYLOAD_LEN - len(payload_arr)), 'constant')
        else:
            payload_arr = payload_arr[:Config.MAX_PAYLOAD_LEN]
        
        # 归一化 
        # 字节的值范围是 0-255。  
        payload_norm = payload_arr.astype(np.float32) / 255.0
        
        #3. 打包成 Tensor (张量)
        processed_flows.append({
            'payload': torch.tensor(payload_norm, dtype=torch.float32),
            #原始载荷[784]
            'sequence': torch.tensor(seq_features, dtype=torch.float32),
            #包长序列[20, 2]
            'label': torch.tensor(label, dtype=torch.long)
            #标签[0, 1, -1]
        })

    return processed_flows

def main():
    print(f"扫描路径: {Config.RAW_DATA_DIR}")
    print(f" 每个文件最大读取包数: {MAX_PKTS_PER_FILE}")
    
    search_pattern = os.path.join(Config.RAW_DATA_DIR, "**", "*.pcap")
    all_pcap_files = glob.glob(search_pattern, recursive=True)
    
    # 按文件大小排序，先处理小的，把大的留最后（或者反过来）
    # 不排序，直接跑
    print(f"找到 {len(all_pcap_files)} 个 PCAP 文件。开始处理...")
    
    all_data = []
    
    # 使用 tqdm 显示总进度
    pbar = tqdm(all_pcap_files)
    for pcap_file in pbar:
        # 在进度条旁边显示当前文件名 (只显示文件名，不显示路径)
        fname = os.path.basename(pcap_file)
        pbar.set_description(f"Processing {fname[:20]}...") 
        
        flows = process_pcap_fast(pcap_file)
        all_data.extend(flows)
        
    print(f"\n全部处理完成！共提取流: {len(all_data)}")
    
    save_path = os.path.join(Config.PROCESSED_DATA_DIR, "all_traffic_data.pt")
    torch.save(all_data, save_path)
    print(f"数据已保存至: {save_path}")

if __name__ == "__main__":
    main()