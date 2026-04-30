import os
import glob
from config import Config

def check_data_structure():
    print(f"正在检查目录: {Config.RAW_DATA_DIR}")
    
    # 1. 查找所有文件（不仅仅是 pcap）
    all_files = glob.glob(os.path.join(Config.RAW_DATA_DIR, "**", "*"), recursive=True)
    
    pcap_count = 0
    other_count = 0
    large_files = []
    
    print("\n--- 文件分布检查 ---")
    for f in all_files:
        if os.path.isdir(f):
            continue
            
        # 获取文件大小 (MB)
        size_mb = os.path.getsize(f) / (1024 * 1024)
        
        if f.endswith('.pcap'):
            pcap_count += 1
            # 记录超过 500MB 的大文件
            if size_mb > 500:
                large_files.append((f, size_mb))
        else:
            other_count += 1
            if other_count < 10: # 只打印前10个非pcap文件
                print(f"[非PCAP文件] {f} ({size_mb:.2f} MB)")

    print(f"\n--- 统计结果 ---")
    print(f"PCAP 文件数量: {pcap_count}")
    print(f"其他杂项文件: {other_count}")
    
    if large_files:
        print(f"\n[!!!] 警告：发现 {len(large_files)} 个巨型 PCAP 文件（可能是卡顿原因）：")
        for name, size in large_files:
            print(f"  -> {name}: {size:.2f} MB")
    else:
        print("\n未发现超过 500MB 的巨型文件。")

if __name__ == "__main__":
    check_data_structure()



   
    # ... 前面的代码不变 ...
    
    # 进度条 (修改这一段)
    # 把 tqdm 去掉，或者在里面加 write
    print("开始流式处理...")
    for idx, pcap_file in enumerate(all_pcap_files):
        # --- 新增：打印当前正在处理的文件名 ---
        file_size = os.path.getsize(pcap_file) / (1024*1024) # MB
        print(f"[{idx+1}/{len(all_pcap_files)}] 正在处理: {os.path.basename(pcap_file)} ({file_size:.2f} MB)")
        # ----------------------------------
        
        flows = process_pcap_fast(pcap_file)
        all_data.extend(flows)
    
    # ... 后面的代码不变 ...