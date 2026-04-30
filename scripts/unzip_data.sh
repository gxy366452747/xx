#!/bin/bash

# 设置数据目录路径 (根据你的实际情况调整)
RAW_DIR="./data/raw"

# --- 1. 解压 APT2020 (ZIP 文件) ---
echo "========================================"
echo "正在处理 ADPT2020 数据集..."
# 假设 APT2020 的文件名包含 "2020" 或 "adpt"，这里用通配符匹配
# 如果你的文件名不一样，请修改 *.zip
zip_file=$(find "$RAW_DIR" -maxdepth 1 -name "*2020*.zip" | head -n 1)

if [ -f "$zip_file" ]; then
    echo "发现压缩包: $zip_file"
    # 解压到 raw/APT2020 文件夹，避免散落在外面
    target_dir="$RAW_DIR/ADPT2020"
    mkdir -p "$target_dir"
    
    echo "正在解压到 $target_dir ..."
    unzip -q "$zip_file" -d "$target_dir"
    echo "ADPT2020 解压完成！"
else
    echo "未在 $RAW_DIR 下找到 ADPT2020 的 zip 文件，跳过。"
fi


echo "========================================"
echo "所有解压任务执行完毕！"
echo "请检查 $RAW_DIR 目录下是否已生成 .pcap 文件。"