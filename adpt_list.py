import os
import argparse
from pathlib import Path


def list_directory_structure(path, prefix="", show_files=True, max_depth=None, current_depth=0):
    """
    递归列出文件夹结构
    
    参数:
        path: 目标路径
        prefix: 前缀（用于缩进）
        show_files: 是否显示文件（False则只显示文件夹）
        max_depth: 最大递归深度（None表示无限制）
        current_depth: 当前深度
    """
    if max_depth is not None and current_depth > max_depth:
        return
    
    try:
        items = os.listdir(path)
        # 文件夹在前，文件在后
        dirs = sorted([item for item in items if os.path.isdir(os.path.join(path, item))])
        files = sorted([item for item in items if os.path.isfile(os.path.join(path, item))])
        
        if show_files:
            items_to_show = dirs + files
        else:
            items_to_show = dirs
            
    except PermissionError:
        print(f"{prefix}└── [无权限访问]")
        return
    except FileNotFoundError:
        print(f"路径不存在: {path}")
        return
    except Exception as e:
        print(f"{prefix}└── [错误: {e}]")
        return
    
    for index, item in enumerate(items_to_show):
        item_path = os.path.join(path, item)
        is_last = index == len(items_to_show) - 1
        is_dir = os.path.isdir(item_path)
        
        # 选择连接符
        connector = "└── " if is_last else "├── "
        
        # 文件夹显示 📁，文件显示 📄
        icon = "📁 " if is_dir else "📄 "
        print(f"{prefix}{connector}{icon}{item}")
        
        # 如果是文件夹，递归进入
        if is_dir:
            extension = "    " if is_last else "│   "
            list_directory_structure(
                item_path, 
                prefix + extension, 
                show_files, 
                max_depth, 
                current_depth + 1
            )


def get_directory_stats(path):
    """获取目录统计信息"""
    total_dirs = 0
    total_files = 0
    total_size = 0
    
    for root, dirs, files in os.walk(path):
        total_dirs += len(dirs)
        total_files += len(files)
        for file in files:
            try:
                file_path = os.path.join(root, file)
                total_size += os.path.getsize(file_path)
            except (OSError, FileNotFoundError):
                pass
    
    return total_dirs, total_files, total_size


def format_size(size_bytes):
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='列出目录结构')
    # ⚠️ 这里也要改：加 r 前缀，解决 \d 警告
    parser.add_argument('path', nargs='?', default=r'DeepTraffic-CL\data\raw\CTU-13',
                        help=r'目标路径（默认: DeepTraffic-CL\data\raw\CTU-13）')
    parser.add_argument('-d', '--depth', type=int, default=None,
                        help='最大显示深度')
    parser.add_argument('--dirs-only', action='store_true',
                        help='只显示文件夹')
    parser.add_argument('-s', '--stats', action='store_true',
                        help='显示统计信息')
    
    args = parser.parse_args()
    
    target_path = os.path.expanduser(args.path)
    
    # 检查路径是否存在
    if not os.path.exists(target_path):
        print(f"❌ 路径不存在: {target_path}")
        return
    
    if not os.path.isdir(target_path):
        print(f"❌ 不是有效的目录: {target_path}")
        return
    
    # 打印标题
    print(f"\n{'='*60}")
    print(f"📂 目录结构: {target_path}")
    print(f"{'='*60}\n")
    
    # 列出结构
    list_directory_structure(
        target_path, 
        show_files=not args.dirs_only,
        max_depth=args.depth
    )
    
    # 显示统计信息
    if args.stats:
        print(f"\n{'='*60}")
        print("📊 统计信息:")
        dirs, files, size = get_directory_stats(target_path)
        print(f"   文件夹数量: {dirs}")
        print(f"   文件数量:   {files}")
        print(f"   总大小:     {format_size(size)}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
    # ✅ 就这一行，后面那些 print 全部删掉！