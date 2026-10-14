import os
import pandas as pd
import argparse

def parse_problematic_csv(csv_file):
    """
    解析格式有问题的CSV文件
    """
    try:
        # 首先尝试正常读取
        df = pd.read_csv(csv_file)
        
        # 检查是否所有数据都在第一行
        if len(df) == 1 and len(df.columns) == 1:
            print("检测到CSV格式问题，正在修复...")
            # 读取原始数据
            with open(csv_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # 分割行
            lines = content.split('\n')
            
            # 处理每一行
            data = []
            for line in lines:
                # 分割每行的字段
                parts = line.split(',')
                if len(parts) >= 3:
                    img_id = parts[0].strip()
                    filepath = parts[1].strip()
                    target = parts[2].strip()
                    data.append({'img_id': img_id, 'filepath': filepath, 'target': target})
            
            df = pd.DataFrame(data)
            print(f"修复后数据: {len(df)} 行")
        
        return df
        
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        # 尝试其他方法
        return read_csv_alternative(csv_file)

def read_csv_alternative(csv_file):
    """
    备用的CSV读取方法
    """
    print("使用备用方法读取CSV文件...")
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    data = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        # 分割字段
        parts = line.split(',')
        
        if i == 0:  # 标题行
            if len(parts) >= 3 and parts[0] == 'img_id':
                continue  # 跳过标题行
            else:
                # 如果第一行就是数据
                if len(parts) >= 3:
                    data.append({
                        'img_id': parts[0].strip(),
                        'filepath': parts[1].strip(), 
                        'target': parts[2].strip()
                    })
        else:
            # 数据行
            if len(parts) >= 3:
                data.append({
                    'img_id': parts[0].strip(),
                    'filepath': parts[1].strip(),
                    'target': parts[2].strip()
                })
    
    df = pd.DataFrame(data)
    print(f"备用方法读取到 {len(df)} 行数据")
    return df

def check_files_existence(csv_file, base_path):
    """
    检查CSV文件中的每个文件名是否存在于给定路径的子文件夹中
    """
    # 读取CSV文件（使用修复方法）
    df = parse_problematic_csv(csv_file)
    
    if df is None or len(df) == 0:
        print("无法读取CSV文件或文件为空")
        return
    
    print(f"成功读取CSV文件，共{len(df)}行数据")
    print("前5行数据预览:")
    print(df.head())
    
    # 检查基路径是否存在
    if not os.path.exists(base_path):
        print(f"警告: 基路径 '{base_path}' 不存在")
        return
    
    # 获取所有子文件夹
    subfolders = []
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path):
            subfolders.append(item_path)
    
    print(f"在基路径下找到 {len(subfolders)} 个子文件夹")
    
    # 存储结果
    results = []
    missing_files = []
    
    # 遍历CSV中的每一行
    for index, row in df.iterrows():
        img_id = str(row['img_id']).strip()
        file_found = False
        found_path = None
        
        print(f"正在检查: {img_id}")  # 调试信息
        
        # 在每个子文件夹中搜索
        for subfolder in subfolders:
            potential_path = os.path.join(subfolder, img_id)
            if os.path.exists(potential_path):
                file_found = True
                found_path = potential_path
                break
        
        results.append({
            'img_id': img_id,
            'exists': file_found,
            'found_path': found_path
        })
        
        if not file_found:
            missing_files.append(img_id)
    
    # 输出结果统计
    found_count = sum(1 for result in results if result['exists'])
    print(f"\n=== 检查结果 ===")
    print(f"总文件数: {len(results)}")
    print(f"找到的文件: {found_count}")
    print(f"缺失的文件: {len(missing_files)}")
    
    # 将缺失文件信息写入txt文件
    if missing_files:
        missing_txt_file = "missing_files.txt"
        with open(missing_txt_file, 'w', encoding='utf-8') as f:
            f.write("缺失文件列表:\n")
            f.write("=" * 50 + "\n")
            f.write(f"总文件数: {len(results)}\n")
            f.write(f"找到的文件: {found_count}\n")
            f.write(f"缺失的文件: {len(missing_files)}\n")
            f.write("=" * 50 + "\n\n")
            
            for i, missing_file in enumerate(missing_files, 1):
                f.write(f"{i:3d}. {missing_file}\n")
        
        print(f"\n缺失文件列表已保存到: {missing_txt_file}")
    else:
        print("\n恭喜！所有文件都已找到，没有缺失文件。")
    
    # 保存详细结果到CSV文件
    result_df = pd.DataFrame(results)
    output_csv = "file_check_results.csv"
    result_df.to_csv(output_csv, index=False)
    print(f"详细结果已保存到: {output_csv}")
    
    return results

def simple_check_with_txt():
    """
    简化版本，直接配置参数运行
    """
    # 配置参数
    csv_file = "your_csv_file.csv"  # 替换为你的CSV文件路径
    base_path = "/path/to/your/base/directory"  # 替换为你要搜索的基路径
    
    # 读取CSV文件（使用修复方法）
    df = parse_problematic_csv(csv_file)
    
    if df is None or len(df) == 0:
        print("无法读取CSV文件")
        return
    
    print(f"搜索基路径: {base_path}")
    print(f"检查 {len(df)} 个文件...\n")
    
    # 获取所有子文件夹
    subfolders = [os.path.join(base_path, item) 
                 for item in os.listdir(base_path) 
                 if os.path.isdir(os.path.join(base_path, item))]
    
    print(f"找到 {len(subfolders)} 个子文件夹")
    
    # 检查每个文件
    missing_files = []
    found_files = []
    
    for index, row in df.iterrows():
        img_id = str(row['img_id']).strip()
        found = False
        found_path = None
        
        for subfolder in subfolders:
            file_path = os.path.join(subfolder, img_id)
            if os.path.exists(file_path):
                found = True
                found_path = file_path
                break
        
        status = "✓ 找到" if found else "✗ 缺失"
        print(f"{status}: {img_id}")
        
        if found:
            found_files.append({"img_id": img_id, "path": found_path})
        else:
            missing_files.append(img_id)
    
    # 输出总结到控制台
    print(f"\n=== 总结 ===")
    print(f"总文件数: {len(df)}")
    print(f"找到的文件: {len(found_files)}")
    print(f"缺失的文件: {len(missing_files)}")
    
    # 将缺失文件写入txt
    if missing_files:
        missing_txt = "missing_files.txt"
        with open(missing_txt, 'w', encoding='utf-8') as f:
            f.write("缺失文件报告\n")
            f.write("=" * 60 + "\n")
            f.write(f"搜索路径: {base_path}\n")
            f.write(f"检查时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n")
            f.write(f"总文件数: {len(df)}\n")
            f.write(f"成功找到: {len(found_files)}\n")
            f.write(f"缺失文件: {len(missing_files)}\n")
            f.write("=" * 60 + "\n\n")
            f.write("缺失文件详情:\n")
            f.write("-" * 60 + "\n")
            
            for i, missing in enumerate(missing_files, 1):
                f.write(f"{i:4d}. {missing}\n")
        
        print(f"\n缺失文件列表已保存到: {missing_txt}")
    else:
        print("\n恭喜！所有文件都已找到，没有缺失文件。")

def main():
    parser = argparse.ArgumentParser(description='检查CSV中的文件是否存在于指定路径的子文件夹中')
    parser.add_argument('--csv', type=str, help='CSV文件路径')
    parser.add_argument('--path', type=str, help='要搜索的基路径')
    
    args = parser.parse_args()
    
    if args.csv and args.path:
        check_files_existence(args.csv, args.path)
    else:
        print("使用简化版本（请修改代码中的文件路径）")
        simple_check_with_txt()

if __name__ == "__main__":
    # 直接运行示例
    csv_file = "/mnt/newdisk/mnt/newdisk/dataset/action/k400/label/timesformer_v15/cub_unlabel_train.csv"  # 替换为你的CSV文件路径
    base_path = "/mnt/newdisk/mnt/newdisk/dataset/action/k400/video/v15/frame/train_256/"  # 替换为你要搜索的基路径
    
    check_files_existence(csv_file, base_path)