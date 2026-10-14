import random
import numpy as np


def digit_rule_split():
    """按数字末位规则划分已知/未知类"""
    all_classes = list(range(174))
    known = []
    unknown = []
    
    for num in all_classes:
        # 获取数字末位（0-9）
        last_digit = num % 10
        
        # 判断是否属于未知类（末位为3/5/7）
        if last_digit in {3,5,7}:
            unknown.append(num)
        else:
            known.append(num)
    
    return known, unknown

def process_file(input_file, output_a, output_b):
    # 打开输入和输出文件
    with open(input_file, 'r') as infile, \
         open(output_a, 'w') as file_a, \
         open(output_b, 'w') as file_b:
        
        # 逐行读取输入文件
        for line in infile:
            # 去除行首尾的空白字符，并分割为序号和类别
            line = line.strip()
            if not line:
                continue
            try:
                video_id, video_category = line.split()
                video_category = int(video_category)  # 转换序号为整数
            except ValueError:
                print(f"格式错误，跳过行: {line}")
                continue

            # 如果是奇数序号，直接存入 a.txt
            # known_classes, unknown_classes = digit_rule_split()
            if random.random() < 0.5:
                if video_category >121:
                    file_a.write(f"{video_id} {video_category}\n")
                else:
                    if random.random() < 0.5:
                        file_a.write(f"{video_id} {video_category}\n")
                    else:
                        file_b.write(f"{video_id} {video_category}\n")
            else:
                pass

# 使用示例
process_file('/mnt/newdisk/mnt/newdisk/dataset/action/ssv2_all/label/initial/v15_k70_u30/class_sort.txt', '/mnt/newdisk/mnt/newdisk/dataset/action/ssv2_all/label/initial/v15_k70_u30/unlabel.txt', '/mnt/newdisk/mnt/newdisk/dataset/action/ssv2_all/label/initial/v15_k70_u30/label.txt')