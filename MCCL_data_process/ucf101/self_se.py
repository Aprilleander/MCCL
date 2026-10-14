import json
import random

def filter_json(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    filtered_data = []
    for item in data:
        if item["label"] > 50:
            if random.random() > 0.5:
                continue  # 50% 概率舍弃
        filtered_data.append(item)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=4, ensure_ascii=False)

# 示例调用
input_file = "/mnt/newdisk/mnt/newdisk/dataset/DGC/DCCL/UCF101/v100/label/v100_for/ucf101_unlabel_train.json"  # 替换为你的输入文件
output_file = "/mnt/newdisk/mnt/newdisk/dataset/DGC/DCCL/UCF101/v100/label/v100_for/ucf101_unlabel_balance.json"
filter_json(input_file, output_file)