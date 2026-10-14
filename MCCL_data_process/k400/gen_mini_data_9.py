import json
import pandas as pd
import random
import shutil
import os

def process_files(json_path,output_json):
    # 读取 JSON 文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 随机筛选 1/3 的数据
    sampled_data = random.sample(data, len(data) // 4)
    sampled_ids = {str(item['id']) for item in sampled_data}  # 确保 ID 为字符串
    
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, ensure_ascii=False, indent=4)
    
    # 处理 cub.csv
    # cub_df = pd.read_csv(cub_csv_path, sep=' ')
    # cub_df['img_id'] = cub_df['img_id'].astype(str)
    # cub_df[cub_df['img_id'].isin(sampled_ids)].to_csv(output_cub, sep=' ', index=False)
    
    # 处理 label_train.csv
    # label_df = pd.read_csv(label_csv_path, sep=' ')
    # label_df['original_vido_id '] = label_df['original_vido_id '].astype(str)
    # label_df[label_df['original_vido_id '].isin(sampled_ids)].to_csv(output_label, sep=' ', index=False)


# 调用示例
if __name__ == "__main__":

    input_folder='/mnt/newdisk/mnt/newdisk/dataset/action/k400/label/timesformer_v5/timesformer_v15/'
    output_folder='/mnt/newdisk/mnt/newdisk/dataset/action/k400/label/timesformer_v5/timesformer_v4/'

    json_path = os.path.join(input_folder,'label_validation.json')
    
    output_json = os.path.join(output_folder,'label_validation.json')

    process_files(json_path, output_json)