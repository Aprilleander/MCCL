import json
import random

def filter_videos_by_ids(txt_path, json_path, output_path,output_path2):
    # 读取原始JSON文件（假设是数组格式）
    with open(json_path, 'r', encoding='utf-8') as f:
        video_dict = {item['id']: item for item in json.load(f)}

    # 读取需要查询的ID列表（保持顺序）
    with open(txt_path, 'r', encoding='utf-8') as f:
        target_ids = [(line.strip().split()[0]) for line in f if line.strip()]

    # 构建结果列表（保持原始顺序）
    result = []
    result2 =[]
    for vid in target_ids:
        if random.random() < 0.9:
            if vid in video_dict:
                result.append(video_dict[vid])
            else:
                print(f"警告：ID {vid} 未在JSON中找到")
        else:
            if vid in video_dict:
                result2.append(video_dict[vid])
            else:
                print(f"警告：ID {vid} 未在JSON中找到")

        
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    with open(output_path2, 'w', encoding='utf-8') as f:
        json.dump(result2, f, ensure_ascii=False, indent=2)


# 使用示例
filter_videos_by_ids('/mnt/newdisk/mnt/newdisk/dataset/action/ssv2_all/label/initial/v100_cutv15/file2.txt', '/mnt/newdisk/mnt/newdisk/dataset/action/ssv2_all/label/initial/v100_cutv15/something-something-v2-train.json', '/mnt/newdisk/mnt/newdisk/dataset/action/ssv2_all/label/initial/v100_cutv15/something-something-v2-train_v70.json', '/mnt/newdisk/mnt/newdisk/dataset/action/ssv2_all/label/initial/v100_cutv15/something-something-v2-validation_v70.json')