import os
import json
import random

def generate_json_files(path1, path2):
    # 存储各个类别的视频文件数据
    label_data = []
    val_data = []
    unlabel_data = []

    # 遍历 path1 下的所有文件夹
    for folder in os.listdir(path1):
        folder_path = os.path.join(path1, folder)

        if not os.path.isdir(folder_path):
            continue
        
        # 获取文件夹中的类别信息
        parts = folder.split("_")

        if len(parts) ==2 and parts[-1].startswith("g"):
            try:
                # 提取 g 后面的数字部分，去掉前面的 "g"，并转化为整数
                class_id = int(parts[-1][1:])  # 例如 g016 -> 16
            except ValueError:
                print(f"⚠️ 无法解析类别 ID: {folder}, 跳过该文件夹...")
                continue
        else:
            print(f"⚠️ 类别格式不正确: {folder}, 跳过该文件夹...")
            continue

        # 构建当前文件夹的字典数据

        data = {"id": folder, "label": class_id}

        if class_id %2!=0:
            # 类别大于等于 51，全部写入 unlabel.json
            unlabel_data.append(data)
        else:
            # 类别小于 51，有一半的概率写入 label 或 unlabel
            if random.random() < 0.5:
                if random.random() < 0.9:
                    label_data.append(data)  # 90% 概率写入 label.json
                else:
                    val_data.append(data)  # 10% 概率写入 val.json
            else:
                unlabel_data.append(data)

    # 确保输出目录存在
    os.makedirs(path2, exist_ok=True)

    # 写入 JSON 文件
    with open(os.path.join(path2, 'ibc127_label_train.json'), 'w') as f:
        json.dump(label_data, f, indent=4)

    with open(os.path.join(path2, 'ibc127_label_validation.json'), 'w') as f:
        json.dump(val_data, f, indent=4)

    with open(os.path.join(path2, 'ibc127_unlabel_train.json'), 'w') as f:
        json.dump(unlabel_data, f, indent=4)

# 设置输入路径和输出路径
path1 = "/mnt/newdisk/mnt/newdisk/dataset/DGC/DCCL/ibc127/frame/"  # MP4 存放路径
path2 = "/mnt/newdisk/mnt/newdisk/dataset/DGC/DCCL/ibc127/label/v100/"  # 输出文件夹

# 生成 JSON 文件
generate_json_files(path1, path2)
