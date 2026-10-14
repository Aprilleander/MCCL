import os
import json
import random

def process_data(folder_path, txt_file_path):
    # 读取类别信息
    category_map = {}
    with open(txt_file_path, 'r') as f:
        for line in f:
            category_id, category_name = line.strip().split()
            category_map[category_name] = int(category_id) - 1  # 将类别号从1转换为从0开始

    # 获取文件夹名并按类别进行分类
    category_folders = {}
    for folder in os.listdir(folder_path):
        category_name = folder.rsplit('_')[1]
        if category_name not in category_folders:
            category_folders[category_name] = []
        category_folders[category_name].append(folder)

    labeled_samples = []
    unlabeled_samples = []
    val_samples = []
    s_label_samples = []

    # 处理每个类别
    for category_name, folders in category_folders.items():
        category_id = category_map[category_name]
        num_folders = len(folders)

        if category_id <51:  # 偶数个类别，取一半作为label和val
            random.shuffle(folders)
            label_folders = folders[:(num_folders // 2)]
            unlabel_folders = folders[(num_folders // 2):]

            label_train_folders = label_folders[:int(0.9*len(label_folders))]
            label_val_folders = label_folders[int(0.9*len(label_folders)):]
            

            labeled_samples.extend([{"id": folder, "label": category_id} for folder in label_train_folders])
            s_label_samples.extend([{"id": folder, "label": category_id} for folder in label_folders])

            val_samples.extend([{"id": folder, "label": category_id} for folder in label_val_folders])


            unlabeled_samples.extend([{"id": folder, "label": category_id} for folder in unlabel_folders])

        else:  # 奇数个类别，所有样本为unlabel
            unlabeled_samples.extend([{"id": folder, "label": category_id} for folder in folders])

    # 将结果写入JSON文件
    with open('/mnt/newdisk/mnt/newdisk/dataset/DGC/DCCL/UCF101/v100/label/v100_for/label.json', 'w') as f:
        json.dump(labeled_samples, f, indent=4)

    with open('/mnt/newdisk/mnt/newdisk/dataset/DGC/DCCL/UCF101/v100/label/v100_for/val.json', 'w') as f:
        json.dump(val_samples, f, indent=4)

    with open('/mnt/newdisk/mnt/newdisk/dataset/DGC/DCCL/UCF101/v100/label/v100_for/unlabel.json', 'w') as f:
        json.dump(unlabeled_samples, f, indent=4)

    # with open('/mnt/newdisk/mnt/newdisk/dataset/DGC/DCCL/UCF101/v100/label/v100/s_label.json', 'w') as f:
    #     json.dump(s_label_samples, f, indent=4)

# 使用示例
folder_path = '/mnt/newdisk/mnt/newdisk/dataset/DGC/DCCL/UCF101/v100/UCF101_rgb/'
txt_file_path = '/mnt/newdisk/mnt/newdisk/dataset/DGC/DCCL/UCF101/v100/label/ucfTrainTestlist/classInd.txt'
process_data(folder_path, txt_file_path)


