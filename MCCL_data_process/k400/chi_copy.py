import os
import shutil
import csv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def copy_file(src_file, dest_file):
    os.makedirs(os.path.dirname(dest_file), exist_ok=True)
    shutil.copy2(src_file, dest_file)

def copy_files_from_csv(csv_file, path1, path2, num_workers=4):
    if not os.path.exists(path2):
        os.makedirs(path2)
    
    files_to_copy = []
    with open(csv_file, newline='', encoding='utf-8') as file:
        lines = file.readlines()
        header = lines[0]  # 跳过表头
        for line in lines[1:]:
            row = line.strip().split()  # 按空格分割字符串
            if len(row) >= 4:
                relative_path = row[3]  # 例如 'abseiling/FF2UY_0079_0089/000003.jpg'
                src_file = os.path.join(path1, relative_path)
                dest_file = os.path.join(path2, relative_path)
                files_to_copy.append((src_file, dest_file))
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(lambda x: copy_file(*x), files_to_copy), total=len(files_to_copy), desc="Copying Files"))

# 示例调用
csv_file = "/mnt/newdisk/mnt/newdisk/dataset/action/k400/label/timesformer_v15_for/unlabel.csv"  # CSV 文件路径
path1 = "/mnt/newdisk/mnt/newdisk/dataset/action/k400/video/v30/frame/train_256/"  # 视频文件总路径
path2 = "/mnt/newdisk/mnt/newdisk/dataset/action/k400/video/v15/frame/train_256/"  # 目标路径
copy_files_from_csv(csv_file, path1, path2,num_workers=4)