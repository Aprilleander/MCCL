import csv
import os

def convert_csv_to_txt(csv_path, txt_path):
    os.makedirs(os.path.dirname(txt_path),exist_ok =True)

    with open(csv_path, 'r') as csv_file, open(txt_path, 'w') as txt_file:
        reader = csv.reader(csv_file)
        next(reader)  # 跳过表头
        for row in reader:
            imgid = row[0]
            target = row[2]
            txt_file.write(f"{imgid}.mp4 {target}\n")

# 使用示例
csv_path = '/mnt/newdisk/mnt/newdisk/dataset/action/ssv2_all/label/initial/v15/cub_unlabel_train_v15.csv'
txt_path = '/mnt/newdisk/mnt/newdisk/dataset/action/ssv2_all/label/initial/v15/videoclip/unlabel_val_v15.txt'
convert_csv_to_txt(csv_path, txt_path)