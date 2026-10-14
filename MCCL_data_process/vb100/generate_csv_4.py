import os
import json
import csv
import re


def generate_csv_from_json(json_file, root_path, output_csv):
    """
    根据给定的 JSON 文件和路径，读取指定文件夹下的图片数量，生成 CSV 文件。

    Args:
        json_file (str): 输入的 JSON 文件路径。
        root_path (str): 包含视频文件夹的路径。
        output_csv (str): 输出 CSV 文件路径。
    """
    # 读取 JSON 文件
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 初始化 CSV 文件的内容
    csv_rows = []

    # 遍历 JSON 数据中的每个 id
    video_id_counter = 0  # video_id 从 0 开始

    with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        #writer.writerow(["original_video_id", "video_id", "frame_id", "path", "labels"])
        writer.writerow(["original_vido_id video_id frame_id path labels"])

        for item in data:
            folder_name = item["id"]  # 文件夹名称，例如 "K4X07_c096"
            # label = extract_category_as_number(folder_name)
            label = item["label"]
            folder_path = os.path.join(root_path, folder_name)

            # 检查文件夹是否存在
            if not os.path.isdir(folder_path):
                print(f"警告: 文件夹 {folder_path} 不存在，跳过。")
                continue

            # 读取文件夹下所有图片（假设图片格式为 jpg、png 等）
            image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

            # 遍历图片，生成 CSV 行
            for frame_id, image_name in enumerate(image_files):
                path = os.path.join(folder_name, image_name)  # 相对路径，例如 "K4X07_c096/000001.jpg"
                combined_info = f'{folder_name} {video_id_counter} {frame_id} {path} \"\"'
                # combined_info = f'{folder_name} {video_id_counter} {frame_id} {path} {label}'

                writer.writerow([combined_info])
            # video_id 递增
            video_id_counter += 1

    print(f"CSV 文件已生成: {output_csv}")


# 使用示例
if __name__ == "__main__":
    json_file = "/mnt/newdisk/mnt/newdisk/dataset/DGC/DCCL/birds/label/v100/vb100_unlabel_train.json"          # 替换为你的 JSON 文件路径
    root_path = "/mnt/newdisk/mnt/newdisk/dataset/DGC/DCCL/birds/frame/"    # 替换为包含视频文件夹的路径
    output_csv = "/mnt/newdisk/mnt/newdisk/dataset/DGC/DCCL/birds/label/v100/vb100_unlabel_train.csv"         # 替换为输出的 CSV 文件路径
    generate_csv_from_json(json_file, root_path, output_csv)