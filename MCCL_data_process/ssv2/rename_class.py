import os

def rename_video_folders(txt_file, video_folder_path):
    # 读取txt文件内容
    id_to_category = {}
    with open(txt_file, "r") as f:
        for line in f:
            video_id, category = line.strip().split()
            id_to_category[video_id] = category
    
    # 遍历视频文件夹并重命名
    for folder_name in os.listdir(video_folder_path):
        if not os.path.isdir(os.path.join(video_folder_path, folder_name)):
            continue  # 跳过非文件夹项
        
        # 解析原始文件夹名，确保包含对应的video_id
        parts = folder_name.split('_')
        video_id = parts[0]
         
        # 获取视频类别，并重命名文件夹
        if video_id in id_to_category:
            category = id_to_category[video_id]
            new_name = f"c{category}_{folder_name}"
            old_path = os.path.join(video_folder_path, folder_name)
            new_path = os.path.join(video_folder_path, new_name)
            os.rename(old_path, new_path)
            print(f"Renamed: {folder_name} -> {new_name}")
        else:
            print(f"Video ID {video_id} not found in TXT file. Skipping {folder_name}.")

# 示例用法
txt_file = "/mnt/newdisk/mnt/newdisk/dataset/DGC/DCCL/ssv2/labels/temp_label/image_class_labels.txt"  # 替换为你的txt文件路径
video_folder_path = "/mnt/newdisk/mnt/newdisk/dataset/DGC/DCCL/ssv2/ssv2_frame_rename_class/"  # 替换为你的视频文件夹路径
rename_video_folders(txt_file, video_folder_path)