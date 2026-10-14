import os
import shutil

def copy_video_folders(txt_file, source_dir, dest_dir):
    """
    根据txt中的视频号，将对应的视频文件夹从source_dir复制到dest_dir。
    
    参数:
        txt_file (str): 包含 '视频号 视频类别' 的 txt 文件路径。
        source_dir (str): 视频文件夹的源路径。
        dest_dir (str): 目标路径，复制文件夹到该路径。
    """
    # 创建目标目录，如果不存在则创建
    os.makedirs(dest_dir, exist_ok=True)

    # 从txt文件中读取视频号
    with open(txt_file, 'r') as file:
        lines = file.readlines()
        video_ids = {line.split()[0] for line in lines}  # 提取视频号，去重
    
    # 遍历源路径中的所有文件夹
    for folder_name in os.listdir(source_dir):
        folder_path = os.path.join(source_dir, folder_name)

        # 检查是否是目录，以及文件夹名是否匹配视频号
        if os.path.isdir(folder_path):
            video_id = folder_name.split('_')[0]  # 提取文件夹名前缀的视频号
            if video_id in video_ids:
                # 复制文件夹到目标路径
                dest_path = os.path.join(dest_dir, folder_name)
                shutil.copytree(folder_path, dest_path, dirs_exist_ok=True)
                print(f"复制成功: {folder_name} -> {dest_path}")
    
    print(f"处理完成，文件夹已复制到 {dest_dir}")

# 使用示例
input_txt = '/mnt/newdisk/mnt/newdisk/dataset/action/ssv2_all/label/initial/v70/label70.txt'
sour = '/mnt/newdisk/mnt/newdisk/dataset/action/ssv2_all/ssv2_frame_all/'
dist = '/mnt/newdisk/mnt/newdisk/dataset/action/ssv2_all/ssv2_label_v70/'
copy_video_folders(input_txt, sour, dist)