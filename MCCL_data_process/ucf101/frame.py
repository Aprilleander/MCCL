import os
import shutil
from pathlib import Path

def select_and_copy_images(src_path, dest_path):
    # 确保目标路径存在
    os.makedirs(dest_path, exist_ok=True)
    
    # 遍历源路径的子文件夹
    for folder in os.listdir(src_path):
        folder_path = os.path.join(src_path, folder)
        
        # 确保当前路径是文件夹
        if not os.path.isdir(folder_path):
            continue
        
        # 获取所有图片，按名字排序
        images = sorted([img for img in os.listdir(folder_path) if img.endswith(('.jpg', '.png'))])
        total_images = len(images)
        
        # 计算需要挑选的图片数量（40%）

        num_select = max(1, int(total_images * 0.2))

        step = max(1, total_images // num_select)  # 间隔
        selected_indices = range(0, total_images, step)[:num_select]
        selected_images = [images[i] for i in selected_indices]
        
        # 创建目标文件夹
        dest_folder = os.path.join(dest_path, folder)
        os.makedirs(dest_folder, exist_ok=True)
        
        # 复制图片并重新编号
        for idx, image in enumerate(selected_images, start=1):
            src_image_path = os.path.join(folder_path, image)
            dest_image_name = f"{idx:06}.jpg"
            dest_image_path = os.path.join(dest_folder, dest_image_name)
            shutil.copy(src_image_path, dest_image_path)
        
        print(f"Processed folder: {folder}, Total: {total_images}, Selected: {num_select}")

# 示例调用
src_path = "/mnt/newdisk/mnt/newdisk/dataset/DGC/DCCL/UCF101/v100/UCF101_rgb/"  # 替换为你的源路径
dest_path = "/mnt/newdisk/mnt/newdisk/dataset/DGC/DCCL/UCF101/v100/UCF101_rgb_n/"  # 替换为你的目标路径
select_and_copy_images(src_path, dest_path)