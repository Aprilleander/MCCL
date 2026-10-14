import os
import shutil

def count_images_and_delete(path1):
    if not os.path.exists(path1):
        print(f"路径 {path1} 不存在")
        return

    for folder in os.listdir(path1):
        folder_path = os.path.join(path1, folder)
        if not os.path.isdir(folder_path):
            continue  # 跳过非文件夹项

        # 统计文件夹内图片数量（常见图片格式）
        image_count = sum(1 for file in os.listdir(folder_path) 
                          if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')))

        print(f"文件夹 {folder} 内图片数量: {image_count}")

        # 删除图片数量少于 8 的文件夹
        if image_count < 8:
            shutil.rmtree(folder_path)
            print(f"已删除文件夹: {folder}")

# 使用示例
path1 = "/mnt/newdisk/mnt/newdisk/dataset/DGC/DCCL/ibc127/frame/"  # 修改为你的实际路径
count_images_and_delete(path1)