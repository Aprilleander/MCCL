import os

def rename_images(base_path):
    """
    遍历给定路径，修改子文件夹中图片文件名，去掉前缀 'img_'。
    :param base_path: 根目录路径
    """
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.startswith("img_") and file.endswith(".jpg"):
                old_path = os.path.join(root, file)
                new_filename = file.replace("img_", "", 1)  # 去掉前缀 'img_'
                new_path = os.path.join(root, new_filename)
                
                # 重命名文件
                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} -> {new_path}")

# 示例用法
base_path = "/mnt/newdisk/mnt/newdisk/dataset/action/k400/video/v30/frame/train_256/"  # 替换为实际路径
rename_images(base_path)