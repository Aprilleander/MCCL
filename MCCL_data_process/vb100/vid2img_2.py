import os
import re
import subprocess

def read_class_mapping(txt_path):
    """读取类别映射表，返回 类名 -> 类ID 字典"""
    class_mapping = {}
    try:
        with open(txt_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                
                # 仅分割一次，确保类别名完整
                parts = line.rsplit(" ", maxsplit=1)
                if len(parts) == 2 and parts[1].isdigit():
                    class_name = parts[0].strip()  # 类别名
                    class_id = int(parts[1])  # 类别 ID
                    class_mapping[class_name] = class_id
                else:
                    print(f"⚠️ 无效的类别行: {line}")
                    
    except Exception as e:
        print(f"⚠️ 读取类别文件 {txt_path} 失败: {e}")
    
    return class_mapping

def extract_class_name(filename):
    """去掉最后两个 `_` 及其后的部分，返回类别名"""
    parts = filename.split("_")  # 按 `_` 拆分
    if len(parts) < 3:
        return parts[0]  # 直接取第一个部分作为类别
    return "_".join(parts[:-2])  # 去掉最后两个字段，拼接回类别名

# 处理 MP4 文件
def process_videos(path1, path2, class_mapping, fps=5):
    """遍历所有 MP4 文件，分割并提取帧"""
    if not os.path.exists(path2):
        os.makedirs(path2)

    for file in os.listdir(path1):
        if file.endswith(".mp4"):
            try:
                file_path = os.path.join(path1, file)
                filename_no_ext = os.path.splitext(file)[0]  # 获取无扩展名的文件名
                
                # 提取类别名
                class_name = extract_class_name(filename_no_ext)
                
                if class_name:
                    class_id = class_mapping[class_name]
                    target_folder = os.path.join(path2, f"{filename_no_ext}_g{class_id:03d}")
                    os.makedirs(target_folder, exist_ok=True)

                    # 使用 ffmpeg 提取帧
                    save_frames_ffmpeg(file_path, target_folder, fps)
                else:
                    print(f"⚠️ 类别未找到: {file}，跳过该文件")

            except Exception as e:
                print(f"❌ 处理 {file} 失败: {e}")
                continue  # 继续下一个视频

# 使用 ffmpeg 分割视频并保存帧
def save_frames_ffmpeg(video_path, output_folder, fps):
    """调用 ffmpeg 提取帧"""
    try:
        command = [
            'ffmpeg',
            '-i', video_path,  # 输入视频
            '-vf', f'fps={fps}',  # 设置帧率
            os.path.join(output_folder, '%06d.jpg')  # 帧的命名格式
        ]
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)

        if result.returncode != 0:
            print(f"⚠️ ffmpeg 处理 {video_path} 失败: {result.stderr}")

    except Exception as e:
        print(f"❌ 提取 {video_path} 帧失败: {e}")

# 参数配置
path1 = "/mnt/newdisk/mnt/newdisk/dataset/DGC/DCCL/birds/split/"  # MP4 存放路径
path2 = "/mnt/newdisk/mnt/newdisk/dataset/DGC/DCCL/birds/frame/"  # 输出文件夹
txt_path = "/mnt/newdisk/mnt/newdisk/dataset/DGC/DCCL/birds/video/vb100_annotations_v1/species.txt"  # TXT 文件路径

# 执行程序
class_mapping = read_class_mapping(txt_path)
process_videos(path1, path2, class_mapping)