import os
import pandas as pd
import subprocess

def extract_frames_ffmpeg(path1, csv_file, path2, frame_rate=1):
    """
    使用 FFmpeg 提取视频帧，帧格式 '000001.jpg'
    
    :param path1: 视频存放路径
    :param csv_file: CSV 文件路径，第一列视频名称，第二列类别
    :param path2: 输出图片存放路径
    :param frame_rate: 提取帧率（1 表示每秒 1 帧，0.5 表示每 2 秒 1 帧）
    """
    # 读取 CSV 文件
    df = pd.read_csv(csv_file, header=0,dtype=str)
    df = df.iloc[:,:2]
    df.columns= ['video_file','category']

    # 确保目标路径存在
    os.makedirs(path2, exist_ok=True)

    for _, row in df.iterrows():
        video_name = str(row['video_file'])  # 视频文件名
        category = str(row['category'])  # 类别编号

        # 构造目标文件夹
        video_name = os.path.splitext(video_name)[0]
        target_folder = os.path.join(path2, f"{video_name}_g{category}")
        os.makedirs(target_folder, exist_ok=True)

        # 视频文件路径
        video_file = os.path.join(path1, f"{video_name}.mp4")
        if not os.path.exists(video_file):
            print(f"Warning: {video_file} not found!")
            continue

        # FFmpeg 命令
        ffmpeg_cmd = [
            "ffmpeg", "-i", video_file,  # 输入视频
            "-vf", f"fps={frame_rate}",  # 控制帧率
            os.path.join(target_folder, "%06d.jpg"),  # 输出帧文件
            "-hide_banner", "-loglevel", "error"  # 隐藏输出信息
        ]

        # 执行 FFmpeg
        subprocess.run(ffmpeg_cmd)
        print(f"Extracted frames from {video_file} -> {target_folder}")

# 示例调用
path1 = "/mnt/newdisk/mnt/newdisk/dataset/DGC/DCCL/ibc127/ibc-video/movie/"
csv_file = "/mnt/newdisk/mnt/newdisk/dataset/DGC/DCCL/ibc127/label/video.csv"
path2 = "/mnt/newdisk/mnt/newdisk/dataset/DGC/DCCL/ibc127/frame/"
frame_rate = 1  # 每秒提取 2 帧

extract_frames_ffmpeg(path1, csv_file, path2, frame_rate)