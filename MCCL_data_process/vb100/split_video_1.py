import os
import subprocess

# 设置输入和输出路径
path1 = "/mnt/newdisk/mnt/newdisk/dataset/DGC/DCCL/birds/video/all_video/"  # 替换为你的 path1
path2 = "/mnt/newdisk/mnt/newdisk/dataset/DGC/DCCL/birds/split/"  # 替换为你的 path2

# 确保输出目录存在
os.makedirs(path2, exist_ok=True)

# 遍历 path1 下的所有文件夹
for folder in os.listdir(path1):
    folder_path = os.path.join(path1, folder)
    
    if not os.path.isdir(folder_path):  # 确保是文件夹
        continue
    
    # 遍历文件夹内的 MP4 视频文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".mp4"):
            video_path = os.path.join(folder_path, filename)
            base_name = os.path.splitext(filename)[0]  # 获取文件名（不含扩展名）

            # 获取视频时长（单位：秒）
            cmd_duration = [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", video_path
            ]
            result = subprocess.run(cmd_duration, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            try:
                duration = float(result.stdout.strip())
            except ValueError:
                print(f"⚠️ 无法获取 {filename} 的时长，跳过...")
                continue
            
            # 计算每个片段的时长
            segment_duration = duration / 5  # 切割成 5 份
            
            # 执行切割
            for i in range(5):
                start_time = i * segment_duration
                output_filename = f"{base_name}_{i+1:02d}.mp4"
                output_path = os.path.join(path2, output_filename)

                # 调用 ffmpeg 进行视频切割，使用 libx264 编码器防止黑屏
                cmd_split = [
                    "ffmpeg", "-i", video_path, "-ss", str(start_time), "-t", str(segment_duration),
                    "-c:v", "libx264", "-c:a", "aac", "-strict", "experimental", "-avoid_negative_ts", "1",
                    output_path, "-y"
                ]
                subprocess.run(cmd_split, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # 静默执行

                print(f"✅ 已切割 {filename} -> {output_filename}")

print("🎉 所有视频切割完成！")