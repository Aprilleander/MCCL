import json
import os
import re

def process_videos(file_train,file_val,output_file):

    with open(output_file, 'w') as output_f:
        with open(file_train,'r') as file1:
            for line in file1:
                video_id,video_path = line.strip().split()
                output_f.write(f"{video_id} 1\n")

        with open(file_val,'r') as file2:
            for line in file2:
                video_id,video_path = line.strip().split()
                output_f.write(f"{video_id} 0\n")
            

def main():
    file_train = '/mnt/newdisk/mnt/newdisk/dataset/DGC/DCCL/ssv2/labels/generate_1/train_path.txt'  # 第一个JSON文件的路径，包含类别名称和ID
    file_val = '/mnt/newdisk/mnt/newdisk/dataset/DGC/DCCL/ssv2/labels/generate_1/val_path.txt'  # 第二个JSON文件的路径，包含视频信息
    output_file = '/mnt/newdisk/mnt/newdisk/dataset/DGC/DCCL/ssv2/labels/generate_1/train_test_split.txt'  # 视频文件夹的父目录路径
    
    process_videos(file_train,file_val,output_file)

if __name__ == "__main__":
    main()