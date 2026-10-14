import json
import os
import re

def load_category_file(filepath):
    category_data ={}
    with open(filepath,'r',encoding='utf-8') as f:
        for index, line in enumerate(f):
            category_name = line.strip()
            category_data[category_name] =str(index)
    return category_data


def load_json_file(filepath):
    """加载JSON文件并返回数据"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def write_class_file(category_data, output_class_file):
    """生成 class.txt 文件"""
    with open(output_class_file, 'w', encoding='utf-8') as f:
        for index, (category_name, category_id) in enumerate(category_data.items()):
            f.write(f"{category_id} {category_name}\n")

def clean_template(template):
    return template.replace("[","").replace("]","").strip()

def process_videos(video_data, category_data, video_folder_path, output_image_file, output_class_id_file):
    """生成 image.txt 和 class_id.txt 文件，判断视频文件夹是否存在"""
    with open(output_image_file, 'w', encoding='utf-8') as f_image, open(output_class_id_file, 'w', encoding='utf-8') as f_class_id:
        for video in video_data:
            video_id = video['id']
            folder_path = os.path.join(video_folder_path, video_id)
            
            # 判断该视频ID的文件夹是否存在
            if os.path.exists(folder_path):
                # 写入 image.txt 文件
                f_image.write(f"{video_id} {video_id}\n")
                
                # 查找 template 对应的类别ID
                template = clean_template(video['template'])

                matched = False
                for category_name, category_id in category_data.items():
                    if template ==category_name:  # 假设类别名称在模板的开头
                        # 写入 class_id.txt 文件
                        f_class_id.write(f"{video_id} {category_id}\n")
                        matched =True
                        break
                if not matched:
                    print("no match!")
                    print(folder_path)

def main():
    # 文件路径
    categories_file = '/mnt/newdisk/mnt/newdisk/dataset/DGC/DCCL/ssv2/labels/category.txt'  # 第一个JSON文件的路径，包含类别名称和ID
    videos_file = '/mnt/newdisk/mnt/newdisk/dataset/DGC/DCCL/ssv2/labels/train.json'  # 第二个JSON文件的路径，包含视频信息
    video_folder_path = '/mnt/newdisk/mnt/newdisk/dataset/DGC/DCCL/ssv2/ssv2_frame/'  # 视频文件夹的父目录路径
    output_class_file = '/mnt/newdisk/mnt/newdisk/dataset/DGC/DCCL/ssv2/labels/generate_1/class.txt'  # 输出的类别文件
    output_image_file = '/mnt/newdisk/mnt/newdisk/dataset/DGC/DCCL/ssv2/labels/generate_1/test_path.txt'  # 输出的视频ID与文件夹名称文件
    output_class_id_file = '/mnt/newdisk/mnt/newdisk/dataset/DGC/DCCL/ssv2/labels/generate_1/test_video_class.txt'  # 输出的视频ID与类别ID文件

    # 加载 JSON 数据
    category_data = load_category_file(categories_file)
    video_data = load_json_file(videos_file)

    # 生成 class.txt 文件
    #write_class_file(category_data, output_class_file)

    # 处理视频信息，生成 image.txt 和 class_id.txt 文件
    process_videos(video_data, category_data, video_folder_path, output_image_file, output_class_id_file)

if __name__ == "__main__":
    main()