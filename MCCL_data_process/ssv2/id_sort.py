def load_file(file_path):
    """读取文件并返回每一行的列表"""
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            video_id = int(parts[0])  # 获取视频ID
            info = ' '.join(parts[1:])  # 保存视频的其他信息
            data.append((video_id, info))  # 保存原始视频ID和信息
    return data

def write_file(data, output_path):
    """将处理后的内容写入文件"""
    with open(output_path, 'w') as file:
        for new_id, info in data:
            file.write(f"{new_id} {info}\n")

def generate_mapping(old_to_new, output_path):
    """生成原始ID和新ID的映射文件"""
    with open(output_path, 'w') as file:
        for old_id, new_id in old_to_new.items():
            file.write(f"{old_id} {new_id}\n")

def reorder_video_ids(file1, file2, file3, output_file1, output_file2, output_file3, mapping_file):
    # 分别加载三个文件的内容
    data1 = load_file(file1)
    data2 = load_file(file2)
    data3 = load_file(file3)
    
    # 提取原始ID
    original_ids = [item[0] for item in data1]
    
    # 构建原始ID到新ID的映射（从1开始的顺序编号）
    old_to_new = {old_id: new_id+1 for new_id, old_id in enumerate(original_ids)}
    
    # 使用新的ID重新编号数据
    new_data1 = [(old_to_new[video_id], info) for video_id, info in data1]
    new_data2 = [(old_to_new[video_id], info) for video_id, info in data2]
    new_data3 = [(old_to_new[video_id], info) for video_id, info in data3]
    
    # 将重新编号后的数据写入新文件
    write_file(new_data1, output_file1)
    write_file(new_data2, output_file2)
    write_file(new_data3, output_file3)
    
    # 生成映射文件
    generate_mapping(old_to_new, mapping_file)

# 使用示例
file1 = '/mnt/newdisk/mnt/newdisk/dataset/DGC/DCCL/ssv2/labels/temp_label/images.txt'     # 文件1路径：'视频id 视频路径'
file2 = '/mnt/newdisk/mnt/newdisk/dataset/DGC/DCCL/ssv2/labels/temp_label/image_class_labels.txt'   # 文件2路径：'视频id 视频类'
file3 = '/mnt/newdisk/mnt/newdisk/dataset/DGC/DCCL/ssv2/labels/temp_label/train_test_split.txt'    # 文件3路径：'视频id 视频是否训练'

output_file1 = '/mnt/newdisk/mnt/newdisk/dataset/DGC/DCCL/ssv2/labels/temp_label/images_2.txt'
output_file2 = '/mnt/newdisk/mnt/newdisk/dataset/DGC/DCCL/ssv2/labels/temp_label/image_class_labels_2.txt'
output_file3 = '/mnt/newdisk/mnt/newdisk/dataset/DGC/DCCL/ssv2/labels/temp_label/train_test_split_2.txt'
mapping_file = '/mnt/newdisk/mnt/newdisk/dataset/DGC/DCCL/ssv2/labels/temp_label/mapping.txt'

reorder_video_ids(file1, file2, file3, output_file1, output_file2, output_file3, mapping_file)