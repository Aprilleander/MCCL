import random

def process_file(input_file, output_a, output_b):
    # 打开输入和输出文件
    with open(input_file, 'r') as infile, \
         open(output_a, 'w') as file_a, \
         open(output_b, 'w') as file_b:
        
        # 逐行读取输入文件
        count=0
        for line in infile:
            # 去除行首尾的空白字符，并分割为序号和类别
            line = line.strip()
            if not line:
                continue
            try:
                video_id, video_category = line.split()
                video_category = int(video_category)  # 转换序号为整数
            except ValueError:
                print(f"格式错误，跳过行: {line}")
                continue

            # 如果是奇数序号，直接存入 a.txt
            if video_category % 2 == 1:
                file_a.write(f"{video_id} {video_category}\n")
            # 如果是偶数序号，随机决定存储到哪个文件
            else:
                if random.random() < 0.5:
                    file_a.write(f"{video_id} {video_category}\n")
                else:
                    file_b.write(f"{video_id} {video_category}\n")

# 使用示例

class_txt = 'mnt/newdisk/mnt/newdisk/dataset/action/k400/label/initial/kinetics_label_map.txt'
label_txt =  '/mnt/newdisk/mnt/newdisk/dataset/action/k400/label/timesformer/label.txt'
unlabel_txt ='/mnt/newdisk/mnt/newdisk/dataset/action/k400/label/timesformer/unlabel.txt'

process_file(class_txt, label_txt, unlabel_txt)