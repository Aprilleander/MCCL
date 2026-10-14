import json

def generate_txt_files(a_json_file, b_json_file, output_a, output_b):
    # 读取分类文件 b.json
    with open(b_json_file, 'r') as b_file:
        category_map = json.load(b_file)  # 字典形式 {模板去括号: 类别编号}
    
    # 读取视频信息文件 a.json
    with open(a_json_file, 'r') as a_file:
        video_data = json.load(a_file)  # 假设为列表 [{id: ..., label: ..., template: ..., placeholder: ...}, ...]

    # 打开输出文件
    with open(output_a, 'w') as file_a, open(output_b, 'w') as file_b:
        for video in video_data:
            # 获取视频信息
            video_id = video["id"]
            template = video["template"]
            
            # 去掉模板中的中括号
            normalized_template = template.replace("[", "").replace("]", "")
            
            # 获取类别编号
            category_number = int(category_map.get(normalized_template, -1))  # 默认为 -1，如果没有匹配
            
            # 跳过未找到分类编号的视频
            if category_number == -1:
                print(f"警告: 未找到模板 '{normalized_template}' 对应的类别编号，跳过视频 ID: {video_id}")
                continue
            
            # 根据类别编号的奇偶性写入对应文件
            if category_number % 2 == 0:
                file_a.write(f"{video_id} {category_number}\n")
            else:
                file_b.write(f"{video_id} {category_number}\n")

    print(f"处理完成，结果已保存到 {output_a} 和 {output_b}")

# 使用示例
train_json = '/mnt/newdisk/mnt/newdisk/dataset/action/ssv2_all/label/initial/22w/something-something-v2-train.json'
class_json = '/mnt/newdisk/mnt/newdisk/dataset/action/ssv2_all/label/initial/22w/something-something-v2-labels.json'
label_txt = '/mnt/newdisk/mnt/newdisk/dataset/action/ssv2_all/label/initial/22w/label.txt'
unlabel_txt = '/mnt/newdisk/mnt/newdisk/dataset/action/ssv2_all/label/initial/22w/unlabel.txt'
generate_txt_files(train_json,class_json,label_txt,unlabel_txt)