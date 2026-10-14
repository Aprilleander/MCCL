import json
import subprocess

# 读取JSON文件
def load_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

# 读取类模板文件
def load_templates(txt_file):
    template_map = {}
    with open(txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                index, template = parts
                template_map[template] = int(index)
    return template_map

# 生成输出文件
def generate_files(data, template_map, class_file, path_file, train_file):
    with open(class_file, 'w+', encoding='utf-8') as class_f, \
         open(path_file, 'w+', encoding='utf-8') as path_f, \
         open(train_file, 'w+', encoding='utf-8') as train_f:
        
        count=0
        for entry in data:
            count+=1
            id_ = int(entry['id'])
            template = entry['template'].replace('[', '').replace(']', '')

            if template in template_map:
                class_index = template_map[template]
            else:
                class_index =-1
                
            class_f.write(f"{id_} {class_index}\n")
            path_f.write(f"{id_} {id_}_frame\n")
            train_f.write(f"{id_} 0\n")
            print(id_,count,"well ")
        print("finished! ")


# 主函数
def main():
    json_file = '/mnt/newdisk/mnt/newdisk/dataset/action/ssv2_all/label/initial/v15_k70_u30/something-something-v2-train_v30_sort.json'  # 输入JSON文件
    txt_file = '/mnt/newdisk/mnt/newdisk/dataset/action/ssv2_all/label/initial/v30/initial_label/class_mapping.txt'  # 模板与类序号映射文件
    class_file = '/mnt/newdisk/mnt/newdisk/dataset/action/ssv2_all/label/initial/v5/class_val.txt'  # 输出class.txt
    path_file = '/mnt/newdisk/mnt/newdisk/dataset/action/ssv2_all/label/initial/v5/path_val.txt'  # 输出path.txt
    train_file = '/mnt/newdisk/mnt/newdisk/dataset/action/ssv2_all/label/initial/v5/is_train_val.txt'  # 输出is_train.txt

    data = load_json(json_file)
    template_map = load_templates(txt_file)
    generate_files(data, template_map, class_file, path_file, train_file)

if __name__ == "__main__":
    main()