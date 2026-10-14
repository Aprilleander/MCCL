import json
import random

def filter_json_by_id(json_file, txt_file, output_json,output_json_2):
    # 读取txt文件中存在的id并存储为字典
    id_to_class = {}
    with open(txt_file, 'r') as txt:
        for line in txt:
            line = line.strip()
            if not line:
                continue
            try:
                id_, class_ = line.split()
                id_to_class[id_] = class_
            except ValueError:
                print(f"格式错误，跳过行: {line}")
                continue
    
    # 读取json文件，过滤符合条件的项
    filtered_data = []
    filtered_data_2 = []

    with open(json_file, 'r') as json_file:
        data = json.load(json_file)  # 假定 JSON 文件中存储的是列表形式
        for item in data:
            if str(item["id"]) in id_to_class:
                filtered_data.append(item)
                # if random.random() < 0.9:
                #     filtered_data.append(item)
                # else:
                #     filtered_data_2.append(item)


    # 将结果保存为新的json文件
    
    with open(output_json, 'w') as outfile:
        json.dump(filtered_data, outfile, indent=4, ensure_ascii=False)
    # with open(output_json_2, 'w') as outfile:
    #     json.dump(filtered_data_2, outfile, indent=4, ensure_ascii=False)

    print(f"处理完成，结果已保存到 {output_json}")

# 使用示例
train_json = '/mnt/newdisk/mnt/newdisk/dataset/action/ssv2_all/label/initial/v15_k70_u30/train_v30.json'
label_txt = '/mnt/newdisk/mnt/newdisk/dataset/action/ssv2_all/label/initial/v15_k70_u30/unlabel.txt'
output_json = '/mnt/newdisk/mnt/newdisk/dataset/action/ssv2_all/label/initial/v15_k70_u30/something-something-v2-unlabel_train_v15.json'
# output_json_2 = '/mnt/newdisk/mnt/newdisk/dataset/action/ssv2_all/label/initial/v30_k70_u30/something-something-v2-label_validation_v30.json'
output_json_2 = None
filter_json_by_id(train_json, label_txt, output_json,output_json_2)