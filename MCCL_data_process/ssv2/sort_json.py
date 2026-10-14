import json

def sort_json_by_class(txt_path, json_path, output_path):
    # 1. 读取 txt 中 id 到 class 的映射
    id_to_class = {}
    with open(txt_path, 'r') as f:
        for line in f:
            if line.strip():
                vid, cls = line.strip().split()
                id_to_class[int(vid)] = int(cls)

    # 2. 读取 json 数据
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    # 3. 为每个 json 项目添加其对应 class（用于排序）
    for item in json_data:
        vid = int(item['id'])
        item['class'] = id_to_class.get(vid, float('inf'))  # 默认一个很大值（如缺失）

    # 4. 按 class 排序
    sorted_data = sorted(json_data, key=lambda x: x['class'])

    # 5. 可选：删除临时加的 class 字段
    for item in sorted_data:
        item.pop('class', None)

    # 6. 保存为新 JSON 文件
    with open(output_path, 'w') as f:
        json.dump(sorted_data, f, indent=2)

# 示例使用
txt_path = '/mnt/newdisk/mnt/newdisk/dataset/action/ssv2_all/label/initial/v15_k70_u30/class_sort.txt'
json_path = '/mnt/newdisk/mnt/newdisk/dataset/action/ssv2_all/label/initial/v15/something-something-v2-label_validation_v15.json'
output_path = '/mnt/newdisk/mnt/newdisk/dataset/action/ssv2_all/label/initial/v15_k70_u30/something-something-v2-label_validation_v15.json'

sort_json_by_class(txt_path, json_path, output_path)