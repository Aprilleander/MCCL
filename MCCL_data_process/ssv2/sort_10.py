import json

def sort_videos_by_class_id(input_path, output_path):
    # 读取已生成的JSON文件
    with open(input_path, 'r', encoding='utf-8') as f:
        videos = json.load(f)
    
    # 排序算法（稳定排序，保持相同class_id的原始顺序）
    sorted_videos = sorted(videos, key=lambda x: (
        int(x['class_id']) if isinstance(x.get('class_id'), (int, str)) else float('inf'),
        videos.index(x)  # 保留原始顺序作为次级排序条件
    ))
    
    # 保存排序结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sorted_videos, f, 
                 ensure_ascii=False, 
                 indent=2,
                 sort_keys=False)

# 使用示例

json_path = '/mnt/newdisk/mnt/newdisk/dataset/action/ssv2_all/label/initial/v15_k70_u30/something-something-v2-train_v30_class.json'
output_path = '/mnt/newdisk/mnt/newdisk/dataset/action/ssv2_all/label/initial/v15_k70_u30/something-something-v2-train_v30_sort.json'

sort_videos_by_class_id(
    json_path,  # 已生成的原始JSON文件
    output_path     # 排序后的输出文件
)