import json

def enrich_videos_with_class_id(json_path, txt_path, output_path):
    # 读取并预处理原始JSON数据
    with open(json_path, 'r', encoding='utf-8') as f:
        videos = json.load(f)
    
    # 转换视频ID为整数类型
    for video in videos:
        video['id'] = int(video['id'])

    # 构建视频ID到类别ID的映射字典
    id_mapping = {}
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            try:
                video_id_str, class_id_str = line.split(maxsplit=1)
                id_mapping[int(video_id_str)] = int(class_id_str)
            except ValueError as e:
                print(f"第 {line_num} 行格式错误: {line} ({str(e)})")

    # 创建增强后的视频数据
    enriched_videos = []
    for video in videos:
        video_id = video['id']
        if video_id in id_mapping:
            # 创建新字典保持原始数据不变
            new_video = video.copy()
            new_video['class_id'] = id_mapping[video_id]
            enriched_videos.append(new_video)
        else:
            print(f"警告: 未找到视频ID {video_id} 对应的类别信息")

    # 保持原有顺序写入新文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(enriched_videos, f, 
                 ensure_ascii=False, 
                 indent=2,
                 sort_keys=False)  # 保持原始字段顺序

# 使用示例

txt_path = '/mnt/newdisk/mnt/newdisk/dataset/action/ssv2_all/label/initial/v15_k70_u30/class_sort.txt'
json_path = '/mnt/newdisk/mnt/newdisk/dataset/action/ssv2_all/label/initial/v15_k70_u30/something-something-v2-train_v30.json'
output_path = '/mnt/newdisk/mnt/newdisk/dataset/action/ssv2_all/label/initial/v15_k70_u30/something-something-v2-train_v30_class.json'

enrich_videos_with_class_id(
    json_path, 
    txt_path,
    output_path 
)