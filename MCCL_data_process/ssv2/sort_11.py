import json
import random
from typing import Dict, Any, List  # 注意导入方式

def categorize_videos(
    input_path: str,
    train_path: str,
    val_path: str,
    unlabel_path: str,
    seed: int = 42
):
    random.seed(seed)
    
    train_set = []
    val_set = []
    unlabel_set = []
    errors = []

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"无效的JSON格式: {str(e)}")

    for idx, item in enumerate(data):
        if random.random() < 0.5:
            try:
                class_id = int(item['class_id'])
                
                if class_id > 121:
                    unlabel_set.append(item)
                else:
                    if random.random() < 0.5:
                        if random.random() < 0.1:
                            val_set.append(item)
                        else:
                            train_set.append(item)
                    else:
                        unlabel_set.append(item)
                        
            except (KeyError, ValueError) as e:
                errors.append({
                    'index': idx,
                    'error': str(e),
                    'raw_data': item
                })
                continue
        else:
            pass

    _save_json(train_set, train_path)
    _save_json(val_set, val_path)
    _save_json(unlabel_set, unlabel_path)
    
    if errors:
        print(f"发现 {len(errors)} 条错误数据，详情见 errors.log")
        with open('errors.log', 'w', encoding='utf-8') as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)

def _save_json(data: List[Dict[str, Any]], path: str):  # 注意类型注解写法
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
            sort_keys=False
        )

# 使用示例
all_path = '/mnt/newdisk/mnt/newdisk/dataset/action/ssv2_all/label/initial/v15_k70_u30/something-something-v2-train_v30_sort.json'
label_path = '/mnt/newdisk/mnt/newdisk/dataset/action/ssv2_all/label/initial/v15_k70_u30/something-something-v2-label_train_v15.json'
val_path = '/mnt/newdisk/mnt/newdisk/dataset/action/ssv2_all/label/initial/v15_k70_u30/something-something-v2-label_val_v15.json'
unlabel_path = '/mnt/newdisk/mnt/newdisk/dataset/action/ssv2_all/label/initial/v15_k70_u30/something-something-v2-unlabel_train_v15.json'

if __name__ == "__main__":
    categorize_videos(
        input_path=all_path,
        train_path=label_path,
        val_path=val_path,
        unlabel_path=unlabel_path
    )