import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import pandas as pd

def visualize_class_distribution(pred, label):
    if len(pred) != len(label):
        raise ValueError("The length of 'pred' and 'label' must be the same.")

    # 获取预测类的所有唯一值
    unique_pred_classes = set(pred)

    # 统计每个预测类对应的真实标签分布
    class_label_distribution = {}
    for cls in unique_pred_classes:
        indices = [i for i, p in enumerate(pred) if p == cls]
        corresponding_labels = [label[i] for i in indices]
        label_count = Counter(corresponding_labels)
        class_label_distribution[cls] = label_count

    # 打印统计结果
    for cls, label_count in class_label_distribution.items():
        print(f"Predicted Class {cls}:")
        for lbl, count in label_count.items():
            print(f"  True Label {lbl}: {count} samples")

    # 准备数据进行热力图可视化
    pred_classes = sorted(class_label_distribution.keys())
    all_labels = sorted(set(label))
    heatmap_data = np.zeros((len(pred_classes), len(all_labels)))

    for i, cls in enumerate(pred_classes):
        for j, lbl in enumerate(all_labels):
            heatmap_data[i, j] = class_label_distribution[cls].get(lbl, 0)

    # 画热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt="g", cmap="viridis",
                xticklabels=all_labels, yticklabels=pred_classes)
    plt.title("True Label Distribution for Each Predicted Class")
    plt.xlabel("True Label")
    plt.ylabel("Predicted Class")
    plt.savefig("./heatmap.png",dpi=300,bbox_inches='tight')
    #plt.show()
    print("save images")



def generate_label_pred_excel(pred, label):
    output_file = './pred.xlsx'

    if len(pred) != len(label):
        raise ValueError("The length of 'pred' and 'label' must be the same.")

    # 获取 label 中的所有唯一值
    unique_labels = set(label)

    # 统计每个 label 中的索引在 pred 中的值分布
    data = []  # 用于存储统计结果
    for lbl in unique_labels:
        indices = [i for i, l in enumerate(label) if l == lbl]
        corresponding_preds = [pred[i] for i in indices]
        pred_counts = Counter(corresponding_preds)
        
        # 将统计结果转换为表格格式
        for pred_value, count in pred_counts.items():
            data.append({
                "Label": lbl,
                "Predicted Value": pred_value,
                "Count": count
            })

    # 转换为 DataFrame 并保存为 Excel 文件
    df = pd.DataFrame(data)
    #df.sort_values(by=["Label", "Predicted Value"], inplace=True)
    df.to_excel(output_file, index=False)
    print(f"Excel file saved to {output_file}")
