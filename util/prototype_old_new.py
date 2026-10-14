import pandas as pd
import numpy as np
from collections import defaultdict

import pandas as pd
import numpy as np


def analyze_prediction_by_true_class(targets, pre_logits, output_file='prediction_distribution.xlsx'):
    """
    以真实标签为基准，统计每个真实类别对应的预测标签分布
    
    参数:
    - targets: 真实标签数组 [n]
    - pre_logits: 预测标签数组 [n]
    - output_file: 输出Excel文件名
    
    返回:
    - distribution_df: 包含预测分布统计的DataFrame
    """
    
    # 将输入转换为numpy数组（如果是PyTorch张量）
    if hasattr(targets, 'cpu'):
        targets = targets.cpu().numpy()
    if hasattr(pre_logits, 'cpu'):
        pre_logits = pre_logits.cpu().numpy()
    
    # 获取所有真实类别
    true_classes = np.unique(targets)
    true_classes = sorted(true_classes)
    
    # 存储结果
    results = []
    
    for true_class in true_classes:
        # 找到真实标签为当前类别的样本索引
        true_class_indices = np.where(targets == true_class)[0]
        true_class_total = len(true_class_indices)
        
        if true_class_total == 0:
            continue
        
        # 这些样本的预测标签
        predictions_for_true_class = pre_logits[true_class_indices]
        
        # 统计预测标签的分布
        unique_predictions, prediction_counts = np.unique(predictions_for_true_class, return_counts=True)
        prediction_percentages = (prediction_counts / true_class_total) * 100
        
        # 为每个预测标签添加结果
        for pred_label, count, percentage in zip(unique_predictions, prediction_counts, prediction_percentages):
            results.append({
                '真实标签': true_class,
                '预测标签': pred_label,
                '数量': count,
                '百分比': percentage
            })
    
    # 创建DataFrame
    distribution_df = pd.DataFrame(results)
    
    # 按照真实标签和预测标签排序
    distribution_df = distribution_df.sort_values(['真实标签', '预测标签'])
    
    # 保存到Excel
    distribution_df.to_excel(output_file, index=False)
    
    print(f"预测分布统计已保存到 {output_file}")



def analyze_pseudo_label_distribution(targets, pse_label, output_file='pseudo_label_simple.xlsx'):
    """
    简化版伪标签与真实标签对应关系分析
    
    参数:
    - targets: 真实标签 [n]
    - pse_label: 伪标签 [n]
    - output_file: 输出Excel文件名
    """
    
    # 将输入转换为numpy数组（如果是PyTorch张量）
    if hasattr(targets, 'cpu'):
        targets = targets.cpu().numpy()
    if hasattr(pse_label, 'cpu'):
        pse_label = pse_label.cpu().numpy()
    
    # 获取所有唯一的伪标签
    unique_pseudo_labels = np.unique(pse_label)
    
    # 创建结果数据
    result_data = []
    
    for pseudo_label in unique_pseudo_labels:
        # 获取该伪标签对应的所有样本
        pseudo_indices = np.where(pse_label == pseudo_label)[0]
        pseudo_total = len(pseudo_indices)
        
        if pseudo_total == 0:
            continue
            
        # 获取对应的真实标签
        corresponding_targets = targets[pseudo_indices]
        
        # 统计真实标签的分布
        unique_true, true_counts = np.unique(corresponding_targets, return_counts=True)
        true_percentages = (true_counts / pseudo_total) * 100
        
        for true_label, count, percentage in zip(unique_true, true_counts, true_percentages):
            result_data.append({
                '伪标签': pseudo_label,
                '真实标签': true_label,
                '数量': count,
                '百分比': percentage / 100  # 转换为小数形式，如0.973333333
            })
    
    # 创建DataFrame
    df = pd.DataFrame(result_data)
    
    # 按照伪标签和真实标签排序
    df = df.sort_values(['伪标签', '真实标签'])
    
    # 保存到Excel
    df.to_excel(output_file, index=False)
    
    print(f"伪标签分析完成，结果已保存到 {output_file}")
    print(f"共统计了 {len(df)} 条对应关系")
    


def analyze_predictions_detailed(targets, mask_old_new, pse_label, predicted_similar, output_file='detailed_analysis_results.xlsx'):
    """
    详细分析预测结果的统计信息
    
    参数:
    - targets: 真实标签 [9935]
    - mask_old_new: 预测是否属于已知类 [9935], True表示预测为已知类
    - pse_label: 伪标签 [9935]
    - predicted_similar: 预测置信度 [9935]
    - output_file: 输出Excel文件名
    """

    if hasattr(targets, 'cpu'):
        targets = targets.cpu().numpy()
    if hasattr(mask_old_new, 'cpu'):
        mask_old_new = mask_old_new.cpu().numpy()
    if hasattr(pse_label, 'cpu'):
        pse_label = pse_label.cpu().numpy()
    if hasattr(predicted_similar, 'cpu'):
        predicted_similar = predicted_similar.cpu().numpy()
    
    
    # 生成mask：偶数为True（已知类），奇数为False（未知类）
    mask = (targets % 2 == 0)
    
    # 分析已知类（偶数标签）
    known_class_results = analyze_known_classes_detailed(targets, mask, mask_old_new, pse_label, predicted_similar)
    
    # 分析未知类（奇数标签）
    unknown_class_results = analyze_unknown_classes_detailed(targets, mask, mask_old_new, pse_label, predicted_similar)
    
    # 创建Excel写入器
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # 保存已知类结果
        known_class_results['prediction_details'].to_excel(writer, sheet_name='已知类预测分布', index=False)
        unknown_class_results['prediction_details'].to_excel(writer, sheet_name='未知类预测分布', index=False)
        
        # 保存总体统计
        overall_stats = create_overall_stats_detailed(targets, mask, mask_old_new, pse_label, predicted_similar)
        overall_stats.to_excel(writer, sheet_name='总体统计', index=False)
    
    print(f"详细分析完成，结果已保存到 {output_file}")
    return known_class_results, unknown_class_results


import numpy as np
import pandas as pd

def calculate_and_save_class_accuracy(targets, pre_logits, output_file='class_accuracy.xlsx'):
    """
    计算每个类别的预测准确率并保存到Excel
    
    参数:
    - targets: 真实标签数组 [n]
    - pre_logits: 预测标签数组 [n]
    - output_file: 输出Excel文件名
    
    返回:
    - accuracy_df: 包含准确率统计的DataFrame
    """
    
    # 将输入转换为numpy数组（如果是PyTorch张量）
    if hasattr(targets, 'cpu'):
        targets = targets.cpu().numpy()
    if hasattr(pre_logits, 'cpu'):
        pre_logits = pre_logits.cpu().numpy()
    
    # 获取所有类别
    all_classes = np.unique(np.concatenate([targets, pre_logits]))
    all_classes = sorted(all_classes)
    
    # 存储结果
    results = []
    
    for class_label in all_classes:
        # 找到预测为当前类别的样本
        predicted_as_class = (pre_logits == class_label)
        
        # 计算这些样本中真实标签也为当前类别的数量
        correct_predictions = np.sum((targets == class_label) & predicted_as_class)
        
        # 计算预测为当前类别的总样本数
        total_predicted_as_class = np.sum(predicted_as_class)
        
        # 计算准确率
        if total_predicted_as_class > 0:
            accuracy = correct_predictions / total_predicted_as_class
        else:
            accuracy = 0.0
        
        # 添加到结果列表
        results.append({
            '类别': class_label,
            '预测正确数量': correct_predictions,
            '预测为该类别的总数': total_predicted_as_class,
            '准确率': accuracy
        })
    
    # 创建DataFrame
    accuracy_df = pd.DataFrame(results)
    
    # 保存到Excel
    accuracy_df.to_excel(output_file, index=False)
    
    print(f"类别准确率统计已保存到 {output_file}")
    

def analyze_known_classes_detailed(targets, mask, mask_old_new, pse_label, predicted_similar):
    """详细分析已知类（偶数标签）"""
    
    # 获取已知类样本
    known_class_indices = np.where(mask)[0]
    known_class_targets = targets[known_class_indices]
    known_class_mask_old_new = mask_old_new[known_class_indices]
    known_class_pse_label = pse_label[known_class_indices]
    known_class_similar = predicted_similar[known_class_indices]
    
    # 获取所有已知类别
    known_classes = np.unique(known_class_targets)
    known_classes = sorted(known_classes)
    
    summary_data = []
    prediction_details_data = []
    
    for class_label in known_classes:
        # 获取当前类别的所有样本索引
        class_indices = np.where(known_class_targets == class_label)[0]
        class_total = len(class_indices)
        
        if class_total == 0:
            continue
            
        # 当前类别的数据
        class_mask_old_new = known_class_mask_old_new[class_indices]
        mask = known_class_targets[class_indices]
        class_pse_label = known_class_pse_label[class_indices]
        class_similar = known_class_similar[class_indices]
        
        # 1. 统计mask_old_new预测情况
        # 预测为已知类
        predicted_known = np.sum(class_mask_old_new)
        predicted_known_ratio = predicted_known / class_total
        
         # 2. 统计伪标签预测情况
        unique_predictions, prediction_counts = np.unique(class_pse_label, return_counts=True)
        prediction_ratios = prediction_counts / class_total
        
        # 3. 统计置信度
        if len(class_similar) > 0:
            overall_avg_conf = np.mean(class_similar)
            overall_max_conf = np.max(class_similar)
            overall_min_conf = np.min(class_similar)
        else:
            overall_avg_conf = overall_max_conf = overall_min_conf = 0
        

        # 5. 按预测标签统计置信度
        for pred_label, pred_count, pred_ratio in zip(unique_predictions, prediction_counts, prediction_ratios):
            pred_indices = np.where(class_pse_label == pred_label)[0]
            pred_confidences = class_similar[pred_indices]
            
            prediction_details_data.append({
                '真实标签': class_label,
                '预测标签': pred_label,
                '预测数量': pred_count,
                '预测占比': pred_ratio,
                'old_new': predicted_known_ratio,
                '平均置信度': np.mean(pred_confidences),
            })
        
    return {
        'prediction_details': pd.DataFrame(prediction_details_data)
    }


def analyze_unknown_classes_detailed(targets, mask, mask_old_new, pse_label, predicted_similar):
    """详细分析未知类（奇数标签）"""
    
    # 获取未知类样本
    unknown_class_indices = np.where(~mask)[0]
    unknown_class_targets = targets[unknown_class_indices]
    unknown_class_mask_old_new = mask_old_new[unknown_class_indices]
    unknown_class_pse_label = pse_label[unknown_class_indices]
    unknown_class_similar = predicted_similar[unknown_class_indices]
    
    # 获取所有未知类别
    unknown_classes = np.unique(unknown_class_targets)
    unknown_classes = sorted(unknown_classes)
    
    summary_data = []
    prediction_details_data = []
    
    for class_label in unknown_classes:
        # 获取当前类别的所有样本索引
        class_indices = np.where(unknown_class_targets == class_label)[0]
        class_total = len(class_indices)
        
        if class_total == 0:
            continue
            
        # 当前类别的数据
        class_mask_old_new = unknown_class_mask_old_new[class_indices]
        class_pse_label = unknown_class_pse_label[class_indices]
        class_similar = unknown_class_similar[class_indices]
        
        # 1. 统计mask_old_new预测情况
        # 预测为已知类
        predicted_known = np.sum(class_mask_old_new)
        predicted_known_ratio = predicted_known / class_total
        

        # 2. 统计伪标签预测情况
        unique_predictions, prediction_counts = np.unique(class_pse_label, return_counts=True)
        prediction_ratios = prediction_counts / class_total
        
        # 3. 统计置信度
        if len(class_similar) > 0:
            overall_avg_conf = np.mean(class_similar)
            overall_max_conf = np.max(class_similar)
            overall_min_conf = np.min(class_similar)
        else:
            overall_avg_conf = overall_max_conf = overall_min_conf = 0
        
        # 4. 按预测标签统计置信度
        for pred_label, pred_count, pred_ratio in zip(unique_predictions, prediction_counts, prediction_ratios):
            pred_indices = np.where(class_pse_label == pred_label)[0]
            pred_confidences = class_similar[pred_indices]
            
            prediction_details_data.append({
                '真实标签': class_label,
                '预测标签': pred_label,
                '预测数量': pred_count,
                '预测占比': pred_ratio,
                'old_new': predicted_known_ratio,
                '平均置信度': np.mean(pred_confidences),
            })
        
    return {
        'prediction_details': pd.DataFrame(prediction_details_data)
    }

def create_overall_stats_detailed(targets, mask, mask_old_new, pse_label, predicted_similar):
    """创建详细总体统计信息"""
    
    total_samples = len(targets)
    known_samples = np.sum(mask)
    unknown_samples = total_samples - known_samples
    
    # 已知类总体统计
    known_indices = np.where(mask)[0]
    known_mask_old_new = mask_old_new[known_indices]
    known_pse_label = pse_label[known_indices]
    known_targets = targets[known_indices]
    known_similar = predicted_similar[known_indices]
    
    known_correct = np.sum(known_mask_old_new & (known_pse_label == known_targets))
    known_wrong_class = np.sum(known_mask_old_new & (known_pse_label != known_targets))
    known_wrong_unknown = np.sum(~known_mask_old_new)
    
    # 未知类总体统计
    unknown_indices = np.where(~mask)[0]
    unknown_mask_old_new = mask_old_new[unknown_indices]
    unknown_pse_label = pse_label[unknown_indices]
    unknown_targets = targets[unknown_indices]
    unknown_similar = predicted_similar[unknown_indices]
    
    unknown_correct = np.sum(~unknown_mask_old_new)
    unknown_wrong = np.sum(unknown_mask_old_new)
    
    # 置信度统计
    known_avg_conf = np.mean(known_similar) if len(known_similar) > 0 else 0
    unknown_avg_conf = np.mean(unknown_similar) if len(unknown_similar) > 0 else 0
    
    overall_data = [{
        '统计类型': '总体',
        '总样本数': total_samples,
        '已知类样本数': known_samples,
        '未知类样本数': unknown_samples,
        
        # 已知类统计
        '已知类正确预测数量': known_correct,
        '已知类正确预测比例': known_correct / known_samples if known_samples > 0 else 0,
        '已知类错误分类数量': known_wrong_class,
        '已知类错误分类比例': known_wrong_class / known_samples if known_samples > 0 else 0,
        '已知类预测为未知类数量': known_wrong_unknown,
        '已知类预测为未知类比例': known_wrong_unknown / known_samples if known_samples > 0 else 0,
        '已知类平均置信度': known_avg_conf,
        
        # 未知类统计
        '未知类正确预测数量': unknown_correct,
        '未知类正确预测比例': unknown_correct / unknown_samples if unknown_samples > 0 else 0,
        '未知类错误预测数量': unknown_wrong,
        '未知类错误预测比例': unknown_wrong / unknown_samples if unknown_samples > 0 else 0,
        '未知类平均置信度': unknown_avg_conf,
        
        # 总体准确率
        '总体准确率': (known_correct + unknown_correct) / total_samples if total_samples > 0 else 0
    }]
    
    return pd.DataFrame(overall_data)

# 使用示例
if __name__ == "__main__":
    # 示例数据（请替换为您的实际数据）
    n_samples = 9935
    targets = np.random.randint(0, 20, n_samples)  # 0-19的标签
    mask_old_new = np.random.choice([True, False], n_samples, p=[0.7, 0.3])
    pse_label = np.random.randint(0, 20, n_samples)
    predicted_similar = np.random.rand(n_samples)
    
    # 运行详细分析
    known_results, unknown_results = analyze_predictions_detailed(
        targets, mask_old_new, pse_label, predicted_similar, 'detailed_analysis_results.xlsx'
    )