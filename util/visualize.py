import matplotlib.pyplot as plt
import numpy as np
import torch
from graphviz import Digraph
from torch.autograd import Variable
from sklearn.manifold import TSNE
from collections import defaultdict
from sklearn.metrics import accuracy_score
# from sklearn.utils.linear_assignment_ import linear_assignment  # sklearn <0.21
# 若是 sklearn >=0.21，请用下面这个
from scipy.optimize import linear_sum_assignment
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
import os


def compute_per_class_cluster_accuracy(true_labels,y_pred, num_classes=101):
    """
    true_labels: np.array of shape [N]
    cluster_labels: np.array of shape [N]
    """
    true_labels = np.array(true_labels)
    cluster_labels = np.array(y_pred)
    class_accs = []
    class_majority = {}

    for cls in range(num_classes):
        idx = np.where(true_labels == cls)[0]
        if len(idx) == 0:
            continue
        cls_true = true_labels[idx]
        cls_pred = cluster_labels[idx]

        # 找出该类中最常见的聚类标签
        unique, counts = np.unique(cls_pred, return_counts=True)
        major_cluster = unique[np.argmax(counts)]
        acc = np.mean(cls_pred == major_cluster)
        class_accs.append(acc)
        class_majority[cls] = (major_cluster, acc)

    mean_acc = np.mean(class_accs)
    print(f"\nAverage Per-Class Clustering Accuracy: {mean_acc * 100:.2f}%")
    return class_accs, class_majority

def hungarian_match(y_true, y_pred, num_classes):
    """
    对预测聚类标签 y_pred 进行最佳对齐，使其与真实标签 y_true 对应。
    返回映射后的 y_pred。
    """
    y_true = np.array(y_true, dtype=int)

    y_pred = np.array(y_pred)

    cost_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    for i in range(len(y_true)):
        cost_matrix[y_true[i], y_pred[i]] += 1

    row_ind, col_ind = linear_sum_assignment(cost_matrix.max() - cost_matrix)

    mapping = {col: row for row, col in zip(row_ind, col_ind)}

    # 重映射预测标签
    aligned_preds = np.array([mapping[p] if p in mapping else -1 for p in y_pred])
    return aligned_preds, mapping



def save_class_accuracy_simple(y_true, majority_map, save_path,filename="class_accuracy.xlsx"):
    """
    将每个类别的准确率保存到Excel文件（简洁版）
    
    参数:
    y_true: 真实标签数组
    majority_map: 包含每个类别准确率的字典 {cls_id: (majority_label, accuracy)}
    filename: Excel文件名
    
    返回:
    accuracy_df: 包含所有类别准确率的DataFrame
    """
    # 获取所有唯一的真实类别标签（按升序排序）
    unique_classes = np.sort(np.unique(y_true))
    
    # 创建存储数据的列表
    accuracy_data = []
    
    # 为每个类别填充准确率
    for cls_id in unique_classes:
        if cls_id in majority_map:
            _, acc = majority_map[cls_id]
        else:
            acc = 0.0  # 如果没有该类别在majority_map中，设为0
        
        accuracy_data.append({
            '真实类别': int(cls_id),  # 类别ID
            '准确率': float(acc)     # 准确率（小数）
        })
    
    # 创建DataFrame并排序
    accuracy_df = pd.DataFrame(accuracy_data)
    accuracy_df = accuracy_df.sort_values('真实类别')
    
    # 保存到Excel文件
    accuracy_df.to_excel(os.path.join(save_path,filename), index=False)
    
    print(f"✓ 已保存到文件: {filename}")
    print(f"✓ 包含 {len(accuracy_df)} 个类别")
    
    return accuracy_df


def make_dot(var, params=None):
    """ Produces Graphviz representation of PyTorch autograd graph
    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    if params is not None:
        assert isinstance(params.values()[0], Variable)
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '('+(', ').join(['%d' % v for v in size])+')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)
    add_nodes(var.grad_fn)
    return dot



def dimensional_reduction_visualization(save_dir, features, labels, camid):

    from tsnecuda import TSNE
    plot_only = features.size(0)
    plot_data = features.cpu()
    plot_label = labels.cpu()
    camid = camid.cpu().numpy()
    print('Çalculating 2D embedding')
    low_dim_embs = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(plot_data.numpy()[:plot_only, :])
    labels = plot_label.numpy()[:plot_only]
    np.save('low_dim_embs.npy', low_dim_embs)
    np.save('labels.npy', labels)
    print('Panting 2D embedding')

    # plot_with_labels(save_dir, low_dim_embs, labels, camid)
    plot_without_labels(save_dir, low_dim_embs, labels, camid)


def dimensional_reduction_visualization_from_numpy(save_dir, features, labels, camid, percentage, with_label=True):

    from tsnecuda import TSNE
    import os
    plot_only = np.random.choice(np.arange(0, np.shape(features)[0]), int(np.shape(features)[0]*percentage), replace=False)
    plot_data = features
    plot_label = labels
    camid = camid
    print('Çalculating 2D embedding')
    low_dim_embs = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(plot_data[plot_only, :])
    labels = plot_label[plot_only]
    np.save(os.path.join(save_dir, 'low_dim_embs.npy'), low_dim_embs)
    np.save(os.path.join(save_dir, 'labels.npy'), labels)
    print('Panting 2D embedding')
    if with_label:
        plot_with_labels(save_dir, low_dim_embs, labels, camid)
    else:
        plot_without_labels(save_dir, low_dim_embs, labels, camid)


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

# def plot_with_new_labels(save_path, features, cluster_labels, true_labels=None, 
#                          use_true_labels=False, highlight_classes=None):
#     """
#     参数说明：
#     - features: numpy array [N, 768]，高维特征
#     - cluster_labels: numpy array [N,]，聚类得到的预测标签
#     - true_labels: numpy array [N,]，真实标签
#     - use_true_labels: 是否用真实标签着色
#     - highlight_classes: 需要标注类别号的类别列表
#     """
#     assert len(features.shape) == 2 and features.shape[1] > 2, "Feature dimension should be high, like 768"
    
#     # Step 1: TSNE降维
#     print("Running t-SNE...")
#     tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto', perplexity=30)
#     reduced_feats = tsne.fit_transform(features)
#     print("t-SNE finished.")

#     # Step 2: 绘图
#     plt.figure(figsize=(12, 10))
#     labels_to_plot = true_labels if (use_true_labels and true_labels is not None) else cluster_labels
#     num_classes = np.unique(labels_to_plot)

#     # 用于存储每个类别的中心点
#     class_centers = {}
#     scatter_handles = []  # 存储图例的句柄
    
#     for class_id in num_classes:
#         idx = labels_to_plot == class_id
#         scatter = plt.scatter(reduced_feats[idx, 0], reduced_feats[idx, 1], 
#                             s=10, alpha=0.6, label=f'Class {class_id}')
        
#         # 计算每个类别的中心点（用于标注）
#         if np.sum(idx) > 0:
#             class_centers[class_id] = np.mean(reduced_feats[idx], axis=0)
    
#     # Step 3: 为指定类别添加文本标注
#     if highlight_classes is not None:
#         highlight_classes = np.array(highlight_classes, dtype=int)
#         for class_id in highlight_classes:
#             if class_id in class_centers:
#                 center_x, center_y = class_centers[class_id]
#                 # 使用一个较大的字体和背景框突出显示
#                 plt.text(center_x, center_y, f'{class_id}', 
#                         fontsize=14, fontweight='bold',
#                         bbox=dict(boxstyle="round,pad=0.3", 
#                                 facecolor='yellow', 
#                                 edgecolor='black',
#                                 alpha=0.8))
    
#     # Step 4: 只显示highlight_classes的图例
#     if scatter_handles:
#         plt.legend(scatter_handles, legend_labels, 
#                   markerscale=2, fontsize=10, 
#                   bbox_to_anchor=(1.05, 1), loc='upper left')
    
#     # plt.title("t-SNE Visualization of Feature Embeddings")
#     plt.axis('off')
#     if len(num_classes) <= 50:  # 如果类别太多，不显示图例
#         plt.legend(markerscale=2, fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300)
#     plt.close()
#     print(f"Saved plot to {save_path}")

def get_vibrant_colors(n_colors):
    """
    获取鲜艳且区分度好的颜色，特别适合30个类别
    """
    # 推荐用于多类别的专业色彩映射
    if n_colors <= 20:
        # 对于≤20类，使用tab20（最专业）
        cmap = plt.cm.get_cmap('tab20')
    elif n_colors <= 40:
        # 对于20-40类，组合tab20和tab20b
        cmap1 = plt.cm.get_cmap('tab20')
        cmap2 = plt.cm.get_cmap('tab20b')
        colors = []
        for i in range(20):
            colors.append(cmap1(i/20))
        for i in range(n_colors-20):
            colors.append(cmap2(i/20))
        return colors[:n_colors]
    else:
        # 超过40类，使用gist_ncar（160种鲜艳颜色）
        cmap = plt.cm.get_cmap('gist_ncar')
    
    return [cmap(i/n_colors) for i in range(n_colors)]

# 生成最高饱和度颜色函数
def generate_max_saturation_colors(n_colors):
    """生成n种最高饱和度的颜色（HSV中S=1.0, V=1.0）"""
    import colorsys
    colors = []
    # 使用黄金分割比例避免颜色聚集
    golden_ratio_conjugate = 0.618033988749895
    hue = 0.0
    
    for i in range(n_colors):
        # 转换为RGB
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)  # 最大饱和度和亮度
        # 转换为十六进制
        hex_color = '#{:02X}{:02X}{:02X}'.format(
            int(rgb[0] * 255),
            int(rgb[1] * 255), 
            int(rgb[2] * 255)
        )
        colors.append(hex_color)
        # 使用黄金分割比例增加色调
        hue += golden_ratio_conjugate
        hue %= 1.0  # 保持在0-1之间
    
    return colors



def plot_with_new_labels(save_path, features, cluster_labels, true_labels=None, 
                         use_true_labels=False, highlight_classes=None):
    """
    参数说明：
    - features: numpy array [N, 768]，高维特征
    - cluster_labels: numpy array [N,]，聚类得到的预测标签
    - true_labels: numpy array [N,]，真实标签
    - use_true_labels: 是否用真实标签着色
    - highlight_classes: 需要对比的类别列表，这些类别之间颜色区分明显
    """
    assert len(features.shape) == 2 and features.shape[1] > 2, "Feature dimension should be high, like 768"
    
    # Step 1: TSNE降维
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto', perplexity=30)
    reduced_feats = tsne.fit_transform(features)
    print("t-SNE finished.")

    # Step 2: 绘图
    plt.figure(figsize=(12, 10))
    labels_to_plot = true_labels if (use_true_labels and true_labels is not None) else cluster_labels
    num_classes = np.unique(labels_to_plot)

    # 用于存储每个类别的中心点
    class_centers = {}
    scatter_handles = []  # 存储图例的句柄
    legend_labels = []    # 存储图例的标签

        # 生成所有类别的高饱和度颜色
    color_list = generate_max_saturation_colors(len(num_classes))

    # 为highlight_classes创建特殊的高对比度颜色
    # highlight_color_map = {}
    # if highlight_classes is not None:
    #     highlight_classes = np.array(highlight_classes, dtype=int)
    #     # 使用最鲜艳的一组纯色作为highlight颜色
    #     highlight_colors = [
    #         '#FF0000', '#00FF00', '#0000FF', '#FF00FF', '#FFFF00', '#00FFFF',
    #         '#FF6600', '#FF0099', '#00FF66', '#9900FF', '#00CCFF', '#FFCC00',
    #         '#FF3300', '#33FF00', '#0033FF', '#FF33FF', '#FFFF33', '#33FFFF',
    #         '#CC00FF', '#00FFCC', '#FFCC00', '#00CCFF', '#FF0066', '#66FF00'
    #     ]
        
    #     # 为每个highlight类别分配颜色
    #     for idx, class_id in enumerate(highlight_classes):
    #         # 优先使用纯色，如果highlight类别太多则用生成的颜色
    #         if idx < len(highlight_colors):
    #             highlight_color_map[class_id] = highlight_colors[idx]
    #         else:
    #             # 从生成的颜色中取，但要确保与普通类别不同
    #             highlight_idx = (idx + 5) % len(color_list)  # 偏移5个位置
    #             highlight_color_map[class_id] = color_list[highlight_idx]


    # 合并为一个循环
    # for i, class_id in enumerate(num_classes):
    #     idx = labels_to_plot == class_id
        
    #     # 确定颜色：如果是highlight类别用特殊颜色，否则用生成的高饱和度颜色
    #     if highlight_classes is not None and class_id in highlight_classes:
    #         color = highlight_color_map[class_id]
    #         s = 35  # highlight类别更大
    #         alpha = 0.95  # 更高透明度
    #         edgecolor = 'black'
    #         linewidth = 1.2
    #         marker = 'o'  # 圆形
    #         zorder = 10  # 确保highlight类别在最上层
    #     else:
    #         color = color_list[i]
    #         s = 35  # 普通类别稍小但仍清晰
    #         alpha = 0.9  # 高透明度
    #         edgecolor = 'None'  # 白色边框增强对比
    #         linewidth = 1.2
    #         marker = 'o'
    #         zorder = 5
        
    #     # 绘制散点（确保当前类别的数据点存在）
    #     if np.sum(idx) > 0:
    #         scatter = plt.scatter(reduced_feats[idx, 0], reduced_feats[idx, 1], 
    #                             s=s, alpha=alpha, color=color, 
    #                             edgecolors=edgecolor, linewidth=linewidth,
    #                             marker=marker, zorder=zorder)
            
    #         # 计算每个类别的中心点
    #         class_centers[class_id] = np.mean(reduced_feats[idx], axis=0)
            
    #         # 如果这个类别在highlight_classes中，则添加到图例
    #         if highlight_classes is not None and class_id in highlight_classes:
    #             # 确保不重复添加同一个类别的图例
    #             # 检查是否已经添加过这个class_id的图例
    #             if f'Class {class_id}' not in legend_labels:
    #                 scatter_handles.append(scatter)
    #                 legend_labels.append(f'Class {class_id}')

    # 1. 统一调色盘：不要再用 generate_max_saturation
    # 建议全量使用 tab20 或类似的学术调色盘
    cmap = plt.get_cmap('tab20')
    all_colors = [cmap(i % 20) for i in range(len(num_classes))]

    for i, class_id in enumerate(num_classes):
        idx = labels_to_plot == class_id
    
        # 普通类别：使用 tab20 的颜色
        color = all_colors[i]
        s = 20         # 显著调小点的大小，这是模仿(a)(b)(c)的关键
        alpha = 1.0   # 增加透明度，让重叠部分呈现出层次感
        edgecolor = 'None'
        zorder = 5
        
        if np.sum(idx) > 0:
            plt.scatter(reduced_feats[idx, 0], reduced_feats[idx, 1], 
                        s=s, alpha=alpha, color=color, 
                        edgecolors=edgecolor, linewidth=0.5,
                        marker='o', zorder=zorder)


    # Step 4: 只显示highlight_classes的图例
    if scatter_handles:
        # 移除重复的图例条目
        unique_handles = []
        unique_labels = []
        seen_labels = set()
        
        for handle, label in zip(scatter_handles, legend_labels):
            if label not in seen_labels:
                seen_labels.add(label)
                unique_handles.append(handle)
                unique_labels.append(label)
        
        plt.legend(unique_handles, unique_labels, 
                markerscale=2, fontsize=10, 
                bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {save_path}")


def visual_gcd(save_path,all_feats, y_pred, y_true, num_classes):
    # 确保输入是整数类型
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    
    # 假设你已经有：targets（真实标签），preds（聚类标签）
    aligned_preds, cluster_to_class_map = hungarian_match(y_true, y_pred, num_classes)

    # 然后计算每类准确率
    accs, majority_map = compute_per_class_cluster_accuracy(y_true, y_pred, num_classes)

    # 保存所有类别的准确率
    accuracy_df = save_class_accuracy_simple(
        y_true=y_true,
        majority_map=majority_map,
        save_path=save_path,
        filename="所有类别准确率.xlsx"
    )

    class_acc_pairs = [(cls_id, acc) for cls_id, (_, acc) in majority_map.items()]

    top10 = sorted(class_acc_pairs, key=lambda x: x[1], reverse=True)[:30]
    
    print("Top-10 Classes by Clustering Accuracy:")
    for rank, (cls_id, acc) in enumerate(top10, start=1):
        print(f"{rank:2d}. Class {cls_id:3d} - Accuracy: {acc * 100:.2f}%")

    # 原定的30个类别
    top10_class_ids = [0,1,84,85,7,21,28,30,31,35,38,67,81,50,34,25,54,100,10,11,99,51,61,46,44,91,18,49,77,43,64,56,45,95]
    
    # 创建top30类别的掩码
    all_top_mask = np.isin(y_true, top10_class_ids)
    
    # 获取所有top30类别的数据
    feats_all_top = all_feats[all_top_mask]
    targets_all_top = y_true[all_top_mask]
    labels_all_top = y_pred[all_top_mask]
    
    print(f"\n所有top30类别样本数量：{feats_all_top.shape[0]}")
    
    # ============ 图1：全部top30类别的图 ============
    plot_with_new_labels(os.path.join(save_path,'back1_all_top30.png'), 
                        feats_all_top, labels_all_top, targets_all_top,
                        use_true_labels=True, highlight_classes=None)

    # ============ 图1：全部类别的图 ============
    all_class_id = sorted(class_acc_pairs, key=lambda x: x[1], reverse=True)
    all_sequence = [i for i in range(num_classes)]
    plot_with_new_labels(os.path.join(save_path,'back1_all.png'), 
                        all_feats,  y_pred,y_true,
                        use_true_labels=True, highlight_classes=None)
    
    # ============ 图4：指定类别标号显示的图 ============
    # 指定要突出显示的类别
    specified_classes = [0,1,84,85]
    
    # 创建指定类别的掩码
    specified_mask = np.isin(y_true, specified_classes)
    
    # 获取指定类别的数据
    feats_specified = all_feats[specified_mask]
    targets_specified = y_true[specified_mask]
    labels_specified = y_pred[specified_mask]
    
    print(f"\n指定突出显示的类别: {specified_classes}")
    print(f"指定类别样本数量：{feats_specified.shape[0]}")
    
    # 这个图只显示指定类别，并标注它们的标签号
    plot_with_new_labels(os.path.join(save_path,'back1_specified.png'), 
                        feats_specified, labels_specified, targets_specified,
                        use_true_labels=True, highlight_classes=specified_classes)
    
    # ============ 额外：你也可以生成一个包含所有top30类别但只标注指定类别的图 ============
    # 这个图显示所有top30类别，但只标注指定的几个类别
    plot_with_new_labels(os.path.join(save_path,'back1_all_highlight_specified.png'), 
                        feats_all_top, labels_all_top, targets_all_top,
                        use_true_labels=True, highlight_classes=specified_classes)
    
    print("\n" + "="*50)
    print("所有可视化图已保存完成：")
    print("1. back1_all_top30.png - 所有top30类别")
    print("2. back1_even.png - 偶数类别")
    print("3. back1_odd.png - 奇数类别")
    print("4. back1_specified.png - 指定类别（只显示指定类别）")
    print("5. back1_all_highlight_specified.png - 所有top30类别（只标注指定类别）")
    print("="*50)
    
    return top10, accuracy_df


def plot_with_labels(save_path, lowDWeights, labels, camid):
    import matplotlib.pyplot as plt
    padding_rate = 0.1
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    unique = np.unique(labels)
    num_type_color = len(unique)
    cmap = get_cmap(num_type_color)
    label2color_dict = {l: cmap(i) for i, l in enumerate(unique)}

    for x, y, s, cam in zip(X, Y, labels, camid):
        if  cam == 9:
            plt.text(x, y, s, fontsize=3,
                     ha='center', va='center',  # 水平居中,垂直居中
                     bbox=dict(boxstyle='circle',  # 圆圈
                               ec=label2color_dict[s],  # 边框颜色
                               fc=label2color_dict[s]  # 填充颜色
                               ))
        elif cam == 8:
            plt.text(x, y, s, fontsize=3,
                     ha='center', va='center',  # 水平居中,垂直居中
                     bbox=dict(boxstyle='square',  # 圆圈
                               ec=label2color_dict[s],  # 边框颜色
                               fc=label2color_dict[s]  # 填充颜色
                               ))
        elif cam == 7:
            plt.text(x, y, s, fontsize=5,
                     ha='center', va='center',  # 水平居中,垂直居中
                     bbox=dict(boxstyle='circle',  # 圆圈
                               ec=label2color_dict[s],  # 边框颜色
                               fc=label2color_dict[s]  # 填充颜色
                               ))
        elif cam == 3 or cam == 6:
            plt.text(x, y, s, fontsize=1,
                     ha='center', va='center', #水平居中,垂直居中
                     bbox=dict(boxstyle='circle', #圆圈
                               ec=label2color_dict[s],#边框颜色
                               fc=label2color_dict[s]#填充颜色
                               ))
        elif cam == 0:
            plt.text(x, y, s, fontsize=1,
                     ha='center', va='center',  # 水平居中,垂直居中
                     bbox=dict(boxstyle='circle',  # 圆圈
                               ec=label2color_dict[s],  # 边框颜色
                               fc=label2color_dict[s]  # 填充颜色
                               ))
        else:
            plt.text(x, y, s, fontsize=1,
                     ha='center', va='center',  # 水平居中,垂直居中
                     bbox=dict(boxstyle='square',  #
                               ec=label2color_dict[s],  # 边框颜色
                               fc=label2color_dict[s]  # 填充颜色
                               ))
    # plt.xlim(X.min()-padding_rate*X.min(), X.max()+padding_rate*X.max())
    # plt.ylim(Y.min()-padding_rate*Y.min(), Y.max()+padding_rate*Y.max())
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.axis('off')

    plt.title('Visualize Embedding')
    save_path = save_path + '.png'
    plt.savefig(save_path, dpi=800)
    plt.show()

    plt.pause(0.01)



def plot_without_labels(save_path, lowDWeights, labels, camid):
    import matplotlib.pyplot as plt
    padding_rate = 0.1
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    unique = np.unique(labels)
    num_type_color = len(unique)
    cmap = get_cmap(num_type_color)
    label2color_dict = {l: cmap(i) for i, l in enumerate(unique)}

    for x, y, s, cam in zip(X, Y, labels, camid):
        if  cam == 9:
            plt.text(x, y, ' ', fontsize=3,
                     ha='center', va='center',  # 水平居中,垂直居中
                     bbox=dict(boxstyle='circle',  # 圆圈
                               ec=label2color_dict[s],  # 边框颜色
                               fc=label2color_dict[s]  # 填充颜色
                               ))
        elif cam == 8:
            plt.text(x, y, ' ', fontsize=3,
                     ha='center', va='center',  # 水平居中,垂直居中
                     bbox=dict(boxstyle='square',  # 圆圈
                               ec=label2color_dict[s],  # 边框颜色
                               fc=label2color_dict[s]  # 填充颜色
                               ))
        elif cam == 7:
            plt.text(x, y, ' ', fontsize=5,
                     ha='center', va='center',  # 水平居中,垂直居中
                     bbox=dict(boxstyle='circle',  # 圆圈
                               ec=label2color_dict[s],  # 边框颜色
                               fc=label2color_dict[s]  # 填充颜色
                               ))
        elif cam == 3 or cam == 6:
            plt.text(x, y, ' ', fontsize=1,
                     ha='center', va='center', #水平居中,垂直居中
                     bbox=dict(boxstyle='circle', #圆圈
                               ec=label2color_dict[s],#边框颜色
                               fc=label2color_dict[s]#填充颜色
                               ))
        elif cam == 0:
            plt.text(x, y, ' ', fontsize=1,
                     ha='center', va='center',  # 水平居中,垂直居中
                     bbox=dict(boxstyle='circle',  # 圆圈
                               ec=label2color_dict[s],  # 边框颜色
                               fc=label2color_dict[s]  # 填充颜色
                               ))
        else:
            plt.text(x, y, ' ', fontsize=1,
                     ha='center', va='center',  # 水平居中,垂直居中
                     bbox=dict(boxstyle='square',  #
                               ec=label2color_dict[s],  # 边框颜色
                               fc=label2color_dict[s]  # 填充颜色
                               ))
    plt.xlim(X.min()-padding_rate*X.min(), X.max()+padding_rate*X.max())
    plt.ylim(Y.min()-padding_rate*Y.min(), Y.max()+padding_rate*Y.max())
    # plt.xlim(X.min(), X.max())
    # plt.ylim(Y.min(), Y.max())
    plt.axis('off')

    plt.title('Visualize Embedding')
    save_path = save_path + '.png'
    plt.savefig(save_path, dpi=800)
    plt.show()


    plt.pause(0.01)



num_samples = 3000
lowDWeights = np.random.randn(num_samples, 2)  # shape: [100, 2]

# 模拟每个样本的类别标签（比如有 5 类）
labels = np.random.choice([0, 1, 2, 3, 4], size=num_samples)

# 模拟每个样本的摄像头编号（比如从摄像头 0 到 9）
camid = np.random.choice([0, 3, 6, 7, 8, 9], size=num_samples)

# 保存路径（图像将保存为 .png 文件）
save_path = 'embedding_visualization'

# 调用你的绘图函数
plot_with_labels(save_path, lowDWeights, labels, camid)
print("success !")


# # 假设有 30000 个样本，每个样本特征为 512 维
# features = np.random.randn(30000, 512)              # shape: [30000, 512]
# labels = np.random.randint(0, 101, 30000)            # shape: [30000]
# camid = np.random.choice([0, 3, 6, 7, 8, 9], 30000)  # 摄像头编号

# # 保存目录（你可以换成你自己的路径）
# save_dir = './tsne_vis'

# # 调用函数，只画 10% 的样本，带标签
# dimensional_reduction_visualization_from_numpy(
#     save_dir=save_dir,
#     features=features,
#     labels=labels,
#     camid=camid,
#     percentage=0.1,
#     with_label=True
# )

# print("success !")