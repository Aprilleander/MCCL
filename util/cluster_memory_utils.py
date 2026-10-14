import collections
import numpy as np
from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd
from util.general_utils import AverageMeter
from sklearn.cluster import KMeans
import math
from methods.clustering.faster_mix_k_means_pytorch import K_Means as SemiSupKMeans
from util.cluster_and_log_utils import log_accs_from_preds,split_cluster_acc_v2_kmeans,analyze_cluster_target_relation,save_to_excel
import torch.distributed as dist
import os 
import tqdm
import random 
from model import KL_loss,top1_sharpness_loss,max_margin_loss
from torch.autograd import Function
from typing import Dict, Tuple, List
from collections import defaultdict
import networkx as nx
from collections import Counter
from util.visual import visualize_prototype_distances,visualize_all_prototypes

from util.prototype_old_new import analyze_predictions_detailed,analyze_pseudo_label_distribution,analyze_prediction_by_true_class
from typing import List, Tuple


def find_u_prototype(memory,clusters_list,u_feats):
    known_prototypes,known_labels,_= memory.get_known_class_prototypes()

    _,cluster_size,_,_ = process_clusters(clusters_list)

    clu_list,cluster_prototype = filter_clusters_with_memory(clusters_list, memory)

    cluster_prototypes = compute_cluster_similarities(clu_list, memory, u_feats)

    purified_list = purify_cluster_list(clu_list, cluster_prototypes[1],memory,min_cluster_size=25,absolute_min_size=15,buffer_ratio=0.1)

    pseudo_labels, similarities, all_prototypes, all_prototype_labels = assign_pseudo_labels(purified_list[0], u_feats, memory)

    return pseudo_labels,purified_list,all_prototypes,all_prototype_labels


def filter_exclusive_samples(similar, strong_indices,target_class, exclusivity_ratio=2.0, min_target_similarity=0.6):
    """
    筛选专属于目标类别的样本
    
    Args:
        similarities: 相似度矩阵 [n, 51]
        target_class: 目标类别索引 (0-50)
        exclusivity_ratio: 专一性比例，目标相似度/最大其他相似度的最小比值
        min_target_similarity: 目标类别最小相似度
    
    Returns:
        exclusive_indices: 专属样本的索引
        ambiguous_indices: 模糊样本的索引
        analysis_info: 分析信息
    """

    similarities = similar[strong_indices]
    n_samples = similarities.shape[0]
    
    # 获取目标类别相似度
    target_sims = similarities[:, target_class]
    
    # 获取其他类别的最大相似度（排除目标类别）
    other_classes = [i for i in range(similarities.shape[1]) if i != target_class]
    other_sims_sorted = similarities[:, other_classes].sort(dim=1, descending=True)[0][:, :2]
    gap_sum = (target_sims.unsqueeze(1) - other_sims_sorted).sum(dim=1)

    norm_gapsum = (gap_sum-gap_sum.min()) / (gap_sum.max()-gap_sum.min())

    # 计算专一性比例
    # exclusivity_ratios = target_sims / (other_max_sims + 1e-8)

    # 目标类别相似度占所有相似度总和的比例
    total_similarities = similarities.sum(dim=1)
    relative_advantages = target_sims / (total_similarities + 1e-8)
    norm_relative = (relative_advantages-relative_advantages.min())/(relative_advantages.max()-relative_advantages.min())

    total_score = norm_gapsum +norm_relative
    
    mask_9935 = torch.zeros(similar.shape[0], dtype=torch.bool)  # 长度为 9935

    top40_indices_in_filtered = torch.topk(total_score, k=40, largest=True).indices

    indexes = []
    for idx in top40_indices_in_filtered:
        index = strong_indices[idx]
        indexes.append(index)

    return indexes


def evaluate_strong_true_accuracy(strong_true_indices, u_targets):
    """
    评估强真值样本的标签一致性
    
    Args:
        strong_true_indices: 强真值样本的索引列表
        u_targets: 所有样本的真实标签 [n]
    
    Returns:
        main_label_ratio: 主要标签的比例
        main_label: 出现次数最多的标签
        label_distribution: 所有标签的分布
        total_count: 强真值样本总数
    """
    if len(strong_true_indices) == 0:
        return 0.0, None, {}, 0

    if isinstance(strong_true_indices, list):
        strong_true_indices = torch.tensor(strong_true_indices, device=u_targets.device)
    
    # 获取强真值样本的真实标签
    strong_true_labels = u_targets[strong_true_indices]
    
    # 计算标签分布
    label_counts = Counter(strong_true_labels.tolist())
    total_count = len(strong_true_labels)
    
    # 找到出现次数最多的标签
    main_label, main_count = label_counts.most_common(1)[0]
    main_label_ratio = main_count / total_count
    
    # 转换为字典格式
    label_distribution = dict(label_counts)
    
    return main_label_ratio,main_label

def evaluate_cluster_purity(clusters, true_labels):
    """
    评估簇的纯度
    
    Args:
        clusters: 簇列表，每个元素为(prototype, 样本索引列表)
        true_labels: 所有样本的真实标签 [9935]
    
    Returns:
        avg_purity: 平均纯度
        cluster_purities: 每个簇的纯度信息
        overall_stats: 整体统计信息
    """
    if len(clusters) == 0:
        return 0.0, [], {}
    
    cluster_purities = []
    total_samples = 0
    total_correct = 0
    
    for i, (prototype, cluster_indices) in enumerate(clusters):
        if len(cluster_indices) == 0:
            continue
            
        # 获取簇内样本的真实标签
        cluster_true_labels = true_labels[cluster_indices]
        
        # 计算簇的纯度（最多标签的比例）
        label_counts = Counter(cluster_true_labels.tolist())
        majority_label, majority_count = label_counts.most_common(1)[0]
        purity = majority_count / len(cluster_indices)
        
        # 记录簇信息
        cluster_info = {
            'cluster_id': i,
            'size': len(cluster_indices),
            'purity': purity,
            'majority_label': majority_label,
            'majority_count': majority_count,
            'label_distribution': dict(label_counts)
        }
        
        cluster_purities.append(cluster_info)
        
        # 更新总体统计
        total_samples += len(cluster_indices)
        total_correct += majority_count
    
    # 计算平均纯度
    if len(cluster_purities) > 0:
        avg_purity = sum(info['purity'] for info in cluster_purities) / len(cluster_purities)
        weighted_avg_purity = total_correct / total_samples if total_samples > 0 else 0.0
    else:
        avg_purity = 0.0
        weighted_avg_purity = 0.0
    
    # 整体统计信息
    overall_stats = {
        'total_clusters': len(clusters),
        'total_samples_in_clusters': total_samples,
        'avg_purity': avg_purity,
        'weighted_avg_purity': weighted_avg_purity,
        'purity_std': np.std([info['purity'] for info in cluster_purities]) if cluster_purities else 0.0
    }
    
    return avg_purity, cluster_purities, overall_stats

def assign_pseudo_labels(purified_cluster_list, u_feats, memory):
    """
    为u_feats分配伪标签，基于purified_cluster_list中的prototype
    
    Args:
        purified_cluster_list: 净化后的cluster_list
        u_feats: 无标签数据的特征，形状为 [n_samples, feature_dim]
        memory: ClusterMemory对象，用于获取已知类别信息
    
    Returns:
        pseudo_labels: 每个u_feats对应的伪标签，形状为 [n_samples]
        prototype_similarities: 每个u_feats与所有prototype的相似度，形状为 [n_samples, num_prototypes]
        all_prototypes: 所有prototype的tensor [N, feature_dim]
        all_prototype_labels: 所有prototype对应的标签列表
    """
    # 1. 从purified_cluster_list和memory中构建所有prototype和标签
    all_prototype_dict = {}
    all_prototypes = []
    all_prototype_labels = []
    
    # 从memory中获取已知类别的prototype和标签
    known_prototypes_dict, known_labels, _ = memory.get_known_class_prototypes()
    
    # 添加已知类别的prototype
    for label in known_labels:
        all_prototypes.append(known_prototypes_dict[label])
        all_prototype_dict[label]= known_prototypes_dict[label]
        all_prototype_labels.append(label)
    
    # 从purified_cluster_list中提取所有新类prototype和标签
    for cluster_group in purified_cluster_list:
        for cluster in cluster_group:
            if len(cluster) >= 5:  # 确保有分配的新标签
                new_label = cluster[4]
                new_prototype = cluster[2]  # prototype是第3个元素
                
                # 只添加新类别的prototype
                if new_label not in all_prototype_labels:
                    all_prototypes.append(new_prototype)
                    all_prototype_dict[new_label]=new_prototype
                    all_prototype_labels.append(new_label)
    
    # 转换为tensor
    all_prototypes = torch.stack(all_prototypes)  # 形状 [N, feature_dim]
    
    # 2. 归一化特征和原型
    u_feats_norm = F.normalize(u_feats, p=2, dim=1)
    all_prototypes_norm = F.normalize(all_prototypes, p=2, dim=1)
    
    # 3. 计算相似度矩阵
    prototype_similarities = torch.matmul(u_feats_norm, all_prototypes_norm.T)  # 形状 [n_samples, N]
    
    # 4. 为每个u_feat找到最相似的prototype
    max_similarities, max_indices = torch.max(prototype_similarities, dim=1)  # 形状 [n_samples]
    
    # 5. 将prototype索引转换为对应的标签
    pseudo_labels = torch.tensor([all_prototype_labels[idx] for idx in max_indices], 
                                device=u_feats.device, dtype=torch.long)
    
    # 6. 打印分配统计
    unique_labels, counts = torch.unique(pseudo_labels, return_counts=True)
    print("伪标签分配统计:")
    for label, count in zip(unique_labels, counts):
        print(f"标签 {label.item()}: {count.item()} 个样本")
    
    # 7. 分离已知类和新类的分配情况
    known_assignments = {}
    new_assignments = {}
    
    for label in unique_labels:
        label_item = label.item()
        count = counts[unique_labels == label].item()
        
        # 使用memory的mask_init判断已知类
        if 0 <= label_item < len(memory.mask_init) and memory.mask_init[label_item] == 1:
            known_assignments[label_item] = count
        else:
            new_assignments[label_item] = count
    
    print(f"\n已知类分配: {len(known_assignments)} 个类别")
    print(f"新类分配: {len(new_assignments)} 个类别")
    
    return pseudo_labels, prototype_similarities, all_prototypes, all_prototype_dict


def find_strong_ufeats(features,logits,label_indices, prototype, known_samples, threshold=0.7, 
                           prototype_weight=0.5, known_weight=0.5, min_confidence=0.6,max_samples_per_class=50):
    """
    找出与当前类别最相似的强真值样本
    
    Args:
        u_feats: 无标签样本特征 [n, 768]
        prototype: 当前类别的prototype [768]
        known_samples: 已知属于该类的样本 [50, 768]
        threshold: 相似度阈值
        prototype_weight: prototype相似度权重
        known_weight: 已知样本相似度权重
        min_confidence: 最小置信度阈值
    
    Returns:
        strong_true_indices: 强真值样本在u_feats中的索引
        remaining_indices: 剩余样本在u_feats中的索引
        confidence_scores: 所有样本的置信度分数
    """
    u_feats = features
    n_samples = features.shape[0]

    mask = torch.zeros(n_samples, dtype=torch.bool)
    
    # 1. 归一化所有特征
    normalized_u_feats = F.normalize(u_feats, p=2, dim=1)
    normalized_prototype = F.normalize(prototype.unsqueeze(0), p=2, dim=1)
    sim_to_prototype = torch.mm(normalized_u_feats, normalized_prototype.t()).squeeze()  # [n]

    if len(known_samples) >0:
        known_samples_list = [tensor.unsqueeze(0) for tensor in known_samples]  # 每个变成 [1, 768]
        known_samples_ = torch.cat(known_samples_list, dim=0)  
        normalized_known = F.normalize(known_samples_, p=2, dim=1)
        sim_to_known = torch.mm(normalized_u_feats, normalized_known.t())  # [n, 50]
        avg_sim_to_known = sim_to_known.mean(dim=1)
        confidence_scores = (0.8 * sim_to_prototype + 0.2 * avg_sim_to_known)
    
    else:
        confidence_scores = sim_to_prototype

    min_samples = 40  # 最小样本数量
    max_samples = 150  # 最大样本数量

    # 如果样本总数少于最小样本数量，选择所有样本
    if confidence_scores.shape[0] <= min_samples:
        strong_true_indices = torch.arange(confidence_scores.shape[0]).tolist()
    else:
        # 否则选择前max_samples个样本，但至少min_samples个
        k = min(max_samples, confidence_scores.shape[0])
        _, top_indices = torch.topk(confidence_scores, k=k, largest=True)
        strong_true_indices = top_indices.tolist()

    # 将tensor转换为集合进行操作
    label_set = set(label_indices.cpu().numpy())
    strong_set = set(strong_true_indices)

    # 取差集得到weak indices
    weak_set = label_set - strong_set

    # 转换回tensor
    weak_indices = torch.tensor(list(weak_set), device=label_indices.device)
    
    return strong_true_indices,weak_indices


def inverse_convert_labels(converted_labels, all_len):
    original_labels = []
    half_len = all_len // 2  # 已知类别的数量
    
    for conv_label in converted_labels:
        if conv_label < half_len:
            # 这是已知类别（偶数标签）
            original_label = conv_label * 2
        else:
            # 这是未知类别（奇数标签）
            original_label = (conv_label - half_len) * 2 - 1
        original_labels.append(original_label)
    
    return np.array(original_labels)

def convert_labels(labels,all_len):
    converted_labels = []
    for label in labels:
        if label % 2 == 0:
            converted_labels.append(label // 2)
        else:
            converted_labels.append(int(all_len/2) + ((label+1) // 2))
    return np.array(converted_labels)


def save_results(ids, all_preds, prototype_higher, preds_higher, save_path="results.pth"):
    # uq_index, all_preds, cluster_protos_list, preds_ind_list
    def move_to_cpu(x):
        """如果是 Tensor，转为 CPU；如果是 numpy，不变"""
        if isinstance(x, torch.Tensor):
            return x.cpu()
        return x

    results = {
        "ids": move_to_cpu(ids),
        "all_preds": move_to_cpu(all_preds),
        "prototype_higher": [move_to_cpu(ph) for ph in prototype_higher],
        "preds_higher": [move_to_cpu(ph) for ph in preds_higher]
    }
    # 保存到本地文件

    torch.save(results, save_path)
    print(f"Results saved to {save_path}")


def find_clusters_from_relations(u_feats, prototype, u_targets,labels_indices,weak_indices, threshold=0.7, min_cluster_size=10):
    """
    通过关系矩阵发现潜在类别簇
    
    Args:
        u_feats: 所有无标签样本特征 [9935, 768]
        prototype: 当前类别的prototype [768]
        labels_indices: 预测为该类的样本索引 [n]
        threshold: 相似度阈值
        min_cluster_size: 最小簇大小
    
    Returns:
        clusters: 发现的簇列表，每个元素为(prototype, 样本索引列表)
    """
    # 1. 获取当前类别的样本特征
    if len(weak_indices) == 0:
        return []
    
    global_weak_indices = weak_indices
    
    cluster_feats = u_feats[global_weak_indices]  # [n, 768]
    n_samples = cluster_feats.shape[0]
    
    # 如果样本太少，直接返回空列表
    if n_samples < min_cluster_size:
        return []
    
    # 2. 计算关系矩阵
    normalized_feats = F.normalize(cluster_feats, p=2, dim=1)
    relation_matrix = torch.mm(normalized_feats, normalized_feats.t())  # [n, n]
    
    # 3. 使用图论方法发现连通分量
    clusters = _find_connected_components(relation_matrix,global_weak_indices,u_targets,threshold,min_cluster_size)
    
    # 4. 为每个簇计算prototype
    result_clusters = []
    for cluster_indices in clusters:
        # 计算该簇的prototype（平均特征）
        acc,main_label = evaluate_strong_true_accuracy(cluster_indices, u_targets)
        processed_indices = []
        for i, indices in enumerate(cluster_indices):
            if indices.dim() == 0:  # 零维张量（标量）
                processed_indices.append(indices.unsqueeze(0))  # 变为 [1] 形状
            else:
                processed_indices.append(indices)

        # 现在可以安全拼接
        all_indices = torch.cat(processed_indices)
        cluster_features = u_feats[all_indices]
        cluster_prototype = cluster_features.mean(dim=0)
        
        result_clusters.append((main_label,acc,cluster_prototype,cluster_indices))
    
    return result_clusters

def _find_connected_components(relation_matrix, labels_indices, u_targets, threshold=0.7, min_cluster_size=6):
    """
    使用图连通分量算法发现簇
    如果初始阈值找不到簇，会自动降低阈值直到找到有效簇
    """
    n_samples = relation_matrix.shape[0]
    
    current_threshold = threshold
    min_threshold = 0.3  # 设置最小阈值，避免无限循环
    
    valid_clusters = []
    
    while current_threshold >= min_threshold:
        # 创建图
        G = nx.Graph()
        G.add_nodes_from(range(n_samples))
        
        # 添加边（相似度大于当前阈值的样本对）
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                if relation_matrix[i, j] > current_threshold:
                    G.add_edge(i, j)
        
        # 找到所有连通分量
        connected_components = list(nx.connected_components(G))
        
        # 过滤出大小满足要求的簇
        temp_valid_clusters = []
        for component in connected_components:
            if len(component) >= min_cluster_size:
                original_indices = [labels_indices[i] for i in component]
                temp_valid_clusters.append(original_indices)
        
        if len(temp_valid_clusters) > 0:
            valid_clusters = temp_valid_clusters
            print(f"使用阈值 {current_threshold:.1f} 找到 {len(valid_clusters)} 个簇")
            break
        else:
            print(f"阈值 {current_threshold:.1f} 未找到有效簇，降低阈值...")
            current_threshold -= 0.1
            min_cluster_size= 5
    
    # 如果降低阈值后仍然找不到簇，返回空列表
    if len(valid_clusters) == 0:
        print("警告: 即使降低阈值后仍未找到有效簇")
    return valid_clusters


def load_results(load_path="results.pth", local_rank=0):
    # 加载数据
    device = torch.device(f"cuda:{local_rank}")
    results = torch.load(load_path, map_location=device)
    def move_to_device(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        return x  # numpy.ndarray 直接返回
    
    # 确保数据被正确加载到 GPU（如果需要）
    results["ids"] = move_to_device(results["ids"])
    results["all_preds"] = move_to_device(results["all_preds"])
    results["prototype_higher"] = [move_to_device(ph) for ph in results["prototype_higher"]]
    results["preds_higher"] = [move_to_device(ph) for ph in results["preds_higher"]]

    print(f"Results loaded from {load_path}")
    return results


class CM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        temp = ctx.features
        outputs = inputs.mm(temp)

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            temp = ctx.features.t()
            grad_inputs = grad_outputs.mm(temp)

        # momentum update
        for x, y in zip(inputs, targets):
            temp[y] = ctx.momentum * temp[y] + (1. - ctx.momentum) * x
            temp[y] /= temp[y].norm()

        return grad_inputs, None, None, None


def cm(inputs, indexes, features, momentum=0.5, local_rank=None):
    if local_rank is None:
        return CM.apply(inputs, indexes, features.cuda(local_rank,non_blocking=True), torch.Tensor([momentum]).cuda(local_rank,non_blocking=True))
    else:
        return CM.apply(inputs.cuda(local_rank,non_blocking=True), indexes, features.cuda(local_rank,non_blocking=True), torch.Tensor([momentum]).cuda(local_rank,non_blocking=True))


class CM_Hard(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        temp = ctx.features
        outputs = inputs.mm(temp)

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            median = np.argmin(np.array(distances))
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
            ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None


def cm_hard(inputs, indexes, features, momentum=0.5, local_rank=None):
    if local_rank is None:
        return CM_Hard.apply(inputs, indexes, features, torch.Tensor([momentum]).cuda(local_rank,non_blocking=True))
    else:
        return CM_Hard.apply(inputs.cuda(local_rank,non_blocking=True), indexes, features, torch.Tensor([momentum]).cuda(local_rank,non_blocking=True))


class ClusterMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples,local_rank,known_labels,all_sample=1500,temp=0.05, momentum=0.2, use_hard=False, args=None):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard
        if local_rank is not None:
            self.local_rank = local_rank
        else:
            self.local_rank = 0
        
        self.register_buffer('features', torch.zeros(num_samples, num_features))
        self.register_buffer('boundary', torch.zeros(num_samples,4,num_features))
        self.register_buffer('s_features', torch.zeros(num_samples, num_features))
        self.register_buffer('t_features', torch.zeros(num_samples, num_features))
        self.register_buffer('all_feature', torch.zeros(all_sample, num_features))
        self.register_buffer('cls_feature',torch.zeros(num_samples,num_samples))

        self.mask_init = torch.ones(num_samples) * -1  # 初始化为-1，表示未初始化
        if known_labels is not None:
            for label in known_labels:
                if 0 <= label < num_samples:
                    self.mask_init[label] = 1  # 已知类标记为1

        self.feature_lists = [[] for _ in range(num_samples)]  # 存储每个类别的特征列表
        self.feature_indices = [[] for _ in range(num_samples)] 

        # 新增：可学习的prompt，用于存储每个类别的共性
        self.class_prompts = nn.Parameter(torch.randn(num_samples, num_features))
       
        # 记录哪些类别的prompt已经被初始化
        self.prompt_initialized = torch.zeros(num_samples, dtype=torch.bool)

    def forward(self,inputs,targets):
        if self.local_rank is not None:
            inputs = F.normalize(inputs, dim=1).cuda(self.local_rank,non_blocking=True)
            targets = targets.cuda(self.local_rank,non_blocking=True)
        else:
            inputs = F.normalize(inputs, dim=1).cuda(self.local_rank,non_blocking=True)
            targets = targets.cuda(self.local_rank,non_blocking=True)
        if self.use_hard:
            outputs = cm_hard(inputs, targets, self.features, self.momentum, self.local_rank)
        else:
            outputs = cm(inputs, targets, self.features, self.momentum, self.local_rank)

        outputs /= self.temp
        loss = F.cross_entropy(outputs, targets)
        return loss

    def update(self, new_labels, features):
        for i, label in enumerate(new_labels):
            if label == -1:  # 无更新
                continue
            if label < 0 or label >= self.num_samples:
                continue
                
            # 获取对应特征
            feature = features[i].detach().clone()
            # 添加到对应类别的特征列表
            self.feature_lists[label].append(feature)
            
            # 计算该类别的平均特征
            if self.feature_lists[label]:
                avg_feature = torch.stack(self.feature_lists[label]).mean(dim=0)
                
                if self.mask_init[label] == -1:  # 首次初始化
                    self.features[label] = avg_feature
                    self.mask_init[label] =1  # 标记为已初始化
                else: 
                    self.features[label] = avg_feature

    def _init_prompt_with_feature(self, label=None,prototype=None):

        with torch.no_grad():
            if label is not None and prototype is not None:
                self.class_prompts.data[label] = prototype.clone()
                self.prompt_initialized[label] = True
            else:
                # 初始化所有已初始化的类别的prompt
                for label_idx in range(self.num_samples):
                    if not self.prompt_initialized[label_idx]:
                        self.class_prompts.data[label_idx] = self.features[label_idx].clone()
                        self.prompt_initialized[label_idx] = True

    def manual_update_with_prototypes(self, all_new_prototypes_dict):
        """
        手动使用外部prototype字典更新memory
        Args:
            all_new_prototypes_dict: 字典，key为标签，value为对应的prototype向量
        """
        updated_labels = []
        for label, prototype in all_new_prototypes_dict.items():
            if 0 <= label < self.num_samples:
                # 更新memory中的feature
                self.features[label] = prototype.detach().clone()
                
                # 更新对应的feature list，清空并添加新的prototype
                self.feature_lists[label] = [prototype.detach().clone()]
                
                # 标记为已初始化
                self.mask_init[label] = 1
                
                # 更新对应的prompt
                self._init_prompt_with_feature(label, prototype)
                
                updated_labels.append(label)
        
        print(f"手动更新了 {len(updated_labels)} 个类别的prototype: {updated_labels}")

    def get_known_class_prototypes(self,st_type=None):
        """
        获取当前memory中已知类别的prototype及对应标签
        
        Returns:
            known_prototypes_dict: 字典，{label: prototype_tensor}
            known_labels: 列表，已知类别的标签
            feature_indices_dict: 字典，{label: [indices]}
        """
        known_prototypes_dict = {}
        known_labels = []
        feature_indices_dict = {}
        
        if st_type == 'space':
            for label in range(self.num_samples):
                if self.mask_init[label] == 1:  # 已知类别
                    known_prototypes_dict[label] = self.s_features[label].clone()
                    known_labels.append(label)
                    feature_indices_dict[label] = self.feature_indices[label].copy()
        elif st_type == 'time':
            for label in range(self.num_samples):
                if self.mask_init[label] == 1:  # 已知类别
                    known_prototypes_dict[label] = self.t_features[label].clone()
                    known_labels.append(label)
                    feature_indices_dict[label] = self.feature_indices[label].copy()
        else:
            for label in range(self.num_samples):
                if self.mask_init[label] == 1:  # 已知类别
                    known_prototypes_dict[label] = self.features[label].clone()
                    known_labels.append(label)
                    feature_indices_dict[label] = self.feature_indices[label].copy()
        
        return known_prototypes_dict,known_labels, feature_indices_dict

    
# generate new dataset and calculate cluster centers
@torch.no_grad()
def generate_cluster_features(labels, features,num_class):
    centers = collections.defaultdict(list)
    boundary_points = collections.defaultdict(dict)

    for i, label in enumerate(labels):
        if label == -1:
            continue
        centers[labels[i]].append(features[i])
        
            
    final_centers = []
    for idx in range(num_class):
        if idx in centers:
            # 计算该类别的特征中心
            center = torch.stack(centers[idx], dim=0).mean(0)
            boundary_points[idx] = find_boundary_points_for_class(
            centers[idx], center)
        else:
            # 生成归一化随机向量作为该类别的中心
            center = torch.randn_like(features[0])
            boundary_points[idx] = {'first': None, 'second': None}
            
        final_centers.append(center)

    all_centers = torch.stack(final_centers, dim=0) 

    return all_centers

@torch.no_grad()
def generate_cluster_boundary(labels, features,num_class):
    centers = collections.defaultdict(list)
    boundary_points = collections.defaultdict(dict)

    for i, label in enumerate(labels):
        if label == -1:
            continue
        centers[labels[i]].append(features[i])
        
    final_centers = []
    for idx in range(num_class):
        if idx in centers:
            # 计算该类别的特征中心
            center = torch.stack(centers[idx], dim=0).mean(0)
        else:
            # 生成归一化随机向量作为该类别的中心
            center = torch.randn_like(features[0])

        boundary_points[idx] = find_boundary_points_for_class(
        centers[idx], center)
            
        final_centers.append(center)

    all_centers = torch.stack(final_centers, dim=0)
    boundary_points = torch.stack([torch.stack(boundary_points[i], dim=0) for i in range(num_class)], dim=0)

    return all_centers,centers,boundary_points


def find_boundary_points_for_class(class_samples: List[torch.Tensor], 
                                  prototype: torch.Tensor) -> List[torch.Tensor]:
    """
    为单个类别寻找4个边界点
    
    参数:
        class_samples: 该类别的所有样本列表，每个样本是归一化的768维张量
        prototype: 该类别的原型向量
    
    返回:
        List: 包含4个边界点向量的列表
    """
    boundary_points = []
    if len(class_samples) < 4:
        if len(class_samples)!=0:
            boundary_points = class_samples.copy()  # 先添加所有现有样本
        
        # 生成随机向量补足到4个
        for _ in range(4 - len(class_samples)):
            random_vector = torch.randn_like(prototype)
            normalized_random = F.normalize(random_vector, dim=0)
            boundary_points.append(normalized_random)
        
        return boundary_points
    
    # 将样本堆叠成张量
    samples_tensor = torch.stack(class_samples, dim=0)  # [n, 768]
    
    # 计算每个样本与prototype的余弦距离
    prototype_expanded = prototype.unsqueeze(0)  # [1, 768]
    similarities = F.cosine_similarity(samples_tensor, prototype_expanded, dim=1)  # [n]
    distances_to_prototype = 1 - similarities  # [n]
    
    # 初始化边界点列表
    boundary_points = []
    selected_indices = []
    
    # 选择第一个边界点：距离prototype最远的样本
    first_idx = torch.argmax(distances_to_prototype).item()
    boundary_points.append(samples_tensor[first_idx])
    selected_indices.append(first_idx)
    
    # 选择后续边界点：既要远离prototype，也要远离已选边界点
    for _ in range(3):  # 还需要选择3个边界点
        # 计算每个样本到所有已选边界点的平均距离
        avg_distances_to_selected = torch.zeros(samples_tensor.size(0))
        
        for selected_idx in selected_indices:
            selected_point = samples_tensor[selected_idx].unsqueeze(0)  # [1, 768]
            similarities_to_selected = F.cosine_similarity(
                samples_tensor, selected_point, dim=1
            )
            distances_to_selected = 1 - similarities_to_selected
            avg_distances_to_selected += distances_to_selected
        
        avg_distances_to_selected /= len(selected_indices)
        
        # 综合得分：到prototype的距离 + 到已选边界点的平均距离
        combined_scores = distances_to_prototype + avg_distances_to_selected
        
        # 将已选边界点的得分设为-1，确保不会重复选择
        for selected_idx in selected_indices:
            combined_scores[selected_idx] = -1
        
        # 选择综合得分最高的样本作为下一个边界点
        next_idx = torch.argmax(combined_scores).item()
        boundary_points.append(samples_tensor[next_idx])
        selected_indices.append(next_idx)
    
    return boundary_points


def get_random_features_from_linked_list(linked_list, label, num_samples=3):
    """
    根据给定的 label 随机选择 num_samples 个特征（默认为 3 个）。
    """
    if label not in linked_list:
        raise ValueError(f"Label {label} not found in linked_list.")
    
    features_list = linked_list[label]
    
    if len(features_list) < num_samples:
        raise ValueError(f"Not enough features for label {label}. Found {len(features_list)}, but need {num_samples}.")
    
    # 使用 random.sample 从 features_list 中随机选择 3 个特征
    selected_features = random.sample(features_list, num_samples)
    
    # 将选中的特征转换为 tensor 返回（如果需要）
    return torch.stack(selected_features)

@torch.no_grad()
def construct_sup_proj(sup_proj, sup_con_label, linked_list, features, labels):
    augmented_features = []
    augmented_labels = []
    
    device = sup_proj.device  # 获取当前 GPU 设备
    
    for label in sup_con_label.tolist():
        if label in linked_list and len(linked_list[label]) >= 20:
            sampled_features = random.sample(linked_list[label], 2)
            stacked_features = torch.stack(sampled_features, dim=0).to(device)  # 放入同一 GPU
            augmented_features.append(stacked_features)
            augmented_labels.append(label)
    
    augmented_features = torch.stack(augmented_features, dim=0).to(device)  # Shape [num_classes, 2, 768]
    augmented_labels = torch.tensor(augmented_labels, device=device)
    return augmented_features, augmented_labels

def extract_features(model, data_loader,test_meter, args,print_freq=100, local_rank=0):
    model.eval()
    features = []
    s_features = []
    t_features = []
    labels = []
    indexs = []
    preds = []
    cls_list = []

    with torch.no_grad():
        for i, _item in enumerate(data_loader):
            #images, label, uq_idxs,index = batch
            imgs = _item[0]
            if isinstance(imgs, (list,)):
                imgs = imgs[0].cuda(local_rank,non_blocking=True)
            else:
                imgs = imgs.cuda(local_rank,non_blocking=True)

            label = _item[2]
            uq_idxs = _item[3]
            index = _item[4]

            mask = torch.tensor(np.array([True if x.item() in args.train_classes else False for x in label]))

            outputs = model(imgs)
            
            feature = outputs[0].data.cpu()

            pred = outputs[1].data.cpu()

            s_feature = outputs[2].data.cpu()
            t_feature = outputs[3].data.cpu()
            
            features.append(feature)
            s_features.append(s_feature)
            t_features.append(t_feature)
            cls_list.append(pred)

            test_meter.update_stats(
                pred, label, mask,uq_idxs
            )

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}]\t'
                      .format(i + 1))

    pred,label,mask = test_meter.finalize_preds()

    features = torch.cat(features, dim=0)
    s_features = torch.cat(s_features, dim=0)
    t_features = torch.cat(t_features, dim=0)
    cls_list = torch.cat(cls_list,dim=0)
    

    return pred, label,features,cls_list,s_features,t_features



def extract_labeled_protos(model, train_loader, merge_len,args,local):
    model.eval()

    all_feats = []
    targets = np.array([])
    mask = np.array([])
    ids=np.array([])
    mask_cls=np.array([])
    metrics=dict()
    with torch.no_grad():
        for batch_idx, _item in enumerate(train_loader):

            images = _item[0]
            label = _item[1]
            uq_idx = _item[2]
            mask_lab_ = _item[3]
            pseu_idx = _item[4]
            
            images = images.cuda(local,non_blocking=True)
            # images = images.to(device)
            label, mask_lab_ = label.cuda(local,non_blocking=True), mask_lab_.cuda(local,non_blocking=True).bool()
            # Pass features through base model and then additional learnable transform (linear layer)
            outputs = model(images)
            feats = outputs[2].data.cpu()

            all_feats.append(feats.numpy())
        
            targets = np.append(targets, label.cpu().numpy())
            ids=np.append(ids,pseu_idx.cpu().numpy())
            mask = np.append(mask, mask_lab_.cpu().bool().numpy())

    mask = mask.astype(bool)
    mask_cls = mask_cls.astype(bool)

    new_targets = convert_labels(targets,args.num_labeled_classes + args.num_unlabeled_classes)

    mask_cls = np.array([True if x.item() in range(len(args.train_classes))
                    else False for x in new_targets])

    all_feats = np.concatenate(all_feats)
    l_feats = all_feats[mask]  # Get labelled set
    u_feats = all_feats[~mask]  # Get unlabelled set
    l_targets = new_targets[mask]  # Get labelled targets
    u_targets = new_targets[~mask]  # Get unlabelled targets
    n_samples =len(targets)

    cluster_size=math.ceil(n_samples /(args.num_labeled_classes + args.num_unlabeled_classes))
    kmeanssem = SemiSupKMeans(k=args.num_labeled_classes + args.num_unlabeled_classes, tolerance=1e-4,
                              max_iterations=10, init='k-means++',
                              n_init=1, random_state=None, n_jobs=None, pairwise_batch_size=1024,
                              mode=None, protos=None,cluster_size=cluster_size)

    l_feats, u_feats, l_targets, u_targets = (torch.from_numpy(x).cuda(local,non_blocking=True) for
                                              x in (l_feats, u_feats, l_targets, u_targets))

    kmeanssem.fit_mix(u_feats, l_feats, l_targets)
    all_preds = kmeanssem.labels_
    mask_cls=mask_cls[~mask]
    preds = all_preds.cpu().numpy()[~mask]
    
    all_acc, old_acc, new_acc = split_cluster_acc_v2_kmeans(y_true=u_targets.cpu().numpy(), y_pred=preds, mask=mask_cls)

    metrics["all_acc"], metrics["old_acc"], metrics["new_acc"] = all_acc, old_acc, new_acc
    
    prototype_higher=[]
    prototypes = kmeanssem.cluster_centers_
    prototype_higher.append(prototypes.clone())
    n_labeled=args.num_labeled_classes
    n_novel= args.num_unlabeled_classes
    label_proto = prototypes.cpu().numpy()[:args.num_labeled_classes,:]
    preds_higher=[]

    preds_higher.append(all_preds.clone())
    print('Hierarchy clustering')
    mask_known=(all_preds<args.num_labeled_classes).cpu().numpy()
    l_feats = all_feats[mask_known]  # Get labelled set
    u_feats = all_feats[~mask_known]
    l_feats, u_feats= (torch.from_numpy(x).cuda(local,non_blocking=True) for  x in (l_feats, u_feats))

    while n_labeled>1:
        n_labeled=max(int(n_labeled/2),1)
        n_novel=max(int(n_novel/2),1)

        kmeans_l = KMeans(n_clusters=n_labeled, random_state=0).fit(label_proto)
        preds_labels = torch.from_numpy(kmeans_l.labels_).cuda(local,non_blocking=True)
        level_l_targets=preds_labels[all_preds[mask_known]]
        if args.unbalanced:
            cluster_size = None
        else:
            cluster_size = math.ceil( n_samples / (n_labeled+n_novel))
        kmeans_higher =SemiSupKMeans(k=n_labeled+n_novel, tolerance=1e-4,
                              max_iterations=10, init='k-means++',
                              n_init=1, random_state=None, n_jobs=None, pairwise_batch_size=1024,
                              mode=None, protos=None,cluster_size=cluster_size)
        kmeans_higher.fit_mix(u_feats, l_feats, level_l_targets)
        preds_level = kmeans_higher.labels_
        prototypes_level = kmeans_higher.cluster_centers_
        prototype_higher.append(prototypes_level.clone())
        preds_higher.append(preds_level.cuda(local,non_blocking=True).clone())
        
    return ids,all_preds, prototype_higher,preds_higher



def load_features_data(path):
    """加载保存的特征数据"""
    data = torch.load(path)
    return {
        'all_feats': data['all_feats'],
        'all_logits': data['all_logits'],
        's_all_feats': data['s_all_feats'],
        't_all_feats': data['t_all_feats'],
        'targets': data['targets'],
        'ids': data['ids'],
        'mask': data['mask'],
        'new_targets': data['new_targets'],
        'mask_cls': data['mask_cls']
    }

def get_similarity_for_labels(similarity_matrix, label_to_idx,idx_to_label, query_labels):
    
    # 获取查询标签对应的索引
    query_labels = query_labels.tolist()
    query_indices = [label_to_idx[label] for label in query_labels]

    prototype_idx = query_indices

    prototype_idx = torch.tensor(prototype_idx, device=similarity_matrix.device)
    
    # 提取对应的子矩阵
    sub_matrix = similarity_matrix[query_indices, :][:, query_indices]
    
    return sub_matrix,prototype_idx


def compute_prototype_similarity(known_prototypes_dict, known_labels, similarity_metric='cosine'):
    """
    从已知类别的prototype字典计算相似度矩阵
    
    参数:
    known_prototypes_dict: 字典格式 {label: prototype_vector}
    known_labels: 所有已知类的标签列表 [label1, label2, ...]
    similarity_metric: 相似度度量方法，可选 'cosine', 'euclidean', 'dot'
    
    返回:
    similarity_matrix: 形状为 [n_classes, n_classes] 的相似度矩阵
    label_to_idx: 标签到矩阵索引的映射字典，格式为 {label: index}
    idx_to_label: 矩阵索引到标签的映射字典，格式为 {index: label}
    """
    # 确保known_labels中的标签都在prototype字典中
    for label in known_labels:
        if label not in known_prototypes_dict:
            raise ValueError(f"标签 {label} 不在已知prototype字典中")
    
    # 创建标签到索引的映射 {label: index}
    label_to_idx = {}
    idx_to_label = {}
    
    for idx, label in enumerate(known_labels):
        label_to_idx[label] = idx
        idx_to_label[idx] = label
    
    n_classes = len(known_labels)
    
    # 构建prototype张量 [n_classes, feature_dim]
    prototypes = []
    for label in known_labels:
        prototype = known_prototypes_dict[label]
        if not torch.is_tensor(prototype):
            prototype = torch.tensor(prototype, dtype=torch.float32)
        prototypes.append(prototype)
    
    prototypes_tensor = torch.stack(prototypes)
    
    # 计算相似度矩阵
    if similarity_metric == 'cosine':
        prototypes_norm = F.normalize(prototypes_tensor, p=2, dim=1)
        similarity_matrix = torch.mm(prototypes_norm, prototypes_norm.t())
    
    elif similarity_metric == 'euclidean':
        distances = torch.cdist(prototypes_tensor, prototypes_tensor, p=2)
        max_distance = torch.max(distances)
        similarity_matrix = 1 - (distances / max_distance)
    
    elif similarity_metric == 'dot':
        similarity_matrix = torch.mm(prototypes_tensor, prototypes_tensor.t())
        min_val = torch.min(similarity_matrix)
        max_val = torch.max(similarity_matrix)
        similarity_matrix = (similarity_matrix - min_val) / (max_val - min_val)
    
    else:
        raise ValueError("不支持的相似度度量方法。请选择 'cosine', 'euclidean' 或 'dot'")
    
    return similarity_matrix, label_to_idx, idx_to_label




def find_most_dissimilar_cluster(clusters, c_prototype):
    """
    找到与当前原型最不相似的聚类
    
    参数:
        clusters: 聚类列表，每个聚类包含原型和样本索引
        c_prototype: 当前类原型
        
    返回:
        most_dissimilar_cluster: 最不相似的聚类
    """
    min_similarity = float('inf')
    most_dissimilar_cluster = None
    
    for cluster in clusters:
        cluster_prototype = cluster[2]  # 聚类原型
        similarity = F.cosine_similarity(
            c_prototype.unsqueeze(0), 
            cluster_prototype.unsqueeze(0)
        ).item()
        
        if similarity < min_similarity:
            min_similarity = similarity
            most_dissimilar_cluster = cluster
    
    return most_dissimilar_cluster


def process_cluster_indices(cluster_indices, device=None):
    """
    处理包含标量张量的cluster_indices列表
    
    参数:
        cluster_indices: 包含标量张量的列表
        device: 目标设备
    
    返回:
        一维整数张量
    """
    # 提取每个标量张量的值并创建一维张量
    indices_list = []
    for idx_tensor in cluster_indices:
        if isinstance(idx_tensor, torch.Tensor):
            # 如果是标量张量，使用item()获取值
            if idx_tensor.dim() == 0:
                indices_list.append(idx_tensor.item())
            else:
                # 如果是一维张量，添加到列表
                indices_list.extend(idx_tensor.flatten().tolist())
        else:
            # 如果不是张量，直接添加
            indices_list.append(idx_tensor)
    
    # 创建一维张量
    if device is not None:
        all_indices = torch.tensor(indices_list, dtype=torch.long, device=device)
    else:
        all_indices = torch.tensor(indices_list, dtype=torch.long)
    
    return all_indices

def mark_new_class(features, cluster, label, hard_label_mask, updated_u_targets, label_indices):
    """
    标记新类
    
    参数:
        features: 特征张量
        cluster: 聚类
        label: 当前标签
        hard_label_mask: 硬标签掩码
        updated_u_targets: 更新的未知标签
        label_indices: 当前类的样本索引
        
    返回:
        new_class_mask: 新类掩码
        new_prototype: 新类原型
    """
    cluster_prototype = cluster[2]  # 聚类原型
    cluster_indices = cluster[3]    # 聚类样本索引

    all_indices = process_cluster_indices(cluster_indices, device=features.device)
    
    if len(all_indices) == 0:
        return None, None
    
    # 更新硬标签掩码和标签
    hard_label_mask[all_indices] = True
    updated_u_targets[all_indices] = torch.tensor(label + 1, dtype=torch.int64, device=updated_u_targets.device)

    
    print(f"标记新类: 从类别 {label} 中发现 {len(all_indices)} 个样本作为新类 {label+1}")
    
    return hard_label_mask,updated_u_targets, cluster_prototype

def assign_weak_samples_to_new_class(features, weak_indices, c_prototype, new_prototype, 
                                   label, hard_label_mask, updated_u_targets, top_k=15):
    """
    从弱样本中分配样本到新类
    
    参数:
        features: 特征张量
        weak_indices: 弱样本索引
        c_prototype: 当前类原型
        new_prototype: 新类原型
        label: 当前标签
        hard_label_mask: 硬标签掩码
        updated_u_targets: 更新的未知标签
        top_k: 选择前k个最相似的样本
    """
    if len(weak_indices) == 0:
        return
    
    # 获取弱样本特征
    weak_features = features[weak_indices]
    
    # 计算与已知类原型和新类原型的相似度
    weak_features_norm = F.normalize(weak_features, p=2, dim=1)
    c_prototype_norm = F.normalize(c_prototype.unsqueeze(0), p=2, dim=1)
    new_prototype_norm = F.normalize(new_prototype.unsqueeze(0), p=2, dim=1)
    sim_to_known = torch.matmul(weak_features_norm, c_prototype_norm.t()).squeeze()
    sim_to_new = torch.matmul(weak_features_norm, new_prototype_norm.t()).squeeze()
    
    # 选择与新类最相似的前top_k个样本
    _, topk_indices = torch.topk(sim_to_new, k=min(top_k, len(weak_indices)))
    selected_weak_indices = weak_indices[topk_indices]
    
    # 给这些样本赋予硬标签
    hard_label_mask[selected_weak_indices] = True
    
    updated_u_targets[selected_weak_indices] = torch.tensor(label + 1, dtype=torch.int64, device=updated_u_targets.device)
    
    print(f"从弱样本中选择 {len(selected_weak_indices)} 个样本赋予新类 {label+1} 标签")
    
    # 对剩余的弱样本分配软标签
    remaining_mask = torch.ones(len(weak_indices), dtype=torch.bool)
    remaining_mask[topk_indices] = False
    remaining_weak_indices = weak_indices[remaining_mask]
    
    if len(remaining_weak_indices) > 0:
        updated_u_targets = assign_soft_labels_to_remaining(
            features, remaining_weak_indices, c_prototype, new_prototype,
            label, updated_u_targets
        )
    return updated_u_targets

def assign_soft_labels_to_remaining(features, remaining_indices, c_prototype, new_prototype,
                                  label, updated_u_targets):
    """
    给剩余的弱样本分配软标签
    
    参数:
        features: 特征张量
        remaining_indices: 剩余样本索引
        c_prototype: 当前类原型
        new_prototype: 新类原型
        label: 当前标签
        updated_u_targets: 更新的未知标签
    """
    if len(remaining_indices) == 0:
        return
    
    # 获取剩余样本特征
    remaining_features = features[remaining_indices]
    
    # 计算相似度
    remaining_features_norm = F.normalize(remaining_features, p=2, dim=1)
    c_prototype_norm = F.normalize(c_prototype.unsqueeze(0), p=2, dim=1)
    new_prototype_norm = F.normalize(new_prototype.unsqueeze(0), p=2, dim=1)
    
    sim_to_known = torch.matmul(remaining_features_norm, c_prototype_norm.t()).squeeze()
    sim_to_new = torch.matmul(remaining_features_norm, new_prototype_norm.t()).squeeze()
    
    # 根据相似度分配软标签（这里使用简单的阈值方法）
    # 您可以根据需要实现更复杂的软标签分配策略
    new_class_mask = sim_to_new > (sim_to_known +0.1)
    
    # 更新标签
    new_class_label = torch.tensor(label + 1, dtype=torch.int64, device=updated_u_targets.device)
    updated_u_targets[remaining_indices[new_class_mask]] = new_class_label

    return updated_u_targets



def classify_known_unknown_detailed(features,u_targets,centers,logits, known_prototypes, confidence_threshold=0.7, kl_threshold=2.0):

    # 确保输入是torch张量
    if not isinstance(features, torch.Tensor):
        features = torch.tensor(features, dtype=torch.float32)
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits, dtype=torch.float32)
    if not isinstance(known_prototypes, torch.Tensor):
        known_prototypes = torch.tensor(known_prototypes, dtype=torch.float32)
    
    n_samples = features.shape[0]
    num_known_classes = known_prototypes.shape[0]

    # 归一化特征和原型
    features_norm = F.normalize(features, p=2, dim=1)  # 形状 [9935, 768]
    prototypes_norm = F.normalize(known_prototypes, p=2, dim=1)  # 形状 [51, 768]

    similarities = torch.matmul(features_norm, prototypes_norm.T)  # 9935x768 * 768x51 = 9935x51
    max_similarities, most_similar_indices = torch.max(similarities, dim=1)  # 两个都是 [9935]

    min_val = most_similar_indices.min()
    max_val = most_similar_indices.max()

    most_similar_labels = most_similar_indices * 2  # 形状 [9935]

    pre_logits = torch.argmax(logits, dim=1)

    # select 
    known_labels = torch.unique(most_similar_labels)
    mask = torch.zeros(len(max_similarities), dtype=torch.bool)

    # visualize_all_prototypes(features,u_targets,pre_logits,known_prototypes,n_samples=200,save_path="./all_vis.png")
            # 初始化返回变量
    hard_label_mask = torch.zeros(n_samples, dtype=torch.bool, device=features.device)

    updated_u_targets = torch.full((n_samples,), -1.0, dtype=torch.int64, device=features.device)
    all_new_prototypes = {}

    for label in known_labels :
        # 找到预测为该类别的所有样本
        
        label_mask = (most_similar_labels == label)

        label_indices = torch.where(label_mask)[0]

        class_u_feats = features[label_indices]
        c_prototype = known_prototypes[int(label/2)]


        if label ==known_labels[-1]:
            continue
        
        if len(label_indices) > 0:
            # 获取这些样本的相似度
            label_similarities = max_similarities[label_indices]
            
            # 按相似度排序（从高到低）
            sorted_indices = torch.argsort(label_similarities, descending=True)
            
            # 选择前1/3的样本
            top_k = max(1, len(label_indices) // 3)  # 确保至少选择1个样本
            top_label_indices = label_indices[sorted_indices[:top_k]]
            
            # 将这些样本的mask设置为True
            mask[top_label_indices] = True
    
    print(f"总共发现 {len(all_new_prototypes)} 个新类")
    print(f"分配硬标签的样本数量: {hard_label_mask.sum().item()}")
    pseudo_labels = torch.zeros(n_samples, dtype=torch.long)
    pseudo_labels = most_similar_labels

    return mask, pseudo_labels, similarities, max_similarities


class GravitationalAssignmentLoss(nn.Module):
    def __init__(self, known_prototypes, cluster_prototypes, cluster_sizes,
                 lambda_grav=1.0, lambda_sim=0.5, temperature=0.1):
        """
        known_prototypes: 已知类prototype [50, feature_dim]
        cluster_prototypes: 未知簇prototype [num_clusters, feature_dim]
        cluster_sizes: 各簇当前大小 [num_clusters]
        """
        super().__init__()
        
        self.known_prototypes = known_prototypes
        self.cluster_prototypes = cluster_prototypes
        self.cluster_sizes = cluster_sizes
        self.num_clusters = cluster_prototypes.shape[0]
        
        self.lambda_grav = lambda_grav
        self.lambda_sim = lambda_sim
        self.temperature = temperature
        self.num_k = known_prototypes.shape[0]
        self.smoothing = 0.1

        self.all_prototypes = torch.cat([known_prototypes, cluster_prototypes], dim=0)
    
    def forward(self, unassigned_features, unassigned_indices,local_rank):
        num_samples = unassigned_features.shape[0]
        num_prototypes = self.all_prototypes.shape[0]
        all_prototypes = self.all_prototypes.cuda(local_rank,non_blocking=True)

        # 计算相似度矩阵
        similarities = F.cosine_similarity(
            unassigned_features.unsqueeze(1),
            all_prototypes.unsqueeze(0),
            dim=2
        ) / self.temperature  # [N, M]
        
        # 简单的损失：最小化与最近prototype的负相似度
        _, nearest_indices = torch.max(similarities, dim=1)
        
        # 创建平滑的标签分布
        smooth_labels = torch.full_like(similarities, self.smoothing / (num_prototypes - 1))
        smooth_labels[torch.arange(num_samples), nearest_indices] = 1.0 - self.smoothing
        
        # 使用KL散度损失（比交叉熵更温和）
        pred_probs = F.log_softmax(similarities, dim=1)
        loss = F.kl_div(pred_probs, smooth_labels, reduction='batchmean')
        
        # 分配概率（使用softmax）
        assignment_probs = F.softmax(similarities, dim=1)
        
        return loss, assignment_probs
    
    

def process_clusters_and_assign_labels(clusters_list, known_prototypes, u_feats):
    """
    处理所有循环中收集的聚类，分配新类标签，并计算硬标签和软标签
    
    Args:
        clusters_list: 所有循环中收集的聚类列表，每个元素是一个cluster的列表
        known_prototypes: 已知类别的原型，形状为[51, 768] (0,2,4,...,100)
        u_feats: 无标签样本特征，形状为[N, 768]
    
    Returns:
        hard_labels: 硬标签，形状为[N]
        soft_labels: 软标签，形状为[N, 101]
        new_class_prototypes: 新类原型，形状为[50, 768]
        new_class_labels: 新类标签，形状为[50]
        new_class_main_labels: 新类对应的main_label列表
        new_class_high_indices: 新类对应的样本indices列表
    """
    # 1. 收集所有小簇的prototype及相关信息
    all_cluster_prototypes = []
    all_main_labels = []  # 保存所有main_label
    all_high_indices_list = []  # 保存所有high_indices

    for clusters in clusters_list:
        for cluster in clusters:
            if isinstance(cluster, tuple) and len(cluster) > 0:
                # 假设每个cluster的结构为 (main_label, ..., prototype, high_indices, ...)
                prototype = cluster[2]
                main_label = cluster[0]
                high_indices = cluster[3]
                if isinstance(prototype, torch.Tensor) and prototype.dim() == 1 and prototype.size(0) == 768:
                    all_cluster_prototypes.append(prototype)
                    all_main_labels.append(main_label)
                    all_high_indices_list.append(high_indices)
     
    if not all_cluster_prototypes:
        # 如果没有收集到任何小簇prototype，返回默认值
        return None, None, None, None, None, None
    
    # 将prototype列表转换为张量
    all_cluster_prototypes = torch.stack(all_cluster_prototypes)  # 形状为[M, 768]
    
    # 2. 构建所有prototype的集合
    all_prototypes = torch.cat([known_prototypes, all_cluster_prototypes], dim=0)  # 形状为[51+M, 768]
    
    # 3. 计算prototype之间的相似度
    normalized_prototypes = F.normalize(all_prototypes, p=2, dim=1)
    similarity_matrix = torch.mm(normalized_prototypes, normalized_prototypes.t())  # 形状为[51+M, 51+M]
    
    # 4. 逐步选择与所有类别最不相似的prototype作为新类
    # 初始化已选新类集合及相关信息
    selected_new_classes = []
    selected_indices = []
    selected_main_labels = []  # 保存选中的main_label
    selected_high_indices_list = []  # 保存选中的样本indices
    
    # 已知类别索引
    known_indices = list(range(51))
    
    # 小簇prototype索引
    cluster_indices = list(range(51, all_prototypes.size(0)))
    
    # 逐个选择新类
    num_new_classes = min(50, len(cluster_indices))
    for i in range(num_new_classes):
        # 计算每个候选prototype与所有已选类别（已知+已选新类）的最小相似度
        min_similarities = []
        
        for candidate_idx in cluster_indices:
            if candidate_idx in selected_indices:
                continue
                
            # 计算与已知类别的相似度
            known_similarities = similarity_matrix[candidate_idx, known_indices]
            
            # 计算与已选新类的相似度
            if selected_indices:
                new_similarities = similarity_matrix[candidate_idx, selected_indices]
                all_similarities = torch.cat([known_similarities, new_similarities])
            else:
                all_similarities = known_similarities
            
            # 取最小相似度（最不相似的）
            min_sim = torch.min(all_similarities)
            min_similarities.append(min_sim.item())
        
        # 如果没有候选了，提前结束
        if not min_similarities:
            break
            
        # 选择最小相似度最大的候选（即与所有已选类别最不相似的）
        max_min_sim_idx = torch.argmax(torch.tensor(min_similarities))
        selected_idx = cluster_indices[max_min_sim_idx]
        
        # 计算在原始小簇列表中的索引
        original_cluster_idx = selected_idx - 51
        
        # 获取对应的main_label和high_indices
        selected_main_label = all_main_labels[original_cluster_idx]
        selected_high_indices = all_high_indices_list[original_cluster_idx]
        
        # 添加到已选集合
        selected_indices.append(selected_idx)
        selected_new_classes.append(all_prototypes[selected_idx])
        selected_main_labels.append(selected_main_label)
        selected_high_indices_list.append(selected_high_indices)
        
        # 从候选列表中移除
        cluster_indices.pop(max_min_sim_idx)
    
    # 获取选中的新类prototype
    new_class_prototypes = torch.stack(selected_new_classes)  # 形状为[K, 768]，K<=50
    
    # 5. 分配新类标签（1, 3, 5, ..., 99）
    num_selected = new_class_prototypes.size(0)
    new_class_labels = torch.arange(1, 2*num_selected, 2, device=new_class_prototypes.device)
    
    # 6. 构建包含所有类别的完整prototype集合
    # 已知类别标签为0,2,4,...,100，共51个
    # 新类别标签为1,3,5,...,99，共50个
    all_class_prototypes = torch.cat([known_prototypes, new_class_prototypes], dim=0)  # 形状为[51+K, 768]
    
    # 创建已知类别标签（0,2,4,...,100）
    known_class_labels = torch.arange(0, 101, 2, device=all_class_prototypes.device)
    
    # 合并所有类别标签
    all_class_labels = torch.cat([known_class_labels[:51], new_class_labels])
    
    # 7. 计算无标签样本的硬标签和软标签
    normalized_u_feats = F.normalize(u_feats, p=2, dim=1)
    normalized_all_prototypes = F.normalize(all_class_prototypes, p=2, dim=1)
    
    # 计算相似度
    similarities = torch.mm(normalized_u_feats, normalized_all_prototypes.t())  # 形状为[N, 51+K]
    
    # 硬标签：选择相似度最高的prototype对应的标签
    _, max_similarity_indices = torch.max(similarities, dim=1)
    hard_labels = all_class_labels[max_similarity_indices]
    
    # 软标签：对相似度进行softmax处理，得到概率分布
    soft_labels = F.softmax(similarities, dim=1)  # 形状为[N, 51+K]
    
    return hard_labels, soft_labels,all_cluster_prototypes, new_class_labels, selected_main_labels, selected_high_indices_list

def compute_cluster_similarities(cluster_list, memory, u_feats):
    """
    合并已知prototype和cluster_list中的新类prototype，计算与u_feats的相似度，并统计每个prototype对应的样本
    
    Args:
        cluster_list: 包含未知簇的列表，每个元素是一个列表，代表与某个已知类相关的未知簇
        memory: ClusterMemory对象，包含已知类别信息
        u_feats: 无标签数据的特征，形状为 [n_samples, feature_dim]
    
    Returns:
        prototype_similarities: 所有prototype与u_feats的相似度矩阵
        prototype_assignments: 字典，记录每个prototype对应的样本索引 {prototype_label: [indices]}
        all_prototypes: 合并后的所有prototype，形状为 [N, 768]
        all_prototype_labels: 所有prototype对应的标签列表
    """
    # 1. 从memory中获取已知类别的prototype和标签
    known_prototypes_dict, known_labels, _ = memory.get_known_class_prototypes()
    
    # 2. 从cluster_list中提取所有新类prototype和标签
    new_prototypes_dict = {}
    for cluster_group in cluster_list:
        for cluster in cluster_group:
            # 假设cluster结构为 [真实标签, 纯度, prototype, 索引列表, 分配的新标签]
            if len(cluster) >= 5:  # 确保有分配的新标签
                new_label = cluster[4]
                new_prototype = cluster[2]  # prototype是第3个元素
                new_prototypes_dict[new_label] = new_prototype
    
    # 3. 合并所有prototype
    all_prototypes = []
    all_prototype_labels = []
    
    # 添加已知类别的prototype
    for label in known_labels:
        all_prototypes.append(known_prototypes_dict[label])
        all_prototype_labels.append(label)
    
    # 添加新类别的prototype
    for label, prototype in new_prototypes_dict.items():
        all_prototypes.append(prototype)
        all_prototype_labels.append(label)
    
    # 转换为tensor
    all_prototypes = torch.stack(all_prototypes)  # 形状 [N, 768]
    
    # 4. 计算相似度
    u_feats_norm = F.normalize(u_feats, p=2, dim=1)  # 归一化u_feats
    all_prototypes_norm = F.normalize(all_prototypes, p=2, dim=1)  # 归一化prototypes
    
    # 计算相似度矩阵
    prototype_similarities = torch.matmul(u_feats_norm, all_prototypes_norm.T)  # 形状 [n_samples, N]
    
    # 5. 找到每个u_feat最相似的prototype
    max_similarities, max_indices = torch.max(prototype_similarities, dim=1)  # 形状 [n_samples]
    
    # 6. 统计每个prototype对应的样本索引
    prototype_assignments = {}
    for i, prototype_idx in enumerate(max_indices):
        prototype_label = all_prototype_labels[prototype_idx]
        
        if prototype_label not in prototype_assignments:
            prototype_assignments[prototype_label] = {
                'count': 0,
                'indices': []
            }
        
        prototype_assignments[prototype_label]['count'] += 1
        prototype_assignments[prototype_label]['indices'].append(i)
    
    # 7. 确保所有prototype都在字典中（即使计数为0）
    for label in all_prototype_labels:
        if label not in prototype_assignments:
            prototype_assignments[label] = {
                'count': 0,
                'indices': []
            }
    
    return prototype_similarities, prototype_assignments, all_prototypes, all_prototype_labels


def purify_cluster_list(cluster_list, prototype_assignments,
                       memory, min_cluster_size=25, absolute_min_size=15, buffer_ratio=0.1):
    """
    净化cluster_list，基于已知类最小相似度和新类簇样本数量
    
    Args:
        cluster_list: 原始的cluster_list
        prototype_assignments: 每个prototype对应的样本分配情况
        memory: ClusterMemory对象，用于判断已知类
        min_cluster_size: 最小簇大小，小于此值且相似度高的新类簇会被过滤
        absolute_min_size: 绝对最小簇大小，小于此值的新类簇直接过滤
        buffer_ratio: 缓冲比例，用于调整相似度阈值
    
    Returns:
        purified_cluster_list: 净化后的cluster_list
        kept_labels: 保留的新类标签集合
        removed_labels: 被移除的新类标签集合
        min_known_similarity: 已知类之间的最小相似度
    """
    # 1. 从cluster_list和memory中构建all_prototypes和all_prototype_labels
    all_prototypes = []
    all_prototype_labels = []
    
    # 从memory中获取已知类别的prototype和标签
    known_prototypes_dict, known_labels, _ = memory.get_known_class_prototypes()
    
    # 添加已知类别的prototype
    for label in known_labels:
        all_prototypes.append(known_prototypes_dict[label])
        all_prototype_labels.append(label)
    
    # 从cluster_list中提取所有新类prototype和标签
    for cluster_group in cluster_list:
        for cluster in cluster_group:
            if len(cluster) >= 5:  # 确保有分配的新标签
                new_label = cluster[4]
                new_prototype = cluster[2]  # prototype是第3个元素
                
                # 只添加新类别的prototype
                if new_label not in all_prototype_labels:
                    all_prototypes.append(new_prototype)
                    all_prototype_labels.append(new_label)
    
    # 转换为tensor
    all_prototypes = torch.stack(all_prototypes)  # 形状 [N, 768]
    
    # 2. 分离已知类和新类
    known_labels_set = set()
    new_labels_set = set()
    
    # 使用memory的mask_init判断已知类
    for label in all_prototype_labels:
        if 0 <= label < len(memory.mask_init) and memory.mask_init[label] == 1:
            known_labels_set.add(label)
        else:
            new_labels_set.add(label)
    
    # 3. 计算已知类之间的相似度
    known_indices = [all_prototype_labels.index(label) for label in known_labels_set if label in all_prototype_labels]
    
    if len(known_indices) > 1:
        known_prototypes = all_prototypes[known_indices]
        known_prototypes_norm = F.normalize(known_prototypes, p=2, dim=1)
        known_similarity_matrix = torch.matmul(known_prototypes_norm, known_prototypes_norm.T)
        
        # 获取已知类之间的最小相似度（不包括对角线）
        mask = torch.eye(len(known_similarity_matrix)) == 0
        min_known_similarity = known_similarity_matrix[mask].min().item()
        
        # 设置相似度阈值：已知类最小相似度减去一个缓冲
        similarity_threshold = min_known_similarity - buffer_ratio * min_known_similarity
    else:
        # 如果只有一个已知类，使用默认阈值
        min_known_similarity = 0.5
        similarity_threshold = 0.45
    
    print(f"已知类数量: {len(known_labels_set)}")
    print(f"已知类之间的最小相似度: {min_known_similarity:.4f}")
    print(f"相似度阈值: {similarity_threshold:.4f}")
    
    # 4. 计算所有prototype之间的相似度矩阵
    all_prototypes_norm = F.normalize(all_prototypes, p=2, dim=1)
    prototype_similarity_matrix = torch.matmul(all_prototypes_norm, all_prototypes_norm.T)
    
    # 5. 筛选新类簇
    kept_labels = set()
    removed_labels = set()
    
    # 创建标签到索引的映射
    label_to_index = {label: idx for idx, label in enumerate(all_prototype_labels)}
    
    # 遍历所有新类标签
    for label in new_labels_set:
        if label not in prototype_assignments:
            removed_labels.add(label)
            continue
            
        count = prototype_assignments[label]['count']
        idx = label_to_index[label]
        
        # 计算该新类簇与所有其他prototype的最大相似度
        max_similarity = 0
        for other_label in all_prototype_labels:
            if other_label == label:
                continue
                
            other_idx = label_to_index[other_label]
            similarity = prototype_similarity_matrix[idx, other_idx].item()
            if similarity > max_similarity:
                max_similarity = similarity
        
        # 判断是否剔除：使用并集方式
        # 条件1: 样本数量小于绝对最小值
        # 条件2: 样本数量小于相对最小值且与其他prototype相似度高
        if count < absolute_min_size or (count < min_cluster_size and max_similarity > similarity_threshold):
            removed_labels.add(label)
            print(f"剔除新类 {label}: 样本数量={count}, 最大相似度={max_similarity:.4f}")
        else:
            kept_labels.add(label)
            print(f"保留新类 {label}: 样本数量={count}, 最大相似度={max_similarity:.4f}")
    
    # 6. 更新cluster_list，只保留筛选后的簇
    purified_cluster_list = []
    
    for cluster_group in cluster_list:
        purified_group = []
        for cluster in cluster_group:
            if len(cluster) >= 5:  # 确保有分配的新标签
                new_label = cluster[4]
                
                # 如果这个标签被保留
                if new_label in kept_labels:
                    purified_group.append(cluster)
        
        purified_cluster_list.append(purified_group)
    
    # 7. 打印筛选结果
    print(f"\n原始新类数量: {len(new_labels_set)}")
    print(f"保留的新类数量: {len(kept_labels)}")
    print(f"移除的新类数量: {len(removed_labels)}")
    
    return purified_cluster_list, kept_labels, removed_labels, min_known_similarity


def filter_clusters_with_memory(cluster_list, memory):
    """
    为cluster_list中的未知簇分配新的类别编号
    
    Args:
        cluster_list: 包含未知簇的列表，每个元素是一个列表，代表与某个已知类相关的未知簇
        memory: ClusterMemory对象，包含已知类别信息
    
    Returns:
        updated_cluster_list: 更新后的cluster_list，每个簇添加了分配的新标签
        cluster_prototypes: 字典，包含所有未知簇的prototype {新标签: prototype}
    """
    # 从memory中获取已知类别信息
    _,known_labels,_ = memory.get_known_class_prototypes()
    
    # 找出所有已使用的标签（已知类别）
    used_labels = set(known_labels)
    
    # 计算总共有多少个未知簇需要分配标签
    total_clusters = 0
    for cluster_group in cluster_list:
        total_clusters += len(cluster_group)
    
    # 生成足够的新标签（按顺序分配未被占用的标签）
    new_labels = []
    label_candidate = 0
    
    while len(new_labels) < total_clusters:
        # 如果当前标签未被使用，则添加到新标签列表
        if label_candidate not in used_labels:
            new_labels.append(label_candidate)
        label_candidate += 1
    
    # 为每个簇分配新标签
    updated_cluster_list = []
    cluster_prototypes = {}
    label_index = 0
    
    for cluster_group in cluster_list:
        updated_group = []
        for cluster in cluster_group:
            if label_index >= len(new_labels):
                break
                
            # 分配新标签
            new_label = new_labels[label_index]
            label_index += 1
            
            # 更新簇信息，添加新标签作为第5个元素
            # 簇结构: [真实标签, 纯度, prototype, 索引列表, 新标签]
            updated_cluster = [
                cluster[0],  # 真实标签
                cluster[1],  # 纯度
                cluster[2],  # prototype
                cluster[3],  # 索引列表
                new_label    # 分配的新标签
            ]
            updated_group.append(updated_cluster)
            
            # 添加到cluster_prototypes字典
            cluster_prototypes[new_label] = cluster[2]  # 存储prototype
        
        updated_cluster_list.append(updated_group)
    
    return updated_cluster_list, cluster_prototypes

def process_clusters(clusters_list):
    """
    处理所有循环中收集的聚类，提取簇的prototype和大小信息
    
    Args:
        clusters_list: 所有循环中收集的聚类列表，每个元素是一个cluster的列表
        known_prototypes: 已知类别的原型，形状为[u_k, 768]
        u_feats: 无标签样本特征，形状为[N, 768]
    
    Returns:
        cluster_prototypes: 所有簇的prototype，形状为[num_clusters, 768]
        cluster_sizes: 每个簇的样本数量，形状为[num_clusters]
        cluster_main_labels: 每个簇的main_label，形状为[num_clusters]
        cluster_high_indices: 每个簇对应的样本indices列表
    """
    cluster_prototypes = []
    cluster_sizes = []
    cluster_main_labels = []
    cluster_high_indices = []
    
    # 遍历所有簇
    for clusters in clusters_list:
        for cluster in clusters:
            if isinstance(cluster, (tuple, list)) and len(cluster) >= 4:
                # 提取簇信息
                main_label = cluster[0]  # 真实类别
                prototype = cluster[2]   # prototype向量
                high_indices = cluster[3] # 样本索引
                
                # 计算簇大小
                if isinstance(high_indices, torch.Tensor):
                    cluster_size = len(high_indices)
                else:
                    cluster_size = len(high_indices)
                
                # 确保prototype是768维向量
                if (isinstance(prototype, torch.Tensor) and 
                    prototype.dim() == 1 and 
                    prototype.size(0) == 768):
                    
                    cluster_prototypes.append(prototype)
                    cluster_sizes.append(cluster_size)
                    cluster_main_labels.append(main_label)
                    cluster_high_indices.append(high_indices)
    
    if len(cluster_prototypes) == 0:
        print("警告: 未找到任何有效的簇")
        return None, None, None, None
    
    cluster_prototypes = torch.stack(cluster_prototypes)
    cluster_sizes = torch.tensor(cluster_sizes, dtype=torch.long)
    cluster_main_labels = torch.tensor(cluster_main_labels, dtype=torch.long)
    
    print(f"找到 {len(cluster_prototypes)} 个有效簇")
    print(f"簇大小统计: 最小={cluster_sizes.min().item()}, 最大={cluster_sizes.max().item()}, 平均={cluster_sizes.float().mean().item():.2f}")
    
    return cluster_prototypes, cluster_sizes, cluster_main_labels, cluster_high_indices


def refine_weak_indices(clusters, weak_indices):
    """
    根据找到的簇进一步筛选weak_indices
    
    Args:
        clusters: 单个簇，包含[真实类别, 纯度, prototype向量, 样本索引列表]
        weak_indices: 原始弱标签索引，形状为[286]的张量
    
    Returns:
        strong_weak_indices: 确定为簇的强标签索引
        remaining_weak_indices: 剩余的弱标签索引
    """
    # 获取簇中的样本索引，并确保是整数列表
    cluster_indices = clusters[3]
    
    # 如果cluster_indices是张量，提取其中的值
    if isinstance(cluster_indices, torch.Tensor):
        # 如果是一维张量，直接转换为列表
        if cluster_indices.dim() == 1:
            cluster_list = cluster_indices.tolist()
        else:
            # 如果是包含标量张量的列表，提取每个张量的值
            cluster_list = [item.item() if isinstance(item, torch.Tensor) else item 
                           for item in cluster_indices]
    else:
        # 如果是普通列表，确保元素是整数
        cluster_list = [item.item() if isinstance(item, torch.Tensor) else item 
                       for item in cluster_indices]
    
    cluster_set = set(cluster_list)
    
    # 将weak_indices转换为整数集合
    if isinstance(weak_indices, torch.Tensor):
        weak_list = weak_indices.tolist()
    else:
        weak_list = weak_indices
    weak_set = set(weak_list)
    
    # 计算交集：在weak_indices中且被分配到簇的样本
    strong_weak_set = weak_set.intersection(cluster_set)
    
    # 剩余weak_indices：在weak_indices中但未被分配到簇的样本
    remaining_weak_set = weak_set - strong_weak_set
    
    # 转换回张量格式
    strong_weak_indices = torch.tensor(list(strong_weak_set), dtype=torch.long)
    remaining_weak_indices = torch.tensor(list(remaining_weak_set), dtype=torch.long)

    
    print(f"原始weak_indices数量: {len(weak_set)}")
    print(f"簇中的样本数量: {len(cluster_set)}")
    print(f"确定为簇的样本数量: {len(strong_weak_indices)}")
    print(f"剩余weak_indices数量: {len(remaining_weak_indices)}")
    
    # 调试信息
    print(f"簇中的前5个索引: {list(cluster_set)[:5]}")
    print(f"weak_set中的前5个索引: {list(weak_set)[:5]}")
    print(f"交集中的索引: {list(strong_weak_set)}")
    
    return strong_weak_indices, remaining_weak_indices


def classify_ufeats_v1(features,u_targets,centers,logits, memory, confidence_threshold=0.7, kl_threshold=2.0):

    # 确保输入是torch张量
    if not isinstance(features, torch.Tensor):
        features = torch.tensor(features, dtype=torch.float32)
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits, dtype=torch.float32)

    known_prototype = memory.get_known_class_prototypes()

    if not isinstance(known_prototypes, torch.Tensor):
        known_prototypes = torch.tensor(known_prototypes, dtype=torch.float32)
    
    n_samples = features.shape[0]
    num_known_classes = known_prototypes.shape[0]

    # 归一化特征和原型
    features_norm = F.normalize(features, p=2, dim=1)  # 形状 [9935, 768]
    prototypes_norm = F.normalize(known_prototypes, p=2, dim=1)  # 形状 [51, 768]

    similarities = torch.matmul(features_norm, prototypes_norm.T)  # 9935x768 * 768x51 = 9935x51
    max_similarities, most_similar_indices = torch.max(similarities, dim=1)  # 两个都是 [9935]

    most_similar_labels = most_similar_indices * 2  # 形状 [9935]

    pre_logits = torch.argmax(logits, dim=1)

    # select 
    known_labels = torch.unique(most_similar_labels)
    mask = torch.zeros(len(max_similarities), dtype=torch.bool)

    # visualize_all_prototypes(features,u_targets,pre_logits,known_prototypes,n_samples=200,save_path="./all_vis.png")
            # 初始化返回变量
    hard_label_mask = torch.zeros(n_samples, dtype=torch.bool, device=features.device)

    updated_u_targets = torch.full((n_samples,), -1, dtype=torch.int64, device=features.device)
    all_new_prototypes = {}

    clusters_list = []
    strong_weak_indices = []

    all_weak = []
    all_weak_strong = []


    for label in known_labels :
        # 找到预测为该类别的所有样本
        
        label_mask = (most_similar_labels == label)

        label_indices = torch.where(label_mask)[0]

        class_u_feats = features[label_indices]
        c_prototype = known_prototypes[int(label/2)]

        # visualize_prototype_distances(class_u_feats,u_targets[label_indices], known_prototypes, prototype_idx=0, n_samples=300)
        strong_indices,weak_indices = find_strong_ufeats(features,pre_logits,label_indices,c_prototype,centers[int(label)],threshold=0.7, prototype_weight=0.5, known_weight=0.5, min_confidence=0.5)
        strong_indices = filter_exclusive_samples(similarities,strong_indices,int(label/2))

        if strong_indices:  # 如果strong_indices非空
            indices_tensor = torch.tensor(strong_indices, device=updated_u_targets.device)
            updated_u_targets[indices_tensor] = label

        acc = evaluate_strong_true_accuracy(strong_indices,u_targets)

        if label ==known_labels[-1]:
            continue

        clusters = find_clusters_from_relations(features,c_prototype,u_targets, label_indices, weak_indices,threshold=0.7, min_cluster_size=6)
        
        if not clusters:
            continue

        for i,cluster in enumerate(clusters):
            strong_weak_indices, weak_indices = refine_weak_indices(cluster, weak_indices)
            all_weak_strong.append(strong_weak_indices)
        all_weak.append(weak_indices)
        
        clusters_list.append(clusters)

        # 找到与当前c_prototype最不相似的聚类
        most_dissimilar_cluster = find_most_dissimilar_cluster(clusters, c_prototype)

        if most_dissimilar_cluster is not None:
            # 标记新类
            hard_label_mask,updated_u_targets, new_prototype = mark_new_class(
                features, most_dissimilar_cluster, label, 
                hard_label_mask, updated_u_targets, label_indices
            )
            
            if new_prototype is not None:
                new_class_label = label + 1
                all_new_prototypes[new_class_label] = new_prototype
                # 从弱样本中分配更多样本到新类
                updated_u_targets = assign_weak_samples_to_new_class(
                    features, weak_indices, c_prototype, new_prototype, 
                    label, hard_label_mask, updated_u_targets, top_k=15
                )
    
        if len(label_indices) > 0:
            # 获取这些样本的相似度
            label_similarities = max_similarities[label_indices]
            
            # 按相似度排序（从高到低）
            sorted_indices = torch.argsort(label_similarities, descending=True)
            
            # 选择前1/3的样本
            top_k = max(1, len(label_indices) // 3)  # 确保至少选择1个样本
            top_label_indices = label_indices[sorted_indices[:top_k]]
            
            # 将这些样本的mask设置为True
            mask[top_label_indices] = True
    
    cluster_prototypes, cluster_sizes, cluster_main_labels, cluster_high_indices = process_clusters(clusters_list)

    
    print(f"总共发现 {len(all_new_prototypes)} 个新类")
    pseudo_labels = torch.zeros(n_samples, dtype=torch.long)
    pseudo_labels = most_similar_labels
    all_weak = torch.cat(all_weak,dim=0)

    return mask, pseudo_labels, similarities, max_similarities,cluster_prototypes,known_prototypes,all_weak,cluster_sizes

def classify_ufeats(features,u_targets,centers,logits, memory, confidence_threshold=0.7, kl_threshold=2.0):

    # 确保输入是torch张量
    if not isinstance(features, torch.Tensor):
        features = torch.tensor(features, dtype=torch.float32)
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits, dtype=torch.float32)
    known_prototype_dict,known_labels_list,_ = memory.get_known_class_prototypes()
    
    known_prototypes = torch.stack([known_prototype_dict[label] for label in known_labels_list])
    known_labels = torch.tensor(known_labels_list, dtype=torch.long)

    n_samples = features.shape[0]
    num_known_classes = known_prototypes.shape[0]

    # 归一化特征和原型
    features_norm = F.normalize(features, p=2, dim=1)  # 形状 [9935, 768]
    prototypes_norm = F.normalize(known_prototypes, p=2, dim=1)  # 形状 [51, 768]

    similarities = torch.matmul(features_norm, prototypes_norm.T)  # 9935x768 * 768x51 = 9935x51
    max_similarities, most_similar_indices = torch.max(similarities, dim=1)  # 两个都是 [9935]

    most_similar_labels = torch.tensor([known_labels_list[idx] for idx in most_similar_indices], device=features.device)

    pre_logits = torch.argmax(logits, dim=1)

    # select 
    mask = torch.zeros(len(max_similarities), dtype=torch.bool)

    clusters_list = []
    strong_weak_indices = []
    all_weak = []
    all_weak_strong = []

    # 创建标签到索引的映射，方便后续查找
    label_to_index = {label: idx for idx, label in enumerate(known_labels_list)}

    for label in known_labels:
        # 找到预测为该类别的所有样本
        
        label_mask = (most_similar_labels == label)

        label_indices = torch.where(label_mask)[0]

        class_u_feats = features[label_indices]

        # 使用标签到索引的映射获取对应的prototype
        prototype_idx = label_to_index[int(label)]
        c_prototype = known_prototypes[prototype_idx]

        # visualize_prototype_distances(class_u_feats,u_targets[label_indices], known_prototypes, prototype_idx=0, n_samples=300)
        strong_indices,weak_indices = find_strong_ufeats(features,pre_logits,label_indices,c_prototype,centers[int(label)],threshold=0.7, prototype_weight=0.5, known_weight=0.5, min_confidence=0.5)
        strong_indices = filter_exclusive_samples(similarities,strong_indices,prototype_idx)

        # acc = evaluate_strong_true_accuracy(strong_indices,u_targets)
        if label ==known_labels[-1]:
            continue

        clusters = find_clusters_from_relations(features,c_prototype,u_targets, label_indices, weak_indices,threshold=0.7, min_cluster_size=6)
        if not clusters:
            continue

        for i,cluster in enumerate(clusters):
            strong_weak_indices, weak_indices = refine_weak_indices(cluster, weak_indices)
            all_weak_strong.append(strong_weak_indices)
        all_weak.append(weak_indices)
        
        clusters_list.append(clusters)
        
        if len(label_indices) > 0:
            label_similarities = max_similarities[label_indices]
            sorted_indices = torch.argsort(label_similarities, descending=True)
        
            top_k = max(1, len(label_indices) // 3)  # 确保至少选择1个样本
            top_label_indices = label_indices[sorted_indices[:top_k]]
            
            mask[top_label_indices] = True
    
    all_weak = torch.cat(all_weak,dim=0)

    return mask,most_similar_labels,clusters_list,all_weak

def classify_ufeats_tube(features,u_targets,centers,logits, memory, confidence_threshold=0.7, kl_threshold=2.0):

    # 确保输入是torch张量
    if not isinstance(features, torch.Tensor):
        features = torch.tensor(features, dtype=torch.float32)
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits, dtype=torch.float32)
    known_prototype_dict,known_labels_list,_ = memory.get_known_class_prototypes()
    
    known_prototypes = torch.stack([known_prototype_dict[label] for label in known_labels_list])
    known_labels = torch.tensor(known_labels_list, dtype=torch.long)

    n_samples = features.shape[0]
    num_known_classes = known_prototypes.shape[0]

    # 归一化特征和原型
    features_norm = F.normalize(features, p=2, dim=1)  # 形状 [9935, 768]
    prototypes_norm = F.normalize(known_prototypes, p=2, dim=1)  # 形状 [51, 768]

    similarities = torch.matmul(features_norm, prototypes_norm.T)  # 9935x768 * 768x51 = 9935x51
    max_similarities, most_similar_indices = torch.max(similarities, dim=1)  # 两个都是 [9935]

    most_similar_labels = torch.tensor([known_labels_list[idx] for idx in most_similar_indices], device=features.device)

    pre_logits = torch.argmax(logits, dim=1)

    # select 
    mask = torch.zeros(len(max_similarities), dtype=torch.bool)

    clusters_list = []
    strong_weak_indices = []
    all_weak = []
    all_weak_strong = []

    # 创建标签到索引的映射，方便后续查找
    label_to_index = {label: idx for idx, label in enumerate(known_labels_list)}

    for label in known_labels:
        # 找到预测为该类别的所有样本
        
        label_mask = (most_similar_labels == label)

        label_indices = torch.where(label_mask)[0]

        class_u_feats = features[label_indices]

        # 使用标签到索引的映射获取对应的prototype
        prototype_idx = label_to_index[int(label)]
        c_prototype = known_prototypes[prototype_idx]

        # visualize_prototype_distances(class_u_feats,u_targets[label_indices], known_prototypes, prototype_idx=0, n_samples=300)
        if label ==known_labels[-1]:
            continue
        
        if len(label_indices) > 0:
            label_similarities = max_similarities[label_indices]
            sorted_indices = torch.argsort(label_similarities, descending=True)
        
            top_k = max(1, len(label_indices) // 3)  # 确保至少选择1个样本
            top_label_indices = label_indices[sorted_indices[:top_k]]
            mask[top_label_indices] = True
    
    return mask,most_similar_labels,None,None


def extract_detailed_known_unknown_info(features, logits,known_prototypes, args):
    """
    从模型中提取特征和logits，然后详细分类已知类和未知类
    """
    # 分类已知类和未知类
    mask, pseudo_labels, confidence_scores, kl_scores = classify_known_unknown_detailed(
        features, logits, known_prototypes,
        confidence_threshold=0.6,
        kl_threshold=0.6
    )
    
    return mask, pseudo_labels, confidence_scores, kl_scores


def only_extract_labeled_stprotos(model, train_loader, memory,merge_len,args,cfg,local):
    
    model.eval()

    all_feats = []
    all_logits = []
    s_all_feats = []
    t_all_feats = []
    targets = np.array([])
    mask = np.array([])
    ids=np.array([])
    mask_cls=np.array([])
    metrics=dict()
    with torch.no_grad():
        for batch_idx, _item in enumerate(train_loader):

            images = _item[0]

            label = _item[2]
            uq_idx = _item[3]
            mask_lab_ = _item[4]
            pseu_idx = _item[5]
            
            images = images.cuda(local,non_blocking=True)
            label, mask_lab_ = label.cuda(local,non_blocking=True), mask_lab_.cuda(local,non_blocking=True).bool()

            # Pass features through base model and then additional learnable transform (linear layer)
            outputs = model(images)
            feats = outputs[0].data.cpu() #sp_feature
            logits = outputs[1].data.cpu()
            t_feats = outputs[2].data.cpu() #t_feature
            s_feats = outputs[3].data.cpu() #s_feature
            
            all_feats.append(feats.numpy())
            all_logits.append(logits.numpy())

            s_all_feats.append(s_feats.numpy())
            t_all_feats.append(t_feats.numpy())
        
            targets = np.append(targets, label.cpu().numpy())
            ids=np.append(ids,pseu_idx.cpu().numpy())
            mask = np.append(mask, mask_lab_.cpu().bool().numpy())
            # mask_cls = np.append(mask_cls,np.array([True if x.item() in range(len(args.train_classes))
            #                                 else False for x in label]))
    mask = mask.astype(bool)
    
    if args.dataset_name == 'vb100':
        new_targets = targets
    else:
        new_targets = convert_labels(targets,args.num_labeled_classes + args.num_unlabeled_classes)

    mask_cls = np.array([True if x.item() in args.train_classes
                     else False for x in new_targets])
    
    mask_cls = mask_cls.astype(bool)

    all_feats = np.concatenate(all_feats)
    all_logits = np.concatenate(all_logits)
    s_all_feats = np.concatenate(s_all_feats)
    t_all_feats = np.concatenate(t_all_feats)

    
    torch.save({
        'all_feats': torch.from_numpy(all_feats),
        'all_logits':torch.from_numpy(all_logits),
        's_all_feats': torch.from_numpy(s_all_feats),
        't_all_feats': torch.from_numpy(t_all_feats),
        'targets': torch.from_numpy(targets),
        'ids': torch.from_numpy(ids),
        'mask': torch.from_numpy(mask),
        'new_targets': torch.from_numpy(new_targets),
        'mask_cls': torch.from_numpy(mask_cls)
    }, cfg.logit_dir)

    print("logit success!")


def extract_labeled_stprotos(model, train_loader, memory,merge_len,args,local):
    
    model.eval()

    all_feats = []
    all_logits = []
    s_all_feats = []
    t_all_feats = []
    targets = np.array([])
    mask = np.array([])
    ids=np.array([])
    mask_cls=np.array([])
    metrics=dict()
    with torch.no_grad():
        for batch_idx, _item in enumerate(train_loader):

            images = _item[0]

            label = _item[2]
            uq_idx = _item[3]
            mask_lab_ = _item[4]
            pseu_idx = _item[5]
            
            images = images.cuda(local,non_blocking=True)
            label, mask_lab_ = label.cuda(local,non_blocking=True), mask_lab_.cuda(local,non_blocking=True).bool()

            # Pass features through base model and then additional learnable transform (linear layer)
            outputs = model(images)
            feats = outputs[0].data.cpu() #sp_feature
            logits = outputs[1].data.cpu()
            t_feats = outputs[2].data.cpu() #t_feature
            s_feats = outputs[3].data.cpu() #s_feature
            
            all_feats.append(feats.numpy())
            all_logits.append(logits.numpy())

            s_all_feats.append(s_feats.numpy())
            t_all_feats.append(t_feats.numpy())
        
            targets = np.append(targets, label.cpu().numpy())
            ids=np.append(ids,pseu_idx.cpu().numpy())
            mask = np.append(mask, mask_lab_.cpu().bool().numpy())
            # mask_cls = np.append(mask_cls,np.array([True if x.item() in range(len(args.train_classes))
            #                                 else False for x in label]))
    mask = mask.astype(bool)
    
    if args.dataset_name == 'vb100':
        new_targets = targets
    else:
        new_targets = convert_labels(targets,args.num_labeled_classes + args.num_unlabeled_classes)

    mask_cls = np.array([True if x.item() in args.train_classes
                     else False for x in new_targets])
    
    mask_cls = mask_cls.astype(bool)

    all_feats = np.concatenate(all_feats)
    all_logits = np.concatenate(all_logits)
    s_all_feats = np.concatenate(s_all_feats)
    t_all_feats = np.concatenate(t_all_feats)

    
    torch.save({
        'all_feats': torch.from_numpy(all_feats),
        'all_logits':torch.from_numpy(all_logits),
        's_all_feats': torch.from_numpy(s_all_feats),
        't_all_feats': torch.from_numpy(t_all_feats),
        'targets': torch.from_numpy(targets),
        'ids': torch.from_numpy(ids),
        'mask': torch.from_numpy(mask),
        'new_targets': torch.from_numpy(new_targets),
        'mask_cls': torch.from_numpy(mask_cls)
    }, args.logit_dir)

    feature_data = load_features_data(args.logit_dir)

    all_feats = feature_data['all_feats']
    all_logits = feature_data['all_logits']
    s_all_feats = feature_data['s_all_feats']
    t_all_feats = feature_data['t_all_feats']
    targets = feature_data['targets']
    ids = feature_data['ids']
    mask = feature_data['mask']
    new_targets = feature_data['new_targets']
    mask_cls = feature_data['mask_cls']
   
    l_feats = all_feats[mask]  # Get labelled set
    u_feats = all_feats[~mask]  # Get unlabelled set
    l_targets = new_targets[mask]  # Get labelled targets
    u_targets = new_targets[~mask]  # Get unlabelled targets
    n_samples =len(targets)

    s_l_feats = s_all_feats[mask] 
    s_u_feats = s_all_feats[~mask] 

    t_l_feats = t_all_feats[mask] 
    t_u_feats = t_all_feats[~mask]

    cluster_size=math.ceil(n_samples /(args.num_labeled_classes + args.num_unlabeled_classes))

    kmeanssem = SemiSupKMeans(k=args.num_labeled_classes + args.num_unlabeled_classes, tolerance=1e-4,
                              max_iterations=10, init='k-means++',
                              n_init=1, random_state=None, n_jobs=None, pairwise_batch_size=1024,
                              mode=None, protos=None,cluster_size=cluster_size)

    s_kmeanssem = SemiSupKMeans(k=args.num_labeled_classes + args.num_unlabeled_classes, tolerance=1e-4,
                              max_iterations=10, init='k-means++',
                              n_init=1, random_state=None, n_jobs=None, pairwise_batch_size=1024,
                              mode=None, protos=None,cluster_size=cluster_size)

    t_kmeanssem = SemiSupKMeans(k=args.num_labeled_classes + args.num_unlabeled_classes, tolerance=1e-4,
                              max_iterations=10, init='k-means++',
                              n_init=1, random_state=None, n_jobs=None, pairwise_batch_size=1024,
                              mode=None, protos=None,cluster_size=cluster_size)

    l_feats, u_feats, l_targets, u_targets = (x.cuda(local,non_blocking=True) for
                                              x in (l_feats, u_feats, l_targets, u_targets))

    s_l_feats, s_u_feats, t_l_feats, t_u_feats = (x.cuda(local,non_blocking=True) for
                                              x in (s_l_feats, s_u_feats, t_l_feats, t_u_feats))

    kmeanssem.fit_mix(u_feats, l_feats, l_targets)
    s_kmeanssem.fit_mix(s_u_feats, s_l_feats, l_targets)
    t_kmeanssem.fit_mix(t_u_feats, t_l_feats, l_targets)

    all_preds = kmeanssem.labels_
    mask_cls=mask_cls[~mask]
    preds = all_preds.cpu().numpy()[~mask]

    s_all_preds = s_kmeanssem.labels_
    s_preds = s_all_preds.cpu().numpy()[~mask]
    t_all_preds = t_kmeanssem.labels_
    t_preds = t_all_preds.cpu().numpy()[~mask]
    
    all_acc, old_acc, new_acc = split_cluster_acc_v2_kmeans(y_true=u_targets.cpu().numpy(), y_pred=preds, mask=mask_cls)
    print(all_acc,old_acc,new_acc)

    s_all_acc, s_old_acc, s_new_acc = split_cluster_acc_v2_kmeans(y_true=u_targets.cpu().numpy(), y_pred=s_preds, mask=mask_cls)
    print(s_all_acc,s_old_acc,s_new_acc)

    t_all_acc, t_old_acc, t_new_acc = split_cluster_acc_v2_kmeans(y_true=u_targets.cpu().numpy(), y_pred=t_preds, mask=mask_cls)
    print(t_all_acc,t_old_acc,t_new_acc)

    metrics["all_acc"], metrics["old_acc"], metrics["new_acc"] = all_acc, old_acc, new_acc
    
    prototype_higher=[]
    prototypes = kmeanssem.cluster_centers_
    s_prototypes = s_kmeanssem.cluster_centers_
    t_prototypes = t_kmeanssem.cluster_centers_

    prototype_higher.append(prototypes.clone())
    prototype_higher.append(s_prototypes.clone())
    prototype_higher.append(t_prototypes.clone())

    n_labeled=args.num_labeled_classes
    n_novel= args.num_unlabeled_classes
    label_proto = prototypes.cpu().numpy()[:args.num_labeled_classes,:]
    preds_higher=[]

    preds_higher.append(all_preds.clone())
    preds_higher.append(s_all_preds.clone())
    preds_higher.append(t_all_preds.clone())

    print('Hierarchy clustering')
    mask_known=(all_preds<args.num_labeled_classes).cpu().numpy()
    l_feats = all_feats[mask_known]  # Get labelled set
    u_feats = all_feats[~mask_known]
    l_feats, u_feats= (x.cuda(local,non_blocking=True) for  x in (l_feats, u_feats))

    while n_labeled>1:
        n_labeled=max(int(n_labeled/2),1)
        n_novel=max(int(n_novel/2),1)

        kmeans_l = KMeans(n_clusters=n_labeled, random_state=0).fit(label_proto)
        preds_labels = torch.from_numpy(kmeans_l.labels_).cuda(local,non_blocking=True)
        level_l_targets=preds_labels[all_preds[mask_known]]
        if args.unbalanced:
            cluster_size = None
        else:
            cluster_size = math.ceil( n_samples / (n_labeled+n_novel))
        kmeans_higher =SemiSupKMeans(k=n_labeled+n_novel, tolerance=1e-4,
                              max_iterations=10, init='k-means++',
                              n_init=1, random_state=None, n_jobs=None, pairwise_batch_size=1024,
                              mode=None, protos=None,cluster_size=cluster_size)
        kmeans_higher.fit_mix(u_feats, l_feats, level_l_targets)
        preds_level = kmeans_higher.labels_
        prototypes_level = kmeans_higher.cluster_centers_
        prototype_higher.append(prototypes_level.clone())
        preds_higher.append(preds_level.cuda(local,non_blocking=True).clone())
        
    return ids,all_preds, prototype_higher,preds_higher


def extract_labeled_stprotos_simple(model, train_loader, memory,centers, merge_len,args,local):
    metrics=dict()

    feature_data = load_features_data(args.logit_dir)

    all_feats = feature_data['all_feats']
    all_logits = feature_data['all_logits']
    s_all_feats = feature_data['s_all_feats']
    t_all_feats = feature_data['t_all_feats']
    targets = feature_data['targets']
    ids = feature_data['ids']
    mask = feature_data['mask']
    new_targets = feature_data['new_targets']
    mask_cls = feature_data['mask_cls']

    k_memory = memory.features[::2]
    
    l_feats = all_feats[mask]  # Get labelled set
    u_feats = all_feats[~mask]  # Get unlabelled set
    l_targets = new_targets[mask]  # Get labelled targets
    u_targets = new_targets[~mask]  # Get unlabelled targets
    n_samples =len(targets)
    all_u_logits = all_logits[~mask]
    true_u_targets = targets[~mask]

    edge_memory = memory.boundary[::2]
    # mask_old_new, pse_labels, predicted_sim, pre_logits = classify_known_unknown_detailed(u_feats, all_u_logits, k_memory, edge_memory)

    # analyze_predictions_detailed(true_u_targets,mask_old_new,pse_labels,predicted_sim,'./analysis_prptotype1007_s_weighte.xlsx')

    # result_df = analyze_cluster_target_relation(true_u_targets[mask_old_new].cpu().numpy(),pse_labels[mask_old_new].cpu().numpy())
    # 保存到Excel
    # save_to_excel(result_df)
    
    s_l_feats = s_all_feats[mask] 
    s_u_feats = s_all_feats[~mask] 

    t_l_feats = t_all_feats[mask] 
    t_u_feats = t_all_feats[~mask]

    cluster_size=math.ceil(n_samples /(args.num_labeled_classes + args.num_unlabeled_classes))

    kmeanssem = SemiSupKMeans(k=args.num_labeled_classes + args.num_unlabeled_classes, tolerance=1e-4,
                              max_iterations=10, init='k-means++',
                              n_init=1, random_state=None, n_jobs=None, pairwise_batch_size=1024,
                              mode=None, protos=None,cluster_size=cluster_size)

    s_kmeanssem = SemiSupKMeans(k=args.num_labeled_classes + args.num_unlabeled_classes, tolerance=1e-4,
                              max_iterations=10, init='k-means++',
                              n_init=1, random_state=None, n_jobs=None, pairwise_batch_size=1024,
                              mode=None, protos=None,cluster_size=cluster_size)

    t_kmeanssem = SemiSupKMeans(k=args.num_labeled_classes + args.num_unlabeled_classes, tolerance=1e-4,
                              max_iterations=10, init='k-means++',
                              n_init=1, random_state=None, n_jobs=None, pairwise_batch_size=1024,
                              mode=None, protos=None,cluster_size=cluster_size)

    l_feats, u_feats, l_targets, u_targets = (x.cuda(local,non_blocking=True) for
                                              x in (l_feats, u_feats, l_targets, u_targets))

    s_l_feats, s_u_feats, t_l_feats, t_u_feats = (x.cuda(local,non_blocking=True) for
                                              x in (s_l_feats, s_u_feats, t_l_feats, t_u_feats))

    kmeanssem.fit_mix(u_feats, l_feats, l_targets)
    s_kmeanssem.fit_mix(s_u_feats, s_l_feats, l_targets)
    t_kmeanssem.fit_mix(t_u_feats, t_l_feats, l_targets)

    all_preds = kmeanssem.labels_
    mask_cls=mask_cls[~mask]
    preds = all_preds.cpu().numpy()[~mask]

    s_all_preds = s_kmeanssem.labels_
    s_preds = s_all_preds.cpu().numpy()[~mask]
    t_all_preds = t_kmeanssem.labels_
    t_preds = t_all_preds.cpu().numpy()[~mask]
    
    all_acc, old_acc, new_acc = split_cluster_acc_v2_kmeans(y_true=u_targets.cpu().numpy(), y_pred=preds, mask=mask_cls)
    print(all_acc,old_acc,new_acc)

    s_all_acc, s_old_acc, s_new_acc = split_cluster_acc_v2_kmeans(y_true=u_targets.cpu().numpy(), y_pred=s_preds, mask=mask_cls)
    print(s_all_acc,s_old_acc,s_new_acc)

    t_all_acc, t_old_acc, t_new_acc = split_cluster_acc_v2_kmeans(y_true=u_targets.cpu().numpy(), y_pred=t_preds, mask=mask_cls)
    print(t_all_acc,t_old_acc,t_new_acc)

    metrics["all_acc"], metrics["old_acc"], metrics["new_acc"] = all_acc, old_acc, new_acc
    
    prototype_higher=[]
    prototypes = kmeanssem.cluster_centers_
    s_prototypes = s_kmeanssem.cluster_centers_
    t_prototypes = t_kmeanssem.cluster_centers_

    prototype_higher.append(prototypes.clone())
    prototype_higher.append(s_prototypes.clone())
    prototype_higher.append(t_prototypes.clone())

    n_labeled=args.num_labeled_classes
    n_novel= args.num_unlabeled_classes
    label_proto = prototypes.cpu().numpy()[:args.num_labeled_classes,:]
    preds_higher=[]

    preds_higher.append(all_preds.clone())
    preds_higher.append(s_all_preds.clone())
    preds_higher.append(t_all_preds.clone())

    print('Hierarchy clustering')
    mask_known=(all_preds<args.num_labeled_classes).cpu().numpy()
    l_feats_k = all_feats[mask_known]  # Get labelled set
    u_feats_k = all_feats[~mask_known]
    l_feats_k, u_feats_k= (x.cuda(local,non_blocking=True) for  x in (l_feats_k, u_feats_k))

    while n_labeled>1:
        n_labeled=max(int(n_labeled/2),1)
        n_novel=max(int(n_novel/2),1)

        kmeans_l = KMeans(n_clusters=n_labeled, random_state=0).fit(label_proto)
        preds_labels = torch.from_numpy(kmeans_l.labels_).cuda(local,non_blocking=True)
        level_l_targets=preds_labels[all_preds[mask_known]]
        if args.unbalanced:
            cluster_size = None
        else:
            cluster_size = math.ceil( n_samples / (n_labeled+n_novel))
        kmeans_higher =SemiSupKMeans(k=n_labeled+n_novel, tolerance=1e-4,
                              max_iterations=10, init='k-means++',
                              n_init=1, random_state=None, n_jobs=None, pairwise_batch_size=1024,
                              mode=None, protos=None,cluster_size=cluster_size)
        kmeans_higher.fit_mix(u_feats_k, l_feats_k, level_l_targets)
        preds_level = kmeans_higher.labels_ 
        prototypes_level = kmeans_higher.cluster_centers_
        prototype_higher.append(prototypes_level.clone())
        preds_higher.append(preds_level.cuda(local,non_blocking=True).clone())
        
    return ids,all_preds, prototype_higher,preds_higher

def extract_labeled_stprotos_energy(model, train_loader, memory,centers, merge_len,args,cfg,local):
    metrics=dict()

    feature_data = load_features_data(cfg.logit_dir)

    all_feats = feature_data['all_feats']
    all_logits = feature_data['all_logits']
    s_all_feats = feature_data['s_all_feats']
    t_all_feats = feature_data['t_all_feats']
    targets = feature_data['targets']
    ids = feature_data['ids']
    mask = feature_data['mask']
    new_targets = feature_data['new_targets']
    mask_cls = feature_data['mask_cls']

    
    l_feats = all_feats[mask]  # Get labelled set
    u_feats = all_feats[~mask]  # Get unlabelled set
    l_targets = new_targets[mask]  # Get labelled targets
    u_targets = new_targets[~mask]  # Get unlabelled targets
    n_samples =len(targets)
    all_u_logits = all_logits[~mask]
    true_u_targets = targets[~mask]
 
    mask_old_new, pse_labels,cluster_list,all_weak = classify_ufeats_tube(u_feats,true_u_targets,centers,all_u_logits, memory)
    
    #analysis cluster
    # cluster_prototypes, cluster_sizes, cluster_main_labels, cluster_high_indices = process_clusters(clusters_list)

    return mask_old_new, pse_labels,cluster_list,all_weak,u_feats

    
def add_gaussian_noise(features, sigma=0.25):
    noise = torch.randn_like(features) * sigma
    return 0.5*features + 0.5*noise

def feature_dropout(features, p=0.3):
    return F.dropout(features, p=p, training=True)

def mixup_features(features, alpha=0.2):
    lam = torch.distributions.Beta(alpha, alpha).sample((features.shape[0],)).to(features.device)
    lam = lam.view(-1, 1)  # 维度对齐
    index = torch.randperm(features.shape[0]).to(features.device)  # 生成混合索引
    return lam * features + (1 - lam) * features[index]

class kmeansMLPProjector(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=1024, out_dim=768):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def update_prototype(backbone,memory,class_meter,args,cfg):
    
    if cfg.NUM_GPUS>1:
        local_rank = args.local_rank
    else:
        local_rank = 0

    pred_feat = memory.all_feature
    pred_feature = pred_feat.cuda(local_rank,non_blocking=True)

    batch_size = 128  # 可选64或128
    results = []

    with torch.no_grad():  # 禁用梯度计算
        for i in range(0, len(pred_feature), batch_size):
            batch = pred_feature[i:i+batch_size]  # 自动处理最后不足batch_size的情况

            pred = backbone.module.get_head(batch)
            results.append(pred.cpu())


    preds = torch.cat(results, dim=0)

    class_meter.reset_pred(preds)

    preds,label,mask = class_meter.finalize_preds()
    
    pred_list = np.array([m.numpy() for m in preds])


    cluster_features = generate_cluster_features(pred_list, pred_feature)

    labelled_len = len(cluster_features)


    unlabelled_len = cfg.MODEL.NUM_CLASSES - labelled_len

    vectors = torch.randn(unlabelled_len,args.feat_dim)
    nor_vectors = F.normalize(vectors, p=2, dim=1)

    class_feature = torch.cat((cluster_features.cpu(),nor_vectors),dim=0)

    cls_feat = backbone.get_head(class_feature)

    memory.features = F.normalize(class_feature, dim=1)
    
    memory.cls_feature = cls_feat

    return memory


def class_vector_sup(backbone,projector,memory,train_loader,test_meter,args,cfg):
    if cfg.NUM_GPUS>1:
        local_rank = args.local_rank
    else:
        local_rank =0
    

    pred,label_true,all_feature,cls_list, s_all_features,t_all_features = extract_features(backbone,train_loader,test_meter,args, print_freq=50, local_rank=local_rank)

    pred,label,mask = test_meter.finalize_preds()
    
    pred_list = np.array([m.numpy() for m in pred])
    label_list = np.array([m.numpy() for m in label])

    cluster_feature,centers,c_boundary = generate_cluster_boundary(label_list, all_feature,cfg.MODEL.NUM_CLASSES)
    s_cluster_feature,_,s_boundary = generate_cluster_boundary(label_list, s_all_features,cfg.MODEL.NUM_CLASSES)
    t_cluster_feature,_,t_boundary = generate_cluster_boundary(label_list, t_all_features,cfg.MODEL.NUM_CLASSES)

    cls_feat= generate_cluster_features(label_list, cls_list,cfg.MODEL.NUM_CLASSES)

    memory.features = F.normalize(cluster_feature, dim=1)
    memory.s_features = F.normalize(s_cluster_feature, dim=1)
    memory.t_features = F.normalize(t_cluster_feature, dim=1)
    memory.boundary = F.normalize(c_boundary, dim=-1)

    memory.cls_feature = cls_feat
    
    memory.all_feature = all_feature

    save_data = {
        # 保存memory的特定组件
        'features': memory.features,
        's_features': memory.s_features,
        't_features': memory.t_features,
        'cls_feature': memory.cls_feature,
        'boundary': memory.boundary,
        'all_feature': memory.all_feature,
        
        # 保存centers
        'centers': centers}

    torch.save(save_data,cfg.memory_dir)

    return memory,centers


def class_vector(memory,cfg):
    save_data = torch.load(cfg.memory_dir, map_location='cpu')
    # 如果提供了memory实例，更新它
    if memory is not None:
        memory.features = save_data['features']
        memory.s_features = save_data['s_features']
        memory.t_features = save_data['t_features']
        memory.cls_feature = save_data['cls_feature']
        memory.boundary = save_data['boundary']
        memory.all_feature = save_data['all_feature']
        
        centers = save_data['centers']
        
        print(f"Memory components loaded from: {cfg.memory_dir}")
        
        return memory, centers
    else:
        # 直接返回数据字典
        print(f"Don't Memory components loaded from: {cfg.memory_dir}")
        return memory,None


def unsupervised_prototype(u_feats,pse_labels,mask_old_new,memory, args):
    unsup = u_feats[~mask_old_new]
    unsup_pse_label = pse_labels[~mask_old_new]

    # 找出所有偶数伪标签类别
    even_labels = torch.unique(unsup_pse_label)
    even_labels = even_labels[even_labels % 2 == 0]  # 只保留偶数标签
    
    new_prototypes = {}
    
    for even_label in even_labels:
    
        mask = (unsup_pse_label == even_label)
        class_samples = unsup[mask]
        
        if len(class_samples) > 0:
            # 计算该类别的原型
            prototype = torch.mean(class_samples, dim=0)
            odd_label = even_label + 1
            if odd_label < (args.num_labeled_classes +args.num_unlabeled_classes):
                new_prototypes[odd_label] = prototype
    
    # 更新 memory 中的 features
    for odd_label, prototype in new_prototypes.items():
        if odd_label < memory.features.shape[0]:  # 确保索引不越界
            memory.features[odd_label] = F.normalize(prototype, dim=0)
    
    # 生成转换后的伪标签和掩码
    converted_pse_labels = pse_labels  # 初始化为-1
    unsup_mask = torch.zeros_like(pse_labels, dtype=torch.bool)
    
    for even_label in even_labels:
        # 找到原始伪标签为该偶数的样本
        mask = (pse_labels == even_label) & (~mask_old_new)
        
        if mask.sum() > 0:
            # 映射到奇数标签
            odd_label = even_label + 1
            if odd_label < (args.num_labeled_classes +args.num_unlabeled_classes):
                converted_pse_labels[mask] = odd_label
                unsup_mask[mask] = True
    
    min_val, max_val = converted_pse_labels.min(), converted_pse_labels.max()
    
    return memory, converted_pse_labels, unsup_mask

def compute_prototype_similarities_with_mapping(all_prototypes_dict):
    """
    计算非连续类别标签的prototype相似度矩阵，并建立索引映射
    
    Args:
        all_prototypes_dict: 字典，{标签: prototype}，标签是非连续且非顺序的
        u_feats: 无标签数据的特征，形状为 [n_samples, feature_dim]
    
    Returns:
        prototype_similarities: 相似度矩阵，形状为 [n_samples, num_prototypes]
        index_to_label: 映射字典，{矩阵索引: 实际标签}
        label_to_index: 映射字典，{实际标签: 矩阵索引}
        sorted_labels: 排序后的标签列表，与矩阵列顺序对应
        all_prototypes_tensor: 所有prototype的tensor [num_prototypes, feature_dim]
    """
    # 1. 创建索引映射
    sorted_labels = sorted(all_prototypes_dict.keys())
    num_prototypes = len(sorted_labels)
    
    # 创建双向映射字典
    index_to_label = {idx: label for idx, label in enumerate(sorted_labels)}
    label_to_index = {label: idx for idx, label in enumerate(sorted_labels)}
    
    print(f"类别数量: {num_prototypes}")
    print(f"类别标签: {sorted_labels}")
    print(f"索引到标签映射: {index_to_label}")
    print(f"标签到索引映射: {label_to_index}")
    
    # 2. 按照排序后的标签顺序构建prototype tensor
    all_prototypes_tensor = torch.stack([all_prototypes_dict[label] for label in sorted_labels])
    
    # 3. 归一化特征和原型
    all_prototypes_norm = F.normalize(all_prototypes_tensor, p=2, dim=1)
    
    # 4. 计算相似度矩阵
    prototype_similarities = torch.matmul(all_prototypes_norm, all_prototypes_norm.T)  # 形状 [n_samples, num_prototypes]
    
    return prototype_similarities,label_to_index 


def assign_pseudo_labels_with_mapping(all_prototypes_dict, u_feats):
    """
    使用非连续类别标签为u_feats分配伪标签
    
    Args:
        all_prototypes_dict: 字典，{标签: prototype}，标签是非连续且非顺序的
        u_feats: 无标签数据的特征，形状为 [n_samples, feature_dim]
    
    Returns:
        pseudo_labels: 每个u_feats对应的伪标签，形状为 [n_samples]
        prototype_similarities: 相似度矩阵，形状为 [n_samples, num_prototypes]
        max_similarities: 每个样本的最大相似度，形状为 [n_samples]
        index_to_label: 映射字典，{矩阵索引: 实际标签}
        label_to_index: 映射字典，{实际标签: 矩阵索引}
    """
    # 计算相似度矩阵和映射
    prototype_similarities, index_to_label, label_to_index, sorted_labels, all_prototypes_tensor = \
        compute_prototype_similarities_with_mapping(all_prototypes_dict, u_feats)
    
    # 为每个u_feat找到最相似的prototype
    max_similarities, max_indices = torch.max(prototype_similarities, dim=1)  # 形状 [n_samples]
    
    # 将prototype索引转换为对应的实际标签
    pseudo_labels = torch.tensor([index_to_label[idx.item()] for idx in max_indices], 
                                device=u_feats.device, dtype=torch.long)
    
    # 打印分配统计
    unique_labels, counts = torch.unique(pseudo_labels, return_counts=True)
    print("\n伪标签分配统计:")
    for label, count in zip(unique_labels, counts):
        print(f"标签 {label.item()}: {count.item()} 个样本")
    
    return pseudo_labels, prototype_similarities, max_similarities, index_to_label, label_to_index




def update_teacher_output(student_out, mask_lab, uq_idx, soft_labels):
    """
    更新teacher输出，将有标签样本保留原值，无标签样本使用软标签
    
    Args:
        student_out: 学生模型的输出，形状为 [batch_size, num_classes]
        mask_lab: 布尔张量，标记哪些样本有标签，形状为 [batch_size]
        uq_idx: 样本的全局索引，形状为 [batch_size]
        soft_labels: 预先计算好的软标签，形状为 [dataset_size, num_classes]
    
    Returns:
        teacher_out: 更新后的教师输出
    """
    # 初始化teacher_out为学生输出的detach版本
    teacher_out = student_out.detach().clone()
    
    # 确保soft_labels在正确的设备上
    soft_labels = soft_labels.to(teacher_out.device)
    
    # 获取无标签样本的掩码
    mask_unlab = ~mask_lab
    
    # 如果有无标签样本
    if mask_unlab.any():
        # 获取无标签样本的全局索引
        unlab_uq_idx = uq_idx[mask_unlab]
        
        # 检查索引是否在有效范围内
        valid_indices = (unlab_uq_idx >= 0) & (unlab_uq_idx < len(soft_labels))
        
        if valid_indices.any():
            # 从soft_labels中获取对应的软标签
            unlab_soft_labels = soft_labels[unlab_uq_idx[valid_indices]]
            
            # 将teacher_out中无标签样本的部分替换为软标签
            teacher_out[mask_unlab][valid_indices] = unlab_soft_labels
    
    return teacher_out


def convert_train_label(mask_old_new, pse_label,train_loader):
    l_mask = len(train_loader.dataset.labelled_dataset.uq_idxs)
    l_mask = torch.ones(l_mask, dtype=torch.bool, device=mask_old_new.device)
    mask = torch.cat([l_mask,mask_old_new],dim=0)
    

    l_label = train_loader.dataset.labelled_dataset._labels
    if not isinstance(l_label, torch.Tensor):
        l_label = torch.tensor(l_label, device=pse_label.device)
    elif l_label.device != pse_label.device:
        l_label = l_label.to(pse_label.device)

    label = torch.cat([l_label, pse_label])

    return mask, label

def compute_u_prototype_matrix(uq_idx, pseudo_label, similarity_matrix, label_to_index, local_rank):
    """
    基于prototype相似度计算批次内样本关系矩阵的函数
    
    参数:
    uq_idx: 批次内样本的索引，形状为 [batch_size]
    pseudo_label: 所有样本的伪标签，形状为 [n_samples]
    similarity_matrix: 类别原型相似度矩阵，形状为 [num_classes, num_classes]
    label_to_index: 标签到相似度矩阵索引的映射字典
    local_rank: GPU本地排名
    
    返回:
    relation_matrix: 批次内样本关系矩阵，形状为 [batch_size, batch_size]
    """
    batch_size = uq_idx.shape[0]
    
    # 获取批次内的伪标签
    batch_labels = pseudo_label
    
    # 将标签转换为相似度矩阵的索引
    batch_indices = torch.tensor([label_to_index[label.item()] for label in batch_labels], 
                                device=pseudo_label.device)
    
    # 直接使用类别相似度矩阵作为关系矩阵的基础
    # 获取批次内样本对应的类别相似度矩阵
    relation_matrix = similarity_matrix[batch_indices.unsqueeze(1), batch_indices.unsqueeze(0)]
    
    # 将对角线设为0（自己与自己的关系设为0）
    relation_matrix.fill_diagonal_(0.0)
    
    # 归一化关系矩阵到[0,1]范围
    min_val = relation_matrix.min()
    max_val = relation_matrix.max()
    relation_matrix = (relation_matrix - min_val) / (max_val - min_val + 1e-8)
    
    # 按行归一化，使每行和为1
    row_sums = relation_matrix.sum(dim=1, keepdim=True)
    relation_matrix = relation_matrix / (row_sums + 1e-8)
    
    return relation_matrix

def compute_prototype_matrix_init_before_20260505(uq_idx, old_mask, pseudo_label,label_to_idx,idx_to_label, similarity_matrix,local_rank):
    """
    向量化实现的计算批次内样本关系矩阵的函数
    
    参数:
    uq_idx: 批次内样本的索引，形状为 [batch_size]
    old_mask: 所有样本的mask，形状为 [n_samples]
    pseudo_label: 所有样本的伪标签，形状为 [n_samples]
    similarity_matrix: 类别原型相似度矩阵，形状为 [51, 51]
    
    返回:
    relation_matrix: 批次内样本关系矩阵，形状为 [batch_size, batch_size]
    """
    _,batch_class_indices = get_similarity_for_labels(similarity_matrix, label_to_idx,idx_to_label,pseudo_label)
    batch_size = len(pseudo_label)
    
    mask_matrix = old_mask.unsqueeze(1) == old_mask.unsqueeze(0)
    
    class_matrix = batch_class_indices.unsqueeze(1) == batch_class_indices.unsqueeze(0)
    
    class_sim_matrix = similarity_matrix[batch_class_indices.unsqueeze(1), batch_class_indices.unsqueeze(0)]
    
    # 初始化关系矩阵
    relation_matrix = torch.zeros((batch_size, batch_size), dtype=torch.float32).cuda(local_rank,non_blocking=True)
    
    for i in range(batch_size):
        for j in range(batch_size):
            # 如果两个样本mask相同
            if mask_matrix[i, j]:
                relation_matrix[i, j] = 0.3* class_sim_matrix[i, j]
            # 如果两个样本mask不同
            else:
                # 如果两个样本类别相同
                if class_matrix[i, j]:
                    relation_matrix[i, j] = class_sim_matrix[i, j]
                else:
                    relation_matrix[i, j] = 0.5* class_sim_matrix[i, j]

    prototype_matrix = relation_matrix.fill_diagonal_(0.0)
    prototype_matrix = (prototype_matrix - prototype_matrix.min()) / (
                        prototype_matrix.max() - prototype_matrix.min() + 0.0000001)
    prototype_matrix = prototype_matrix / prototype_matrix.sum(dim=1)

    return prototype_matrix

def compute_prototype_matrix(uq_idx, old_mask, pseudo_label,label_to_idx,idx_to_label, similarity_matrix,local_rank):
    """
    向量化实现的计算批次内样本关系矩阵的函数
    
    参数:
    uq_idx: 批次内样本的索引，形状为 [batch_size]
    old_mask: 所有样本的mask，形状为 [n_samples]
    pseudo_label: 所有样本的伪标签，形状为 [n_samples]
    similarity_matrix: 类别原型相似度矩阵，形状为 [51, 51]
    
    返回:
    relation_matrix: 批次内样本关系矩阵，形状为 [batch_size, batch_size]
    """
    _,batch_class_indices = get_similarity_for_labels(similarity_matrix, label_to_idx,idx_to_label,pseudo_label)
    batch_size = len(pseudo_label)
    
    mask_matrix = old_mask.unsqueeze(1) == old_mask.unsqueeze(0)
    
    class_matrix = batch_class_indices.unsqueeze(1) == batch_class_indices.unsqueeze(0)
    
    class_sim_matrix = similarity_matrix[batch_class_indices.unsqueeze(1), batch_class_indices.unsqueeze(0)]
    
    # 初始化关系矩阵
    relation_matrix = torch.zeros((batch_size, batch_size), dtype=torch.float32).cuda(local_rank,non_blocking=True)
    
    for i in range(batch_size):
        for j in range(batch_size):
            # 如果两个样本mask相同
            if mask_matrix[i, j]:
                relation_matrix[i, j] = class_sim_matrix[i, j]
            # 如果两个样本mask不同
            else:
                # 如果两个样本类别相同
                if class_matrix[i, j]:
                    relation_matrix[i, j] = 0.5*class_sim_matrix[i, j]
                else:
                    relation_matrix[i, j] = class_sim_matrix[i, j]

    prototype_matrix = relation_matrix.fill_diagonal_(0.0)
    prototype_matrix = (prototype_matrix - prototype_matrix.min()) / (
                        prototype_matrix.max() - prototype_matrix.min() + 0.0000001)
    prototype_matrix = prototype_matrix / prototype_matrix.sum(dim=1)

    return prototype_matrix