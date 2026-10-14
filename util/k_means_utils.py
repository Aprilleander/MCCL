import argparse
from asyncio.log import logger
import os

from torch.utils.data import DataLoader
import numpy as np
from sklearn.cluster import KMeans
import torch
from util.cluster_and_log_utils import log_accs_from_preds

from util.faster_mix_k_means_pytorch import K_Means as SemiSupKMeans

from tqdm import tqdm

from collections import defaultdict

# TODO: Debug
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
import torch.nn as nn


def test_kmeans_semi_sup(model, test_loader, epoch, save_name, device, args, K=None, logger_class=None, in_training=False):
    """
    In this case, the test loader needs to have the labelled and unlabelled subsets of the training data
    """

    if K is None:
        K = args.num_labeled_classes + args.num_unlabeled_classes
    if isinstance(model, (list, tuple, nn.ModuleList, nn.ModuleDict)) and len(model) >= 2:
        co_feat_extractor, att_feat_extractor = model[0], model[1]
        co_feat_extractor.eval()
        att_feat_extractor.eval()

        all_feats, all_co_feats = [], []
        targets = np.array([])
        mask_lab = np.array([])  # From all the data, which instances belong to the labelled set
        mask_cls = np.array([])  # From all the data, which instances belong to Old classes

        print('Collating features...')
        # First extract all features
        for batch_idx, (images, label, _, mask_lab_, attribute) in enumerate(tqdm(test_loader)):
            images = images.to(device)

            co_feats, att_embs = co_feat_extractor(images)
            att_feats = att_feat_extractor(att_embs)
            if args.use_l2_in_ssk:
                co_feats = torch.nn.functional.normalize(co_feats, dim=-1)
                att_feats = torch.nn.functional.normalize(att_feats, dim=-1)

            feats = torch.cat((co_feats, att_feats), dim=1)
            all_co_feats.append(co_feats.cpu().numpy())
            all_feats.append(feats.cpu().numpy())
            targets = np.append(targets, label.cpu().numpy())
            mask_cls = np.append(mask_cls, np.array([True if x.item() in range(len(args.train_classes))
                                                     else False for x in label]))
            mask_lab = np.append(mask_lab, mask_lab_.cpu().bool().numpy())

        # -----------------------
        # K-MEANS
        # -----------------------
        mask_lab = mask_lab.astype(bool)
        mask_cls = mask_cls.astype(bool)

        all_feats = np.concatenate(all_feats)
        all_co_feats = np.concatenate(all_co_feats)

        l_feats = all_feats[mask_lab]  # Get labelled set
        u_feats = all_feats[~mask_lab]  # Get unlabelled set
        l_co_feats = all_co_feats[mask_lab]  # Get labelled set
        u_co_feats = all_co_feats[~mask_lab]  # Get unlabelled set
        l_targets = targets[mask_lab]  # Get labelled targets
        u_targets = targets[~mask_lab]  # Get unlabelled targets

        print('Fitting Semi-Supervised K-Means with concatenated_feature...')
        if in_training:
            max_kmeans_iter = args.train_max_kmeans_iter
        else:
            max_kmeans_iter = args.max_kmeans_iter
        logger_class('max_kmeans_iter: {max_kmeans_iter}!')
        kmeans = SemiSupKMeans(k=K, tolerance=1e-4, max_iterations=max_kmeans_iter, init='k-means++',
                               n_init=args.k_means_init, random_state=None, n_jobs=None, pairwise_batch_size=512,
                               mode=None)

        l_feats, u_feats, l_targets, u_targets = (torch.from_numpy(x).to(device) for
                                                  x in (l_feats, u_feats, l_targets, u_targets))

        kmeans.fit_mix(u_feats, l_feats, l_targets)
        all_preds = kmeans.labels_.cpu().numpy()
        u_targets = u_targets.cpu().numpy()

        # -----------------------
        # EVALUATE
        # -----------------------
        # Get preds corresponding to unlabelled set
        preds = all_preds[~mask_lab]

        # Get portion of mask_cls which corresponds to the unlabelled set
        mask = mask_cls[~mask_lab]
        mask = mask.astype(bool)

        # -----------------------
        # EVALUATE
        # -----------------------
        all_acc, old_acc, new_acc = log_accs_from_preds(y_true=u_targets, y_pred=preds, mask=mask,
                                                        eval_funcs=args.eval_funcs,
                                                        save_name=save_name, T=epoch, print_output=True)
        logger_class(
            'Using concatenated_feature ==> SS-K Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc,
                                                                                                          old_acc,
                                                                                                          new_acc))

        print('Using contrastive_feature Fitting Semi-Supervised K-Means...')
        kmeans = SemiSupKMeans(k=K, tolerance=1e-4, max_iterations=args.max_kmeans_iter, init='k-means++',
                               n_init=args.k_means_init, random_state=None, n_jobs=None, pairwise_batch_size=512,
                               mode=None)

        l_c0_feats, u_co_feats, l_targets, u_targets = (torch.from_numpy(x).to(device) for
                                                        x in (l_co_feats, u_co_feats, l_targets, u_targets))

        kmeans.fit_mix(u_co_feats, l_c0_feats, l_targets)
        all_preds = kmeans.labels_.cpu().numpy()
        u_targets = u_targets.cpu().numpy()

        # -----------------------
        # EVALUATE
        # -----------------------
        # Get preds corresponding to unlabelled set
        preds = all_preds[~mask_lab]

        # Get portion of mask_cls which corresponds to the unlabelled set
        mask = mask_cls[~mask_lab]
        mask = mask.astype(bool)

        # -----------------------
        # EVALUATE
        # -----------------------
        all_acc, old_acc, new_acc = log_accs_from_preds(y_true=u_targets, y_pred=preds, mask=mask,
                                                        eval_funcs=args.eval_funcs,
                                                        save_name=save_name, T=epoch, print_output=True)

        return all_acc, old_acc, new_acc, kmeans


    else:
        model.eval()
        all_feats = []
        targets = np.array([])
        mask_lab = np.array([])  # From all the data, which instances belong to the labelled set
        mask_cls = np.array([])  # From all the data, which instances belong to Old classes

        print('Collating features...')
        # First extract all features
        for batch_idx, (images, label, _, mask_lab_, attribute) in enumerate(tqdm(test_loader)):
            images = images.to(device)
            feats = model(images)
            if args.use_l2_in_ssk:
                feats = torch.nn.functional.normalize(feats, dim=-1)
                
            all_feats.append(feats.cpu().numpy())
            targets = np.append(targets, label.cpu().numpy())            
            mask_cls = np.append(mask_cls, np.array([True if x.item() in range(len(args.train_classes))
                                                    else False for x in label]))
            mask_lab = np.append(mask_lab, mask_lab_.cpu().bool().numpy())

            
        mask_lab = mask_lab.astype(bool)
        mask_cls = mask_cls.astype(bool)
        all_feats = np.concatenate(all_feats)


        l_feats = all_feats[mask_lab]  # Get labelled set
        u_feats = all_feats[~mask_lab]  # Get unlabelled set
        l_targets = targets[mask_lab]  # Get labelled targets
        u_targets = targets[~mask_lab]  # Get unlabelled targets
        if in_training:
            max_kmeans_iter = args.train_max_kmeans_iter
        else:
            max_kmeans_iter = args.max_kmeans_iter
        logger_class(f'Fitting Semi-Supervised K-Means... max_kmeans_iter = {max_kmeans_iter}')

        kmeans = SemiSupKMeans(k=K, tolerance=1e-4, max_iterations=args.max_kmeans_iter, init='k-means++',
                               n_init=args.k_means_init, random_state=None, n_jobs=None, pairwise_batch_size=512,
                               mode=None)

        l_feats, u_feats, l_targets, u_targets = (torch.from_numpy(x).to(device) for
                                                  x in (l_feats, u_feats, l_targets, u_targets))

        kmeans.fit_mix(u_feats, l_feats, l_targets)
        all_preds = kmeans.labels_.cpu().numpy()
        u_targets = u_targets.cpu().numpy()

        # -----------------------
        # EVALUATE
        # -----------------------
        # Get preds corresponding to unlabelled set
        preds = all_preds[~mask_lab]

        # Get portion of mask_cls which corresponds to the unlabelled set
        mask = mask_cls[~mask_lab]
        mask = mask.astype(bool)

        # -----------------------
        # EVALUATE
        # -----------------------
        all_acc, old_acc, new_acc = log_accs_from_preds(y_true=u_targets, y_pred=preds, mask=mask,
                                                        eval_funcs=args.eval_funcs,
                                                        save_name=save_name, T=epoch, print_output=True)

        if args.use_contrastive_cluster and args.contrastive_cluster_method == 'ssk':
            return all_acc, old_acc, new_acc, kmeans, all_feats
        else:
            return all_acc, old_acc, new_acc, kmeans, None

def load_sample_assignments(file_path, local_rank=None):
    # 确保文件存在
    if not os.path.exists(file_path):
        # raise FileNotFoundError(f"文件 {file_path} 不存在")
        return None
    
    # 加载数组
    sample_assignments = np.load(file_path)
    
    # 如果有指定local_rank，打印相关信息
    
    return sample_assignments
    
def test_kmeans(model, test_loader,
                epoch, save_name, device,
                args, logger_class=None, output_kmeans=False):
    model.eval()

    all_feats = []
    all_concat_feats = []
    targets = np.array([])
    mask = np.array([])
    logger_class('Collating features...')

    # First extract all features
    for batch_idx, _item in enumerate(tqdm(test_loader)):
        images = _item[0]
        label = _item[1]
        images = images.to(device)

        # Pass features through base model and then additional learnable transform (linear layer)
        # concat_feats = model(images, concat=True)
        feats = model(images)
        concat_feats = feats.detach().clone()
        feats = torch.nn.functional.normalize(feats, dim=-1)

        all_concat_feats.append(concat_feats.cpu().numpy())
        all_feats.append(feats.cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        # all_feats.append(feats.cpu().detach().numpy())
        # targets = np.append(targets, label.cpu().detach().numpy())
        mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
                                         else False for x in label]))

    all_concat_feats = np.concatenate(all_concat_feats)

    kmeans = KMeans(n_clusters=args.num_labeled_classes + args.num_unlabeled_classes, random_state=0).fit(
        all_concat_feats)
    preds = kmeans.labels_

    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name+'_concate'
                                                    )
    logger_class('unlabel kmeans concate_feats: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc,
                                                                                        new_acc))

    
    all_feats = np.concatenate(all_feats)
    kmeans = KMeans(n_clusters=args.num_labeled_classes + args.num_unlabeled_classes, random_state=0).fit(all_feats)
    preds = kmeans.labels_

    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    )
    logger_class('unlabel kmeans normallize_feats: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc,
                                                                                        new_acc))

    if output_kmeans:
        return all_acc, old_acc, new_acc, kmeans, all_feats
    else:
        return all_acc, old_acc, new_acc




def fake_label_kmeans(model, test_loader,
                epoch, save_name, device,
                args, logger_class=None, K=200):
    model.eval()

    all_feats = []
    all_concat_feats = []
    targets = np.array([])
    mask = np.array([])
    logger_class('Collating features...')

    # First extract all features
    for batch_idx, _item in enumerate(tqdm(test_loader)):
        images = _item[0]
        label = _item[1]
        images = images.to(device)

        # Pass features through base model and then additional learnable transform (linear layer)
        # concat_feats = model(images, concat=True)
        feats = model(images)
        feats = torch.nn.functional.normalize(feats, dim=-1)

        all_feats.append(feats.cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        # all_feats.append(feats.cpu().detach().numpy())
        # targets = np.append(targets, label.cpu().detach().numpy())
        mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
                                         else False for x in label]))

    logger_class(f'Fitting K-Means... with K={K}, dataset K = {args.num_labeled_classes + args.num_unlabeled_classes}')


    all_feats = np.concatenate(all_feats)
    kmeans = KMeans(n_clusters=K, random_state=0).fit(all_feats)
    preds = kmeans.labels_

    logger_class('Using contrastive feature Done')

    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    writer=args.writer)


    return all_acc, old_acc, new_acc, kmeans, all_feats


def collect_unknown_combinations(student_out,mask_lab,uq_idxs,args):
    top_k =3
    known_classes = args.train_classes
    known_logits = student_out[mask_lab]
    known_max_logits, _ = known_logits.max(dim=1)
    t = torch.min(known_max_logits)*0.8

    unknown_mask = ~mask_lab
    unknown_logits = student_out[unknown_mask]
    unknown_uq_idxs = uq_idxs[unknown_mask]

    unknown_max_vals, _ = unknown_logits.max(dim=1)
    
    valid_unknown_mask = (unknown_max_vals < t)

    novel_logits = unknown_logits[valid_unknown_mask]
    novel_idx = unknown_uq_idxs[valid_unknown_mask]

    # 获取已知类别中logits最大的3个索引
    topk_vals, topk_relative_idxs = torch.topk(novel_logits, k=3)

    return novel_idx,topk_relative_idxs

def assign_pseudolabels(args,uq_idx, topk_relative, min_count=2):
    """
    基于未知样本的top-k已知类组合分配伪标签
    
    参数:
        uq_idx: 未知样本索引数组 (N,)
        topk_relative: 每个样本的前3个已知类 (N, 3)
        min_count: 组合最小出现次数阈值
        
    返回:
        pseudolabel_map: 组合到伪标签的映射
        sample_assignments: 样本分配到伪标签的结果 (样本索引, 伪标签)
    """
    novel_k = len(args.unlabeled_classes)
    # 1. 生成排序的组合并计数
    combination_counter = defaultdict(int)
    combination_samples = defaultdict(list)  # 记录每个组合对应的样本索引
    
    for idx, topk in zip(uq_idx, topk_relative):
        # 排序组合以确保一致性 (e.g., [4,32,46] 和 [32,4,46] 视为相同)
        sorted_comb = tuple(sorted(topk))
        combination_counter[sorted_comb] += 1
        combination_samples[sorted_comb].append(idx)
    
    # 2. 过滤低频组合
    filtered_combs = {
        comb: count 
        for comb, count in combination_counter.items() 
        if count >= min_count
    }
    
    # 3. 按频率排序组合 (降序)
    sorted_combs = sorted(
        filtered_combs.items(), 
        key=lambda x: x[1], 
        reverse=True
    )

    if novel_k is not None:
        sorted_combs = sorted_combs[:novel_k]  # 只保留前novel_k个高频组合
    
    # 4. 分配伪标签 (奇数序列)
    pseudolabel_map = {}
    next_label = 1
    
    for comb, count in sorted_combs:
        pseudolabel_map[comb] = next_label
        next_label += 2  # 只使用奇数标签
    
    sample_assignments = []
    for comb, label in pseudolabel_map.items():
        for sample_idx in combination_samples[comb]:
            sample_assignments.append((sample_idx, label))
    
    return pseudolabel_map, np.array(sample_assignments)
    

# 步骤3: 为无标签数据生成伪标签
def generate_pseudolabels(cfg,batch_uq_idx, sample_assignments, mask_lab, class_labels,local_rank):
    """
    为批量数据分配伪标签并更新标签和掩码 (PyTorch Tensor版本)
    
    参数:
        batch_uq_idx: 当前批量的未知样本索引 (形状 [batch_size,])
        sample_assignments: 样本分配结果 (形状 [M, 2], 每行 [样本索引, 伪标签])
        mask_lab: 标签掩码 (形状 [batch_size, 1], 0=无标签, 1=有标签)
        class_labels: 类别标签 (形状 [batch_size,])
        
    返回:
        updated_class_labels: 更新后的类别标签 (Tensor)
        updated_mask_lab: 更新后的标签掩码 (Tensor)
        new_labels: 分配的伪标签数组 (Tensor, 未分配为-1)
    """
    # 确保输入是Tensor
    if not isinstance(batch_uq_idx, torch.Tensor):
        batch_uq_idx = torch.tensor(batch_uq_idx, dtype=torch.long)
    
    if not isinstance(sample_assignments, torch.Tensor):
        sample_assignments = torch.tensor(sample_assignments, dtype=torch.long).cuda(local_rank,non_blocking= True)
    
    # 1. 分配伪标签
    assigned_indices = sample_assignments[:, 0]
    assigned_labels = sample_assignments[:, 1]
    
    # 创建布尔掩码标记哪些样本有分配
    mask = torch.isin(batch_uq_idx, assigned_indices)
    
    # 初始化伪标签数组
    new_labels = torch.full((int(cfg.TRAIN.BATCH_SIZE/cfg.NUM_GPUS),), -1, dtype=torch.long).cuda(local_rank,non_blocking= True)
    
    # 如果有分配的样本
    if torch.any(mask):
        # 获取有分配的样本索引
        assigned_in_batch = batch_uq_idx[mask]
        
        # 为这些样本创建临时字典
        temp_dict = dict(zip(assigned_indices.cpu().numpy(), assigned_labels.cpu().numpy()))
        
        # 获取对应的标签
        labels_for_assigned = torch.tensor(
            [temp_dict.get(idx.item(), -1) for idx in assigned_in_batch],
            dtype=torch.long,
            device=batch_uq_idx.device
        )
        
        # 填充伪标签
        new_labels[mask] = labels_for_assigned
    
    # 2. 更新标签和掩码
    # 创建副本以避免修改原始Tensor
    updated_class_labels = class_labels.clone()
    updated_mask_lab = mask_lab.clone()
    
    # 找到需要更新的位置: 1) 有伪标签分配 2) 原始无标签
    # 将mask_lab从[batch_size, 1]转为[batch_size]
    mask_lab_squeezed = mask_lab.squeeze()
    
    # 更新条件: 新标签有效且原始无标签
    update_mask = (new_labels != -1) & (mask_lab_squeezed == 0)
    
    # 更新类别标签
    updated_class_labels[update_mask] = new_labels[update_mask]
    
    # 更新掩码 (将0改为1)
    # 处理不同维度情况
    if updated_mask_lab.dim() == 2:
        updated_mask_lab[update_mask, 0] = 1
    else:
        updated_mask_lab[update_mask] = 1
    
    return updated_class_labels, updated_mask_lab, new_labels
