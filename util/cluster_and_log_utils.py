import torch
import torch.distributed as dist
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
from util.pred_analysis import visualize_class_distribution,generate_label_pred_excel
from collections import defaultdict
import pandas as pd
import tqdm

from kmeans_pytorch.kmeans_pytorch import kmeans
from sklearn.cluster import KMeans

def all_sum_item(item):
    item = torch.tensor(item).cuda()
    dist.all_reduce(item)
    return item.item()

def topks_correct(preds, labels, ks):
    assert preds.size(0) == labels.size(
        0
    ), "Batch dim of predictions and labels must match"
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, max(ks), dim=1, largest=True, sorted=True
    )
    # (batch_size, max_k) -> (max_k, batch_size).
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size).
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct.
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k.
    topks_correct = [top_max_k_correct[:k, :].float().sum() for k in ks]

    pred_label = top_max_k_inds[0]

    #generate_label_pred_excel(pred_label,labels)

    return topks_correct


def split_cluster_acc_v2(y_true, y_pred,mask):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    First compute linear assignment on all data, then look at how good the accuracy is on subsets

    # Arguments
        mask: Which instances come from old classes (True) and which ones come from new classes (False)
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """

    y_true = y_true.numpy().astype(int)
    y_pred = y_pred.numpy().astype(int)

    # y_true = y_true[:400]
    # y_pred = y_pred[:400]

    # y_true = y_true.astype(int)
    # y_pred = y_pred.astype(int)


    old_classes_gt = set(y_true[mask])
    new_classes_gt = set(y_true[~mask])

    #assert y_pred.size == y_true.size

    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    ind_map = {j: i for i, j in ind}

    total_acc = sum([w[i, j] for i, j in ind])
    total_instances = y_pred.size
    
    # try: 
    #     if dist.get_world_size() > 0:
    #         total_acc = all_sum_item(total_acc)
    #         total_instances = all_sum_item(total_instances)
    # except:
    #     pass
   
    total_acc /= total_instances
    old_acc = 0
    total_old_instances = 0

    for i in old_classes_gt:
        old_acc += w[ind_map[i], i]
        total_old_instances += sum(w[:, i])
    
    # try:
    #     if dist.get_world_size() > 0:
    #         old_acc = all_sum_item(old_acc)
    #         total_old_instances = all_sum_item(total_old_instances)
    # except:
    #     pass

    old_acc /= total_old_instances

    new_acc = 0
    total_new_instances = 0
    for i in new_classes_gt:
        new_acc += w[ind_map[i], i]
        total_new_instances += sum(w[:, i])

    # try:
    #     if dist.get_world_size() > 0:
    #         new_acc = all_sum_item(new_acc)
    #         total_new_instances = all_sum_item(total_new_instances)
    # except:
    #     pass
    
    new_acc /= total_new_instances

    return total_acc, old_acc, new_acc

def split_cluster_acc_v2_kmeans(y_true, y_pred,mask):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    First compute linear assignment on all data, then look at how good the accuracy is on subsets

    # Arguments
        mask: Which instances come from old classes (True) and which ones come from new classes (False)
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """

    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

   
    old_classes_gt = set(y_true[mask])
    new_classes_gt = set(y_true[~mask])

    #assert y_pred.size == y_true.size

    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    ind_map = {j: i for i, j in ind}

    total_acc = sum([w[i, j] for i, j in ind])
    total_instances = y_pred.size

    total_acc /= total_instances
    old_acc = 0
    total_old_instances = 0

    for i in old_classes_gt:
        old_acc += w[ind_map[i], i]
        total_old_instances += sum(w[:, i])
    

    old_acc /= total_old_instances

    new_acc = 0
    total_new_instances = 0
    for i in new_classes_gt:
        new_acc += w[ind_map[i], i]
        total_new_instances += sum(w[:, i])
    
    new_acc /= total_new_instances

    return total_acc, old_acc, new_acc



def split_cluster_acc_v2_balanced(y_true, y_pred, mask):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    First compute linear assignment on all data, then look at how good the accuracy is on subsets

    # Arguments
        mask: Which instances come from old classes (True) and which ones come from new classes (False)
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    old_classes_gt = set(y_true[mask])
    new_classes_gt = set(y_true[~mask])

    # assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    ind_map = {j: i for i, j in ind}

    old_acc = np.zeros(len(old_classes_gt))
    total_old_instances = np.zeros(len(old_classes_gt))

    for idx, i in enumerate(old_classes_gt):
        old_acc[idx] += w[ind_map[i], i]
        total_old_instances[idx] += sum(w[:, i])

    new_acc = np.zeros(len(new_classes_gt))
    total_new_instances = np.zeros(len(new_classes_gt))
    for idx, i in enumerate(new_classes_gt):
        new_acc[idx] += w[ind_map[i], i]
        total_new_instances[idx] += sum(w[:, i])

    # try:
    #     if dist.get_world_size() > 0:
    #         old_acc, new_acc = torch.from_numpy(old_acc).cuda(), torch.from_numpy(new_acc).cuda()
    #         dist.all_reduce(old_acc), dist.all_reduce(new_acc)
    #         dist.all_reduce(total_old_instances), dist.all_reduce(total_new_instances)
    #         old_acc, new_acc = old_acc.cpu().numpy(), new_acc.cpu().numpy()
    #         total_old_instances, total_new_instances = total_old_instances.cpu().numpy(), total_new_instances.cpu().numpy()
    # except:
    #     pass

    total_acc = np.concatenate([old_acc, new_acc]) / np.concatenate([total_old_instances, total_new_instances])
    old_acc /= total_old_instances
    new_acc /= total_new_instances
    total_acc, old_acc, new_acc = total_acc.mean(), old_acc.mean(), new_acc.mean()
    return total_acc, old_acc, new_acc


EVAL_FUNCS = {
    'v2': split_cluster_acc_v2,
    'v2b': split_cluster_acc_v2_balanced
}

def log_accs_from_preds(y_true, y_pred, mask, eval_funcs, save_name, T=None,
                        print_output=True, args=None):

    """
    Given a list of evaluation functions to use (e.g ['v1', 'v2']) evaluate and log ACC results

    :param y_true: GT labels
    :param y_pred: Predicted indices
    :param mask: Which instances belong to Old and New classes
    :param T: Epoch
    :param eval_funcs: Which evaluation functions to use
    :param save_name: What are we evaluating ACC on
    :param writer: Tensorboard logger
    :return:
    """

    mask = mask.astype(bool)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(float)

    for i, f_name in enumerate(eval_funcs):

        acc_f = EVAL_FUNCS[f_name]
        all_acc, old_acc, new_acc = acc_f(y_true, y_pred, mask)
        log_name = f'{save_name}_{f_name}'

        if i == 0:
            to_return = (all_acc, old_acc, new_acc)

        if print_output:
            print_str = f'Epoch {T}, {log_name}: All {all_acc:.4f} | Old {old_acc:.4f} | New {new_acc:.4f}'
            try:
                if dist.get_rank() == 0:
                    try:
                        args.logger.info(print_str)
                    except:
                        print(print_str)
            except:
                pass

    return to_return

def analyze_cluster_target_relation(target, label):
    """
    分析聚类标签与真实标签的对应关系
    
    参数:
    target: 真实标签数组
    label: 聚类标签数组
    
    返回:
    DataFrame: 包含统计结果的表格
    """
    # 检查输入长度是否一致
    if len(target) != len(label):
        raise ValueError("target和label数组长度必须相同")
    
    # 创建映射字典：真实标签 -> (聚类标签 -> 计数)
    mapping = defaultdict(lambda: defaultdict(int))
    
    # 统计每个真实标签对应的聚类标签数量
    for t, l in zip(target, label):
        mapping[t][l] += 1
    
    # 准备结果数据
    results = []
    
    # 对每个真实标签类别进行处理
    for target_class in sorted(mapping.keys()):
        cluster_counts = mapping[target_class]
        total = sum(cluster_counts.values())
        
        # 对每个聚类标签添加一行结果
        for cluster_class, count in cluster_counts.items():
            percentage = (count / total) * 100
            results.append({
                '真实标签': target_class,
                '聚类标签': cluster_class,
                '样本数量': count,
                '百分比': percentage
            })
    
    # 创建DataFrame
    df = pd.DataFrame(results)
    
    return df

def save_to_excel(df, filename="cluster_target_analysis.xlsx"):
    """
    将结果保存到Excel文件
    
    参数:
    df: 包含统计结果的DataFrame
    filename: 输出文件名
    """
    # 创建一个Excel写入器
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # 写入详细数据
        df.to_excel(writer, sheet_name='详细数据', index=False)
        
        # 创建透视表：以真实标签为行，聚类标签为列，百分比为值
        pivot_df = df.pivot_table(
            index='真实标签', 
            columns='聚类标签', 
            values='百分比', 
            fill_value=0,
            aggfunc='sum'
        )
        
        # 写入透视表
        pivot_df.to_excel(writer, sheet_name='百分比汇总')
        
        # 创建样本数量透视表
        count_pivot = df.pivot_table(
            index='真实标签', 
            columns='聚类标签', 
            values='样本数量', 
            fill_value=0,
            aggfunc='sum'
        )
        
        # 写入样本数量透视表
        count_pivot.to_excel(writer, sheet_name='样本数量汇总')


def test_kmeans_all_dimension(model,pdf,excel_path, test_loader,test_meter, epoch,save_name, args,projection_head=None, Use_GPU=False):
    model.eval()
    projection_head.eval()
    all_feats = []
    all_norm_logit = []
    all_proto_logit = []
    all_proj = []

    new_test_meter= test_meter

    targets = np.array([])
    mask = np.array([])
    device = next(model.parameters()).device

    # First extract all features
    with torch.no_grad():
        for batch_idx, (images,_, label, uq_index,index) in enumerate(test_loader):

            images = images.to(device)
            # Pass features through base model and then additional learnable transform (linear layer)
            feats,cls_x,_,_ = model(images)
            proj,logits = projection_head(feats)
            all_feats.append(torch.nn.functional.normalize(feats, dim=-1).detach().cpu().numpy())
            all_norm_logit.append(logits.argmax(1).cpu().numpy())
            
            all_proj.append(torch.nn.functional.normalize(proj, dim=-1).detach().cpu().numpy())
            targets = np.append(targets, label.cpu().numpy())
            mask_ = np.array([True if x.item() in args.train_classes else False for x in label])
            mask = np.append(mask,mask_)
            label = label
            mask_= torch.from_numpy(mask_)
            uq_index= uq_index
            cls_x.cpu()
            logits.cpu()
            mask_.cpu()
            # test_meter.update_stats(
            #     cls_x, label, mask_,uq_index
            # )

            # new_test_meter.update_stats(
            #     logits, label, mask_,uq_index
            # )

    # cnn_all_acc_clsx, cnn_old_acc_clsx,cnn_new_acc_clsx = test_meter.finalize_metrics()

    # cnn_all_acc_logits, cnn_old_acc_logits,cnn_new_acc_logits = new_test_meter.finalize_metrics()


    # Get portion of mask_cls which corresponds to the unlabelled set
    mask = mask.astype(bool)
    all_feats = np.concatenate(all_feats)
    # all_logit = np.concatenate(all_logit)
    all_proj = np.concatenate(all_proj)
    all_norm_logit = np.concatenate(all_norm_logit)
    # all_proto_logit = np.concatenate(all_proto_logit)
    # -----------------------
    # EVALUATE
    # -----------------------

    preds, prototypes = kmeans(X=torch.from_numpy(all_feats).to(device), num_clusters=args.num_unlabeled_classes+args.num_labeled_classes,
                                    distance='euclidean', device=device, tqdm_flag=False)

    preds_GPU, prototypes = preds.cpu().numpy(), prototypes.cpu().numpy()
    
    kmeanss = KMeans(n_clusters=args.num_labeled_classes + args.num_unlabeled_classes, random_state=0).fit(
        all_feats)
    preds = kmeanss.labels_
    
    # preds_logit = all_logit
    # quick_count(preds_logit)
    preds_norm_logit = all_norm_logit

    kmeanss = KMeans(n_clusters=args.num_labeled_classes + args.num_unlabeled_classes, random_state=0).fit(
        all_proj)
    preds_proj = kmeanss.labels_

    # -----------------------
    # EVALUATE
    # -----------------------
    #GPU
    all_acc_g, old_acc_g, new_acc_g = split_cluster_acc_v2_kmeans(y_true=targets, y_pred=preds_GPU, mask=mask)
    # feat
    all_acc, old_acc, new_acc = split_cluster_acc_v2_kmeans(y_true=targets, y_pred=preds, mask=mask)
    print('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
    #logit
    all_acc_ln, old_acc_ln, new_acc_ln = split_cluster_acc_v2_kmeans(y_true=targets, y_pred=preds_norm_logit, mask=mask)
    #logit
    # proj
    all_acc_p, old_acc_p, new_acc_p = split_cluster_acc_v2_kmeans(y_true=targets, y_pred=preds_proj, mask=mask)
    # balance feat
    all_acc_fb, old_acc_fb, new_acc_fb = split_cluster_acc_v2_balanced(y_true=targets, y_pred=preds, mask=mask)

    df = pd.read_excel(excel_path)
    # 格式化所有数值为保留两位小数
    new_row = {
        'Epoch': epoch,
        'All': round(all_acc, 4), 'Old': round(old_acc, 4), 'New': round(new_acc, 4),
        # 'All_cnn_clsx': round(cnn_all_acc_clsx, 4), 'Old_cnn_clsx': round(cnn_old_acc_clsx, 4), 'New_cnn_clsx': round(cnn_new_acc_clsx, 4),
        # 'All_cnn_logits': round(cnn_all_acc_logits, 4), 'Old_cnn_logits': round(cnn_old_acc_logits, 4), 'New_cnn_logits': round(cnn_new_acc_logits, 4),
        'All_gpu': round(all_acc_g, 4), 'Old_gpu': round(old_acc_g, 4), 'New_gpu': round(new_acc_g, 4),
        'All_norm_logit': round(all_acc_ln, 4), 'Old_norm_logit': round(old_acc_ln, 4), 'New_norm_logit': round(new_acc_ln, 4),
        'All_proj': round(all_acc_p, 4), 'Old_proj': round(old_acc_p, 4), 'New_proj': round(new_acc_p, 4),
        'All_feat_balance': round(all_acc_fb, 4), 'Old_feat_balance': round(old_acc_fb, 4), 'New_feat_balance': round(new_acc_fb, 4)

    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_excel(excel_path, index=False)


    # all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
    #                                                 T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
    #                                                 writer=args.writer)
    return all_acc, old_acc, new_acc
