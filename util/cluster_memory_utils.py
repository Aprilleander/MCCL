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
from util.cluster_and_log_utils import log_accs_from_preds,split_cluster_acc_v2_kmeans
import torch.distributed as dist
import os 
import tqdm
import random
from util.loss import KL_loss,top1_sharpness_loss,max_margin_loss
from torch.autograd import Function


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
            # ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            # ctx.features[y] /= ctx.features[y].norm()
            # print(temp.size())
            # print(x.size())
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
    def __init__(self, num_features, num_samples,local_rank,all_sample=1600,temp=0.05, momentum=0.2, use_hard=False, args=None):
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
        self.register_buffer('all_feature', torch.zeros(all_sample, num_features))
        self.register_buffer('cls_feature',torch.zeros(num_samples,num_samples))

    def forward(self,inputs,k_inputs,targets):

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


@torch.no_grad()
def generate_cluster_features(labels, features,num_class):
    centers = collections.defaultdict(list)

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
        final_centers.append(center)

    centers = torch.stack(final_centers, dim=0) 

    return centers


@torch.no_grad()
def generate_cluster_features_and_links(labels, features, num_class, seed=42):
    torch.manual_seed(seed)  # 设置随机种子，确保可复现

    centers = collections.defaultdict(list)
    linked_list = collections.defaultdict(list)  # 存储每个 label 对应的所有 features

    for i, label in enumerate(labels):
        
        centers[label.item()].append(F.normalize((features[i]),dim=0))

        linked_list[label.item()].append(F.normalize((features[i]),dim=0))

    final_centers = []
    for idx in range(num_class):
        if idx in centers:
            # 计算该类别的特征中心
            center = torch.stack(centers[idx], dim=0).mean(0)
        else:
            # 生成归一化随机向量作为该类别的中心
            center = F.normalize(torch.randn_like(features[0]), dim=0)
        final_centers.append(center)

    centers_tensor = torch.stack(final_centers, dim=0)  # [num_class, feature_dim]
    return centers_tensor, linked_list


def extract_features(model, data_loader,test_meter, args,print_freq=100, local_rank=0):
    model.eval()

    features = []
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

            label = _item[1]
            uq_idxs = _item[2]
            index = _item[3]
            mask = torch.tensor(np.array([True if x.item() in args.train_classes else False for x in label]))

            outputs = model(imgs)
            
            feature = outputs[0].data.cpu()

            pred = outputs[1].data.cpu()
            
            features.append(feature)
            cls_list.append(pred)

            test_meter.update_stats(
                pred, label, mask,uq_idxs
            )

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}]\t'
                      .format(i + 1))

    pred,label,mask = test_meter.finalize_preds()

    features = torch.cat(features, dim=0)
    cls_list = torch.cat(cls_list,dim=0)
    
    return pred, label,features,cls_list


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
            # mask_cls = np.append(mask_cls,np.array([True if x.item() in range(len(args.train_classes))
            #                                 else False for x in label]))
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


def class_vector(backbone,projector,memory,train_loader,test_meter,args,cfg):

    local_rank = args.local_rank
    
    pred,label,all_feature,cls_list = extract_features(backbone,train_loader,test_meter,args, print_freq=50, local_rank=local_rank)

    pred_list = np.array([m.numpy() for m in pred])

    label_list = np.array([m.numpy() for m in label])
    
    cluster_features,links = generate_cluster_features_and_links(label_list, all_feature,cfg.MODEL.NUM_CLASSES)
    cls_feat = generate_cluster_features(label_list, cls_list,cfg.MODEL.NUM_CLASSES)
    
    memory.features = F.normalize(cluster_features, dim=1)

    memory.cls_feature = cls_feat

    return memory,links

