import argparse
import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from util.general_utils import AverageMeter, init_experiment,get_mean_lr
from util.cluster_and_log_utils import log_accs_from_preds,split_cluster_acc_v2_kmeans
from util.k_means_utils import test_kmeans_semi_sup,test_kmeans
from util.misc import launch_job
from util.parser import load_config, parse_args
from util.general_utils import AverageMeter, init_experiment
from sklearn.cluster import KMeans
from kmeans_pytorch.kmeans_pytorch import kmeans

from torch.utils.tensorboard import SummaryWriter

from util.cluster_memory_utils import extract_features,only_extract_labeled_stprotos,generate_cluster_features,extract_labeled_protos,extract_labeled_stprotos,extract_labeled_stprotos_energy,feature_dropout,kmeansMLPProjector,class_vector_sup,class_vector,extract_labeled_stprotos_simple,compute_prototype_similarity,compute_prototype_matrix,unsupervised_prototype

from config import exp_root
from model import DINOHead, info_nce_logits, SupConLoss, DistillLoss, ContrastiveLearningViewGenerator, get_params_groups,info_nce_logits_for,LabelSmoothingLoss,top1_sharpness_loss,max_margin_loss,KL_loss,SharpLogitLoss
from backbone import mvit
import datetime

import argparse
import random
from backbone.mvit import MViT
from backbone.MViT_defaults import get_cfg
from backbone_times.vit import vit_base_patch16_224 as vit_times
from backbone_times.vit import TimeSformer as TimeSformer
from backbone_times.Timesformer_defaults import get_timesformer_cfg
from backbone_times.Timesformer_defaults import _assert_and_infer_cfg,load_config
from backbone import checkpoint as cu
from backbone.checkpoint import find_simgcd_latest,load_simgcd_legancy
from util import distributed as du
from util.cluster_memory_utils import ClusterMemory,load_results,save_results,convert_train_label,get_similarity_for_labels
from util.cluster_partion import steady_clustering
from util.meters import TestMeter
from torch.nn import functional as F
from copy import deepcopy

from data.mixup import MixUp
from data.get_datasets import get_datasets, get_class_splits
from data import loader

import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data.distributed
import torch.multiprocessing as mp

from torch.utils.data import Subset
import pandas as pd


def all_false(lst):
    return all([not x for x in lst])

def nlen(dataset):
    return len(dataset._path_to_videos)

def reduce_mean(tensor, nprocs):
    if tensor is None or not isinstance(tensor, torch.Tensor):
        print("Warning: tensor is None or not a Tensor!")
        return torch.tensor(0.0)  # 避免错误
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    # print("Reduced tensor:", rt)  # 调试用
    rt /= nprocs  # 计算均值
    if torch.isnan(rt):
        print("Warning: reduced tensor is NaN!")
        return torch.tensor(0.0)  # 避免 NaN
    return rt

def dynamic_spatiotemporal_fusion(prototype_matrix, s_prototype_matrix, t_prototype_matrix):
    """
    动态时空特征融合函数 - 完全自适应，无人工参数
    """
    s_importance = s_prototype_matrix.mean()
    st_importance = prototype_matrix.mean()  # 假设这是时空混合
    t_importance = t_prototype_matrix.mean()

    total_importance = s_importance + st_importance + t_importance

    # 动态计算系数（避免除零）
    epsilon = 1e-8
    s_coeff = s_importance / (s_importance + st_importance + t_importance + epsilon)
    st_coeff = st_importance / (s_importance + st_importance + t_importance + epsilon)  
    t_coeff = t_importance / (s_importance + st_importance + t_importance + epsilon)
    
    # 动态融合
    confusion_factor = (st_coeff * prototype_matrix +
                   s_coeff * s_prototype_matrix +
                   t_coeff * t_prototype_matrix)
    
    return confusion_factor


def build_label_mapping(train_classes):
    """把有标签（已知）类的原始类别号映射到连续区间 [0, num_labeled)。

    原始标签是非连续偶数（如 ucf101 的 0,2,...,100），而 estimate_k 输出维度只有 84，
    直接把原始标签当索引会越界（device-side assert）。这里建一个 raw -> 连续索引 的映射。
    """
    train_classes_sorted = sorted(int(c) for c in train_classes)
    raw_to_new = {raw: new for new, raw in enumerate(train_classes_sorted)}
    return train_classes_sorted, raw_to_new


def remap_tensor_labels(labels, raw_to_new):
    """把 (CPU) 标签张量从原始类别号重映射为连续索引，未知标签置 -1。"""
    labels = labels.cpu()
    new = torch.full_like(labels, -1, dtype=torch.long)
    for raw, n in raw_to_new.items():
        new[labels == raw] = n
    return new


def remap_memory_slots(memory, raw_to_new, num_labeled_classes):
    """重排 memory 槽位，使槽位 i 存放「重映射后标签为 i」的类原型。

    原始 memory 是按 ``槽位 == 原始类别号`` 构建的（仅 0..83 有效，84..100 被丢弃）。
    重映射后需要把 features / s_features / t_features / cls_feature 的行重排，并重置
    mask_init，让已知类占据 [0, num_labeled_classes) 这些槽位。
    """
    new_to_raw = {new: raw for raw, new in raw_to_new.items()}
    num_slots = memory.features.shape[0]
    feat_dim = memory.features.shape[1]
    device = memory.features.device

    def gather(src, default):
        dst = src.clone()
        for new_label in range(num_slots):
            raw = new_to_raw.get(new_label)
            if raw is not None and 0 <= raw < num_slots:
                dst[new_label] = src[raw]
            else:
                dst[new_label] = default
        return dst

    memory.features = gather(memory.features, F.normalize(torch.randn(feat_dim, device=device), dim=0))
    memory.s_features = gather(memory.s_features, F.normalize(torch.randn(feat_dim, device=device), dim=0))
    memory.t_features = gather(memory.t_features, F.normalize(torch.randn(feat_dim, device=device), dim=0))
    memory.cls_feature = gather(memory.cls_feature, torch.zeros(memory.cls_feature.shape[1], device=device))

    mask_init = torch.ones(num_slots, device=device) * -1
    mask_init[:num_labeled_classes] = 1
    memory.mask_init = mask_init

    return memory


def convert_train_label_local(mask_old_new, pse_label, train_loader, raw_to_new):
    """``convert_train_label`` 的本地重写版，额外把有标签的原始标签重映射为连续索引。

    注意：pse_label 来自 classify_ufeats_tube，它基于「已重排过的 memory」的
    get_known_class_prototypes() 生成，本身已经是 [0, num_labeled) 的连续标签，
    所以这里只重映射 l_label（有标签样本的原始标签），pse_label 保持不动。
    """
    l_mask_len = len(train_loader.dataset.labelled_dataset.uq_idxs)
    l_mask = torch.ones(l_mask_len, dtype=torch.bool, device=mask_old_new.device)
    mask = torch.cat([l_mask, mask_old_new], dim=0)

    l_label = train_loader.dataset.labelled_dataset._labels
    if not isinstance(l_label, torch.Tensor):
        l_label = torch.tensor(l_label, device=pse_label.device)
    elif l_label.device != pse_label.device:
        l_label = l_label.to(pse_label.device)

    l_label = remap_tensor_labels(l_label, raw_to_new)

    label = torch.cat([l_label, pse_label])
    return mask, label


def load_backbone_pretrained_correct(backbone, cfg):
    """正确加载预训练权重到 backbone.model。

    两个坑这里都补上：
    1) vit_times(cfg) 内部的 load_pretrained 在 divided_space_time 分支有 bug，会丢掉
       cls_token/pos_embed/mlp/空间注意力等大部分权重。
    2) 不同来源的 checkpoint 键前缀不一样：
         - 原始 TimeSformer: model_state 里是 'model.' 前缀
         - SeFAR: model_state 里是 'vit.model.' 前缀，且额外带 fusion 模块
           (global_fusion/ln*/fc/post_fusion*) 和拆开的 time_embed.0/.1
       这里统一迭代剥离常见包装前缀后加载，其余(融合头/time_embed 等)自动跳过。
    """
    pretrained_model = cfg.TIMESFORMER.PRETRAINED_MODEL
    if not pretrained_model or not os.path.isfile(pretrained_model):
        return backbone

    checkpoint = torch.load(pretrained_model, map_location='cpu')
    if 'model_state' in checkpoint:
        raw = checkpoint['model_state']
    elif 'state_dict' in checkpoint:
        raw = checkpoint['state_dict']
    else:
        raw = checkpoint

    # 迭代剥离常见包装前缀：module. / vit. / model. / backbone. / encoder.
    state_dict = {}
    for k, v in raw.items():
        name = k
        for prefix in ('module.', 'vit.', 'model.', 'backbone.', 'encoder.'):
            if name.startswith(prefix):
                name = name[len(prefix):]
        state_dict[name] = v

    # 取内层 VisionTransformer
    model = backbone.model if hasattr(backbone, 'model') else backbone
    model_keys = model.state_dict()

    # 分类头维度不一致时（estimate_k vs 预训练类别数）丢弃 head，让 head 随机初始化
    if 'head.weight' in state_dict and 'head.weight' in model_keys:
        if state_dict['head.weight'].shape[0] != model_keys['head.weight'].shape[0]:
            del state_dict['head.weight']
            if 'head.bias' in state_dict:
                del state_dict['head.bias']

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    loaded = len(state_dict) - len(unexpected)
    print(f'[load_backbone_pretrained_correct] loaded={loaded} '
          f'missing={len(missing)} unexpected={len(unexpected)}')
    if missing:
        print(f'  missing keys: {missing}')
    return backbone


def epoch_train(label_meter,class_meter,backbone,projector,memory,centers,train_loader,test_loader_unlabelled,class_merge_loader,merge_len,args,cfg):

    if cfg.NUM_GPUS>1:
        local_rank = args.local_rank
    
    # 在训练开始前初始化Excel
    excel_path = os.path.join(cfg.OUTPUT_DIR, 'kmeans_results.xlsx')
    df = pd.DataFrame(columns=['Epoch', 'All', 'Old', 'New', 'All_1', 'Old_1', 'New_1'])
    df.to_excel(excel_path, index=False)
        
    unsupervised_smoothing = 1
    # b_params_groups = get_params_groups(backbone)
    b_optimizer = SGD(backbone.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # p_params_groups = get_params_groups(projector)
    p_optimizer = SGD(projector.parameters(), lr=args.head_lr, momentum=args.momentum, weight_decay=args.weight_decay)

    new_epoch = 0
    
    if args.load_legancy:    
        checkpoint_dir = find_simgcd_latest(os.path.join(cfg.OUTPUT_DIR,'log/checkpoints'))
        if checkpoint_dir is not None:
            if cfg.NUM_GPUS>1:
                multi_gpu = True
            else:
                multi_gpu = False

            backbone, projector, b_optimizer, p_optimizer, epoch = load_simgcd_legancy(backbone, projector, b_optimizer, p_optimizer, checkpoint_dir, multi_gpu=True, strict=False)
            new_epoch = epoch
        else:
            print("no find checkpoint,training...")

    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    b_exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            b_optimizer,
            T_max=args.epochs-new_epoch,
            eta_min=args.lr * 1e-3,
        )
    
    p_exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            p_optimizer,
            T_max=args.epochs-new_epoch,
            eta_min=args.head_lr * 1e-3,
        )
    
    mixup_fn = MixUp(
            num_classes=args.mlp_out_dim
        )

    cluster_criterion = DistillLoss(
                        args.warmup_teacher_temp_epochs,
                        args.epochs,
                        args.n_views,
                        args.warmup_teacher_temp,
                        args.teacher_temp,
                    )
    args.contrastive_cluster_epochs = args.contrastive_cluster_epochs-new_epoch
    contrastive_cluster_weight_schedule = np.concatenate((
        np.linspace(args.contrastive_cluster_weight * 0.1,
                    args.contrastive_cluster_weight, args.contrastive_cluster_epochs),
        np.ones(args.epochs-new_epoch - args.contrastive_cluster_epochs) * args.contrastive_cluster_weight
        ))
    
    sharpener = SharpLogitLoss(T=0.1, warmup_epochs=4, total_epochs=15,current_epoch=new_epoch)

    only_extract_labeled_stprotos(backbone, class_merge_loader, memory,merge_len,args,cfg,local_rank)
        
    mask_old_new, pse_labels,cluster_list,all_weak,u_feats= extract_labeled_stprotos_energy(backbone,class_merge_loader, memory,centers,merge_len,args,cfg,local_rank)
    # memory.manual_update_with_prototypes(all_new_prototypes)
    known_prototypes_dict,known_labels, feature_indices_dict = memory.get_known_class_prototypes('st')
    s_known_prototypes_dict,_,_ = memory.get_known_class_prototypes('space')
    t_known_prototypes_dict,_,_ = memory.get_known_class_prototypes('time')

    mask_old_new,converted_pse_labels = convert_train_label_local(mask_old_new, pse_labels,train_loader,args.raw_to_new)
    similarity_matrix, label_to_idx, idx_to_label = compute_prototype_similarity(known_prototypes_dict,known_labels)
    s_similarity_matrix,_,_ = compute_prototype_similarity(s_known_prototypes_dict,known_labels)
    t_similarity_matrix,_,_ = compute_prototype_similarity(t_known_prototypes_dict,known_labels)

    all_prototype = memory.features.cuda(local_rank,non_blocking=True)

    for epoch in range(args.epochs):

        epoch += new_epoch
                        
        loss_record = AverageMeter()
        backbone.train()
        
        for batch_idx, batch in enumerate(train_loader):
            images,_, class_labels, uq_idxs, mask_lab,item = batch

            class_labels = remap_tensor_labels(class_labels, args.raw_to_new)
            class_labels, mask_lab = class_labels.cuda(local_rank,non_blocking=True), mask_lab.cuda(local_rank,non_blocking=True).bool()
            images = images.cuda(non_blocking=True)
            mask_lab =mask_lab.squeeze(1)

            # mask_lab = mask_lab[item].cuda(local_rank,non_blocking=True)
            # class_labels = label[item].cuda(local_rank,non_blocking=True)
            
            is_all_unlabel = all_false(mask_lab)

            with torch.cuda.amp.autocast(fp16_scaler is not None):
                backbone_feature,cls_x,t_feat,s_feat = backbone(images)
                student_proj, student_out = projector(backbone_feature)
                teacher_out = student_out.detach()

                if not is_all_unlabel:
                    sup_logits = (student_out[mask_lab] / 0.1)
                    sup_labels = class_labels[mask_lab]
                    cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)
                    loss_sharp = sharpener(sup_logits,sup_labels,memory.cls_feature.cuda(local_rank,non_blocking=True))
                else:
                    # 全无标签的 batch：监督损失置 0，但仍正常 forward/backward，保证两 rank 通信同步
                    cls_loss = torch.zeros((), device=student_out.device)
                    loss_sharp = torch.zeros((), device=student_out.device)
                
                cluster_loss = cluster_criterion(student_out, teacher_out, epoch,cfg)
                avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
                me_max_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
                cluster_loss += args.memax_weight * me_max_loss
                
                if cfg.cst_feature:
                    new_backbone_feature = feature_dropout(backbone_feature)
                    backbone_feature = torch.cat((backbone_feature,new_backbone_feature),dim=0)
                    backbone_feature = torch.nn.functional.normalize(backbone_feature, dim=-1)
                else:
                    new_cls_x = feature_dropout(cls_x)
                    backbone_feature = torch.cat((cls_x,new_cls_x),dim=0)
                    backbone_feature = torch.nn.functional.normalize(backbone_feature, dim=-1)

                #prototype_confusion
                mask_old = mask_old_new[item.cpu()]
                mask_old = torch.cat([mask_old,mask_old])
                pse_ = converted_pse_labels[item.cpu()]
                pse_ = torch.cat([pse_,pse_])

                mask_old_new = mask_old_new.cuda(local_rank,non_blocking=True)
                converted_pse_labels = converted_pse_labels.cuda(local_rank,non_blocking=True)
                similarity_matrix = similarity_matrix.cuda(local_rank,non_blocking=True)
                s_similarity_matrix = s_similarity_matrix.cuda(local_rank,non_blocking=True)
                t_similarity_matrix = t_similarity_matrix.cuda(local_rank,non_blocking=True)
            
                prototype_matrix = compute_prototype_matrix(item,mask_old, pse_,label_to_idx,idx_to_label,similarity_matrix,local_rank)
                s_prototype_matrix = compute_prototype_matrix(item,mask_old, pse_,label_to_idx,idx_to_label,s_similarity_matrix,local_rank)
                t_prototype_matrix = compute_prototype_matrix(item,mask_old, pse_,label_to_idx,idx_to_label,t_similarity_matrix,local_rank)
                confusion_factor = dynamic_spatiotemporal_fusion(prototype_matrix, s_prototype_matrix, t_prototype_matrix)
                
                confusion_factor = (confusion_factor - confusion_factor.min()) / (
                        confusion_factor.max() - confusion_factor.min() + 0.0000001)
                confusion_factor = confusion_factor / confusion_factor.sum(dim=1)

                torch.cuda.empty_cache()

                contrastive_logits, contrastive_labels, similarity= info_nce_logits_for(features=backbone_feature, confusion_factor=confusion_factor,local_rank=local_rank, args=args)
                
                contrastive_loss = LabelSmoothingLoss()(contrastive_logits, contrastive_labels, similarity, unsupervised_smoothing)
                
                pstr = ''
                pstr += f'cls_loss: {cls_loss.item():.4f} '
                pstr += f'cluster_loss: {cluster_loss.item():.4f} '
                pstr += f'sharpen_loss: {loss_sharp.item():.4f} '
                pstr += f'contrastive_loss: {contrastive_loss.item():.4f} '

                loss = 0
                base_contrast_loss =0
                loss += (1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss
                concate_labels = class_labels.cuda(local_rank,non_blocking=True)

                if not is_all_unlabel:
                    known_label = concate_labels[mask_lab]
                    known_memory = student_out[mask_lab]
                    memory_loss = memory(known_memory,known_label)
                    base_memory_loss = (args.sup_weight)*(memory_loss.cuda(local_rank,non_blocking=True)+0.5*loss_sharp)
                else:
                    base_memory_loss = torch.zeros((), device=student_out.device)
                
                loss += base_memory_loss
                loss += (1-args.sup_weight)*contrastive_loss
                pstr += f'memory_loss: {base_memory_loss.item():.4f} '

                # if batch_idx>100:
                #     break
                
            reduced_loss = reduce_mean(loss,args.nprocs)
            loss_record.update(reduced_loss.item(), class_labels.size(0))

            b_optimizer.zero_grad()
            p_optimizer.zero_grad()

            if fp16_scaler is None:
                loss.backward()
                b_optimizer.step()
                p_optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                fp16_scaler.step(b_optimizer)
                fp16_scaler.step(p_optimizer)
                fp16_scaler.update()

            if batch_idx % 100 == 0:
                args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'
                            .format(epoch, batch_idx, len(train_loader), loss.item(), pstr))

        args.logger.info('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))

        args.writer.add_scalar('Loss', loss_record.avg, epoch)

        #args.writer.add_scalar('Train Acc Labelled Data', train_acc_record.avg, epoch)
        args.writer.add_scalar('Backbone_LR', get_mean_lr(b_optimizer), epoch)
        args.writer.add_scalar('Head_LR', get_mean_lr(p_optimizer), epoch)

        args.logger.info('Testing on unlabelled examples in the training data...')

        if local_rank == 0 and epoch%1==0:
            # all_acc, old_acc,new_acc = test(label_meter,backbone,unlabelled_train_loader, epoch=epoch, save_name='Train ACC Unlabelled', args=args,cfg=cfg)

            # args.writer.add_scalars('sim_times',{'Old': old_acc, 'New': new_acc,'All': all_acc,}, epoch)
            # args.logger.info('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
            print("Kmeans result...")
            all_acc, old_acc, new_acc = test_kmeans(backbone,test_loader_unlabelled, epoch,"Test all acc", args,projection_head=None, Use_GPU=False)
            args.logger.info('kmeans acc: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
            # all_acc_1, old_acc_1, new_acc_1 = test_kmeans(backbone,test_loader_unlabelled, epoch,"Test all acc", args,projection_head=None, Use_GPU=True)
            # args.logger.info('gpu kmeans acc: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_1, old_acc_1, new_acc_1))
            df = pd.read_excel(excel_path)
            new_row = {
                'Epoch': epoch,
                'All': all_acc, 'Old': old_acc, 'New': new_acc
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_excel(excel_path, index=False)


        b_exp_lr_scheduler.step()
        p_exp_lr_scheduler.step()

        # 每个 epoch 保存一次 checkpoint，只保留最新一个（避免占满磁盘）。
        # 机器崩了之后，下次启动会自动 find_simgcd_latest 续训（load_legancy 默认开）。
        if args.local_rank == 0:
            # 先删掉旧的 checkpoint，只留最新
            if os.path.isdir(args.model_dir):
                for f in os.listdir(args.model_dir):
                    if f.startswith('model_epoch_') and f.endswith('.pt'):
                        try:
                            os.remove(os.path.join(args.model_dir, f))
                        except Exception:
                            pass

            save_dict = {
                'backbone': backbone.state_dict(),
                'projector': projector.state_dict(),
                'b_optimizer': b_optimizer.state_dict(),
                'p_optimizer': p_optimizer.state_dict(),
                'epoch': epoch + 1,
            }
            save_path = "{0}/model_epoch_{1}.pt".format(args.model_dir, epoch)
            torch.save(save_dict, save_path)
            args.logger.info("model saved to {}.".format(save_path))


def test(test_meter,backbone,test_loader, epoch, save_name, args,cfg):

    #model.eval()
    if cfg.NUM_GPUS>1:
        local_rank = args.local_rank

    backbone.eval()
    # projector.eval()

    #test_meter.iter_tic()

    mask = np.array([])
    for batch_idx, batch in enumerate(tqdm(test_loader)):
        #images = images.cuda(non_blocking=True)
        images, label, uq_idxs,index = batch

        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(images, (list,)):
                for i in range(len(images)):
                    images[i] = images[i].cuda(local_rank,non_blocking=True)
            else:
                images = images.cuda(local_rank,non_blocking=True)
            label = label.cuda()
            uq_idxs = uq_idxs.cuda()
            index = index.cuda()


        with torch.no_grad():
            if isinstance(images, (list,)):
                images = images[0] 
            backbone_feature,cls_x,q,st_feature = backbone(images)

            mask = torch.tensor(np.array([True if x.item() in args.train_classes else False for x in label]))

            cls_x = cls_x.cpu()

            test_meter.update_stats(
                cls_x, label, mask,index
            )

    all_acc, old_acc,new_acc = test_meter.finalize_metrics()

    return all_acc, old_acc,new_acc

def test_kmeans(model, test_loader, epoch,save_name, args,projection_head=None, Use_GPU=True):

    local_rank = 0

    model.eval()
    all_feats = []

    targets = []
    mask = np.array([])

    # First extract all features
    with torch.no_grad():
        for batch_idx, (images,_, label, uq_idx,_) in enumerate(tqdm(test_loader)):
            
            images = images.cuda()
            backbone_feature,cls_x,x_space,x_time = model(images)
            all_feats.append(backbone_feature.detach().cpu().numpy())
            targets.append(label.numpy())
            mask = np.append(mask, np.array([True if x.item() in args.train_classes
                                            else False for x in label]))
        mask = mask.astype(bool)
        all_feats = np.concatenate(all_feats)
        targets = np.concatenate(targets)

    if Use_GPU:
        device = torch.device(f"cuda:{local_rank}")
        preds, prototypes = kmeans(X=torch.from_numpy(all_feats).cuda(local_rank,non_blocking=True), num_clusters=args.num_unlabeled_classes+args.num_labeled_classes,
                                       distance='euclidean', device=device, tqdm_flag=False)
        preds, prototypes = preds.cpu().numpy(), prototypes.cpu().numpy()
        
    else:
        kmeanss = KMeans(n_clusters=args.num_labeled_classes + args.num_unlabeled_classes, random_state=0).fit(all_feats)
        preds = kmeanss.labels_


    all_acc, old_acc, new_acc = split_cluster_acc_v2_kmeans(y_true=targets, y_pred=preds, mask=mask)

    print('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
    return all_acc, old_acc, new_acc



#if __name__ == "__main__":

def train_net(local_rank,nprocs, args):

    args.local_rank = local_rank #当前gpu的rank, such as 0, or 1 or 2...
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:23485', world_size=args.nprocs, rank=local_rank,timeout=datetime.timedelta(seconds=1800) ) # TODO!
    

    cfg = load_config(args)

    # estimate_k 实验的结果统一输出到 result/tube/xstar/estimate_k+<数据集>
    # （运行时覆盖，不改共享 yaml，避免影响原本程序）
    cfg.OUTPUT_DIR = f'/mnt/newdisk/mnt/newdisk/result/tube/xstar/estimate_k+{args.dataset_name}'

    # NUM_GPUS 对齐实际 GPU 数（nprocs），避免 config 写错导致 batch/通信不一致
    cfg.NUM_GPUS = args.nprocs

    # 数据集级路径修正：部分 config 里是过期路径/缺失字段，这里运行时覆盖，不碰共享 yaml
    if args.dataset_name == 'ssv2':
        cfg.TRAIN.BATCH_SIZE = 32  # 每卡 16，加快训练（原配置 8 太慢）
        cfg.DATA.SSv2_PATH_TO_DATA_DIR = '/mnt/newdisk/mnt/newdisk/dataset/action/ssv2_all/label/initial/v15/'
        cfg.DATA.SSv2_PATH_PREFIX = '/mnt/newdisk/mnt/newdisk/dataset/action/ssv2_all/ssv2_frame_v15/'
        cfg.TIMESFORMER.PRETRAINED_MODEL = '/mnt/newdisk/mnt/newdisk/result/action/timesformer/4_7_ssv2_v15/checkpoints/checkpoint_epoch_00007.pyth'
        cfg.memory_dir = '/mnt/newdisk/mnt/newdisk/weight/DGC/tube/kmeans/ssv2/memory_ssv2.pth'
        cfg.logit_dir = '/mnt/newdisk/mnt/newdisk/weight/DGC/tube/kmeans/ssv2/ssv2_251109_logit.pt'
    elif args.dataset_name == 'ibc127':
        cfg.TIMESFORMER.PRETRAINED_MODEL = '/mnt/newdisk/mnt/newdisk/result/action/timesformer/3_30_ibc127/checkpoints/checkpoint_epoch_00005.pyth'

    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.interpolation = 3
    args.crop_pct = 0.875
    

    args = get_class_splits(args,cfg)
    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    # args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes
    args.mlp_out_dim = args.estimate_k

    # 关键：让 memory 槽位数(NUM_CLASSES)与投影头输出维(estimate_k)一致，
    # 否则 memory 损失的 known_memory.mm(features) 会维度不匹配。
    # 这里在运行时覆盖，不改共享 yaml，避免影响原本(非 estimate_k)程序。
    cfg.MODEL.NUM_CLASSES = args.estimate_k

    # 建 raw -> 连续索引 的映射，供整个训练过程统一重映射标签
    _, args.raw_to_new = build_label_mapping(args.train_classes)

    init_experiment(args,cfg,runner_name=['simgcd'])
    args.logger.info(f'Using evaluation function {args.eval_funcs[0]} to print results')
    
    #torch.backends.cudnn.benchmark = True

    backbone = vit_times(cfg)
    projector = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)

    # 正确加载 TimeSformer 预训练权重（vit_times 内部的 load_pretrained 有 bug 会丢权重）
    backbone = load_backbone_pretrained_correct(backbone, cfg)

    if args.warmup_model_dir is not None:
        args.logger.info(f'Loading weights from {args.warmup_model_dir}')
        backbone.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))

    torch.cuda.set_device(local_rank)

    backbone.cuda(local_rank)
    projector.cuda(local_rank)
    
    cudnn.benchmark = True
    """
    if args.weight_path is not None:
        cu.load_checkpoint(args.weight_path, backbone,data_parallel=False)
    """

    for m in backbone.parameters():
        m.requires_grad = True

    for m in projector.parameters():
        m.requires_grad = True
    
    args.logger.info('model build')

    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)
    
    train_dataset, test_dataset, unlabelled_train_examples_test, class_dataset,class_label_dataset,class_merge_dataset,datasets = get_datasets(cfg,args.dataset_name,
                                                                                         train_transform,
                                                                                         test_transform,
                                                                                         args)
                                                            
    train_loader = loader.construct_loader(cfg, "train",train_dataset)
    test_loader_unlabelled = loader.construct_loader(cfg, "test",unlabelled_train_examples_test)
    class_loader = loader.construct_loader(cfg, "class",class_label_dataset)
    
    class_merge_loader = loader.construct_loader(cfg, "test",train_dataset)
    
    merge_len = len(train_loader.dataset.unlabelled_dataset.data)+len(train_loader.dataset.labelled_dataset.data)

   
    memory1 = None
    memory2 = None
   
    class_meter1 = None
    class_meter2 = None
    
    all_feat_memory = 800
    
    if local_rank ==0:

        class_meter1 = TestMeter(
        len(class_loader.dataset.data)// (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TRAIN.NUM_SPATIAL_CROPS),
        cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TRAIN.NUM_SPATIAL_CROPS,
        cfg.MODEL.NUM_CLASSES,
        len(class_loader),
        cfg.DATA.MULTI_LABEL,
        cfg.DATA.ENSEMBLE_METHOD)

        memory1 = ClusterMemory(args.feat_dim, cfg.MODEL.NUM_CLASSES,local_rank,args.train_classes,temp=args.temp,
                            momentum=args.memory_momentum, use_hard=args.use_hard)
        
        memory = memory1
        class_meter = class_meter1
        

    elif local_rank ==1:
        class_meter2 = TestMeter(
        len(class_loader.dataset.data)// (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TRAIN.NUM_SPATIAL_CROPS),
        cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TRAIN.NUM_SPATIAL_CROPS,
        cfg.MODEL.NUM_CLASSES,
        len(class_loader),
        cfg.DATA.MULTI_LABEL,
        cfg.DATA.ENSEMBLE_METHOD)

        memory2 = ClusterMemory(args.feat_dim, cfg.MODEL.NUM_CLASSES,local_rank,args.train_classes,temp=args.temp,
                            momentum=args.memory_momentum, use_hard=args.use_hard)
        memory = memory2
        class_meter = class_meter2
        
    if cfg.NUM_GPUS > 1:
        # Make model replica operate on the current device
        backbone = torch.nn.parallel.DistributedDataParallel(
            module=backbone, device_ids=[local_rank],find_unused_parameters=True
        )
        projector = torch.nn.parallel.DistributedDataParallel(
            module=projector, device_ids=[local_rank],find_unused_parameters=True
        )

    label_meter = TestMeter(
        len(test_loader_unlabelled.dataset.data)// (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TRAIN.NUM_SPATIAL_CROPS),
        cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TRAIN.NUM_SPATIAL_CROPS,
        cfg.MODEL.NUM_CLASSES,
        len(test_loader_unlabelled),
        cfg.DATA.MULTI_LABEL,
        cfg.DATA.ENSEMBLE_METHOD,
    )

    # memory = ClusterMemory.load(load_path, device)
            
    memory,centers = class_vector_sup(backbone,projector,memory,class_loader,class_meter,args,cfg)
    # memory,centers = class_vector(memory,cfg)
    memory._init_prompt_with_feature()

    # 把 memory 槽位重排到重映射后的连续标签空间，否则 memory 损失/原型仍按原始标签越界
    memory = remap_memory_slots(memory, args.raw_to_new, args.num_labeled_classes)

    epoch_train(label_meter,class_meter,backbone,projector,memory,centers,train_loader,test_loader_unlabelled,class_merge_loader,merge_len,args,cfg)

def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    args.nprocs = torch.cuda.device_count()

    mp.spawn(train_net,nprocs=args.nprocs,args=(args.nprocs,args))


if __name__ == "__main__":
    main()