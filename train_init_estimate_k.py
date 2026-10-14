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
from util.cluster_and_log_utils import log_accs_from_preds,split_cluster_acc_v2_kmeans,test_kmeans_all_dimension
from util.k_means_utils import test_kmeans_semi_sup,test_kmeans
from util.misc import launch_job
from util.parser import load_config, parse_args
from util.general_utils import AverageMeter, init_experiment
from sklearn.cluster import KMeans
from kmeans_pytorch.kmeans_pytorch import kmeans

from torch.utils.tensorboard import SummaryWriter

from util.cluster_memory_utils import extract_features,generate_cluster_features,extract_labeled_protos,extract_labeled_stprotos,extract_labeled_stprotos_energy,feature_dropout,kmeansMLPProjector,class_vector_sup,class_vector,extract_labeled_stprotos_simple,compute_prototype_similarity,compute_prototype_matrix,unsupervised_prototype

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

def epoch_train(label_meter,class_meter,backbone,projector,memory,centers,train_loader,test_loader_unlabelled,class_merge_loader,merge_len,args,cfg):

    if cfg.NUM_GPUS>1:
        local_rank = args.local_rank
    
    # 在训练开始前初始化Excel
    excel_path = os.path.join(cfg.OUTPUT_DIR, 'kmeans_results.xlsx')
    df = pd.DataFrame(columns=['Epoch', 'All', 'Old', 'New','All_cnn_clsx','Old_cnn_clsx','New_cnn_clsx','All_cnn_logits','Old_cnn_logits','New_cnn_logits' 'All_gpu', 'Old_gpu', 'New_gpu', 'All_norm_logit', 'Old_norm_logit', 'New_norm_logit','All_proj', 'Old_proj', 'New_proj', 'All_feat_balance', 'Old_feat_balance', 'New_feat_balance'])
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

    all_prototype = memory.features.cuda(local_rank,non_blocking=True)

    for epoch in range(args.epochs):

        epoch += new_epoch
                        
        loss_record = AverageMeter()
        backbone.train()
        
        for batch_idx, batch in enumerate(train_loader):

            # if batch_idx>2:
            #     break
            images,_, class_labels, uq_idxs, mask_lab,item = batch

            class_labels, mask_lab = class_labels.cuda(local_rank,non_blocking=True), mask_lab.cuda(local_rank,non_blocking=True).bool()
            images = images.cuda(non_blocking=True)
            mask_lab =mask_lab.squeeze(1)

            is_all_unlabel = all_false(mask_lab)
            if is_all_unlabel:
                continue

            with torch.cuda.amp.autocast(fp16_scaler is not None):
                backbone_feature,cls_x,t_feat,s_feat = backbone(images)
                student_proj, student_out = projector(backbone_feature)
                teacher_out = student_out.detach()

                sup_logits = (student_out[mask_lab] / 0.1)  
                sup_labels = class_labels[mask_lab]    
                cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)
                
                loss_sharp = sharpener(sup_logits,sup_labels,memory.cls_feature.cuda(local_rank,non_blocking=True))
                
                cluster_loss = cluster_criterion(student_out, teacher_out, epoch,cfg)
                avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
                me_max_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
                cluster_loss += args.memax_weight * me_max_loss
                
                pstr = ''
                pstr += f'cls_loss: {cls_loss.item():.4f} '
                pstr += f'cluster_loss: {cluster_loss.item():.4f} '
                # pstr += f'sharpen_loss: {loss_sharp.item():.4f} '
                # pstr += f'contrastive_loss: {contrastive_loss.item():.4f} '

                loss = 0
                base_contrast_loss =0
                loss += (1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss
                concate_labels = class_labels.cuda(local_rank,non_blocking=True)
                

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
            
            print("Kmeans result...")
            all_acc_f, old_acc_f, new_acc_f = test_kmeans_all_dimension(backbone,df,excel_path,test_loader_unlabelled,label_meter, epoch=epoch, save_name='Train ACC Unlabelled',args=args,projection_head=projector)
            args.logger.info('All KMeans .. Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_f, old_acc_f, new_acc_f))

            # test(label_meter,backbone,unlabelled_train_loader, epoch,'Train ACC Unlabelled', args,cfg)

        b_exp_lr_scheduler.step()
        p_exp_lr_scheduler.step()

        save_dict = {
            'backbone': backbone.state_dict(),
            'projector': projector.state_dict(),
            'b_optimizer': b_optimizer.state_dict(),
            'p_optimizer': p_optimizer.state_dict(),
            'epoch': epoch + 1,
        }
        if epoch % 2 ==0 and args.local_rank == 0:
            save_path = "{0}/model_epoch_{1}.pt".format(args.model_dir,epoch)
            
            torch.save(save_dict, save_path)
            args.logger.info("model saved to {}.".format(save_path))


def test(test_meter,backbone,projector,test_loader, epoch, save_name, args,cfg):

    #model.eval()
    if cfg.NUM_GPUS>1:
        local_rank = args.local_rank

    backbone.eval()
    projector.eval()

    #test_meter.iter_tic()
    new_test_meter= test_meter

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
            backbone_feature,cls_x,_,_ = backbone(images)
            proj,logits = projector(backbone_feature)

            mask = torch.tensor(np.array([True if x.item() in args.train_classes else False for x in label]))

            cls_x = cls_x.cpu()

            test_meter.update_stats(
                cls_x, label, mask,index
            )

            new_test_meter.update_stats(
                logits, label, mask,index
            )

    all_acc, old_acc,new_acc = test_meter.finalize_metrics()

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

    init_experiment(args,cfg,runner_name=['simgcd'])
    args.logger.info(f'Using evaluation function {args.eval_funcs[0]} to print results')
    
    #torch.backends.cudnn.benchmark = True

    if args.warmup_model_dir is not None:
        args.logger.info(f'Loading weights from {args.warmup_model_dir}')
        backbone.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))


    backbone = vit_times(cfg)
    projector = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)

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
            
    # memory,centers = class_vector_sup(backbone,projector,memory,class_loader,class_meter,args,cfg)
    memory,centers = class_vector(memory,cfg)
    memory._init_prompt_with_feature()

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