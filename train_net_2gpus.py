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
from util.misc import launch_job
from util.parser import load_config, parse_args
from util.general_utils import AverageMeter, init_experiment
from sklearn.cluster import KMeans
from kmeans_pytorch.kmeans_pytorch import kmeans

from torch.utils.tensorboard import SummaryWriter

from util.cluster_memory_utils import extract_labeled_protos,feature_dropout,class_vector
from util.loss import DINOHead,  DistillLoss, ContrastiveLearningViewGenerator, get_params_groups,info_nce_logits_for,LabelSmoothingLoss,SharpLogitLoss

import argparse
import random
from backbone.vit import vit_base_patch16_224 as vit_times
from backbone.vit import TimeSformer as TimeSformer
from backbone.Timesformer_defaults import get_timesformer_cfg
from backbone.Timesformer_defaults import _assert_and_infer_cfg,load_config
from backbone import checkpoint as cu
from backbone.checkpoint import find_simgcd_latest,load_simgcd_legancy
from util import distributed as du
from util.cluster_memory_utils import ClusterMemory,load_results,save_results
from torch.nn import functional as F
from copy import deepcopy

from data.get_datasets import get_datasets, get_class_splits
from data import loader
from util.meters import TestMeter

import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data.distributed
import torch.multiprocessing as mp

from torch.utils.data import Subset

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


def epoch_train(label_meter,backbone,projector,memory,train_loader,unlabelled_train_loader,class_merge_loader,merge_len, args,cfg):

    if cfg.NUM_GPUS>1:
        local_rank = args.local_rank
        
    if args.use_vote:
        if cfg.kmeans_dir is not None:
            results = load_results(cfg.kmeans_dir, local_rank)  # 或 "cpu" 视需求而定
            uq_index, all_preds, cluster_protos_list, preds_ind_list = (
                results["ids"], results["all_preds"], results["prototype_higher"], results["preds_higher"]
            )  
        else:
            uq_index, all_preds, cluster_protos_list, preds_ind_list = extract_labeled_protos(backbone,class_merge_loader, merge_len,args,local_rank)
            save_results(uq_index, all_preds, cluster_protos_list, preds_ind_list, save_path=cfg.OUTPUT_DIR+args.dataset_name+"_kmeans.pth")
            
        for i in range(len(preds_ind_list)):
            preds_ind_list[i] = preds_ind_list[i].cuda(local_rank,non_blocking=True).long()
            cluster_protos_list[i] = cluster_protos_list[i].cuda(local_rank,non_blocking=True)

            cluster_protos_list[i]=cluster_protos_list[i]/torch.norm(cluster_protos_list[i],dim=1).unsqueeze(1)

            cluster_distances_list=[]
            cluster_radius_list=[]
        for i in range(len(preds_ind_list)):
            cluster_distances = torch.cdist(cluster_protos_list[i], cluster_protos_list[i])

            cluster_distances_list.append(cluster_distances.clone())
            cluster_radius = \
            (cluster_distances + torch.eye(cluster_distances.shape[0]).cuda(local_rank,non_blocking=True) * cluster_distances.max()).min(dim=1)[0] / 2
            cluster_radius_list.append(cluster_radius.clone())
            
    else:
        uq_index, preds_ind_list,cluster_protos_list,cluster_distances_list,cluster_radius_list = None,None,None,None,None
        
    cluster_momentum = 0.7
    unsupervised_smoothing = 0.8
    b_optimizer = SGD(backbone.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    p_optimizer = SGD(projector.parameters(), lr=args.head_lr, momentum=args.momentum, weight_decay=args.weight_decay)

    new_epoch = 0
    
    if args.load_legancy:    
        checkpoint_dir = find_simgcd_latest(os.path.join(cfg.OUTPUT_DIR,'log/checkpoints'))
        if checkpoint_dir is not None:
            if cfg.NUM_GPUS>1:
                multi_gpu = True
            else:
                multi_gpu = False
            backbone, projector, b_optimizer, p_optimizer, epoch = load_simgcd_legancy(backbone, projector, b_optimizer, p_optimizer, checkpoint_dir, multi_gpu=multi_gpu, strict=False)
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

    cluster_criterion = DistillLoss(
                        args.warmup_teacher_temp_epochs-new_epoch,
                        args.epochs-new_epoch,
                        args.n_views,
                        args.warmup_teacher_temp,
                        args.teacher_temp,
                    )
    args.contrastive_cluster_epochs = args.contrastive_cluster_epochs-new_epoch
    
    sharpener = SharpLogitLoss(T=0.1, warmup_epochs=4, total_epochs=15,current_epoch=new_epoch)
    
    for epoch in range(args.epochs):

        epoch += new_epoch
                        
        loss_record = AverageMeter()
        
        backbone.train()
        
        for batch_idx, batch in enumerate(train_loader):
            images, class_labels, uq_idxs, mask_lab,item = batch

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
                
                cluster_loss = cluster_criterion(student_out, teacher_out, epoch)
                avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
                me_max_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
                cluster_loss += args.memax_weight * me_max_loss
                
                new_backbone_feature = feature_dropout(backbone_feature)
                backbone_feature = torch.cat((backbone_feature,new_backbone_feature),dim=0)
                backbone_feature = torch.nn.functional.normalize(backbone_feature, dim=-1)

                with torch.no_grad():
                    confusion_factor=0
                    pair_dist = torch.cdist(backbone_feature,backbone_feature)
                    
                    n_labeled = args.num_labeled_classes
                    n_unlabeled = args.num_unlabeled_classes

                    for i in range(len(preds_ind_list)):
                        if i >5:
                            continue
                        cluster_labels=(preds_ind_list[i][np.argsort(uq_index)[item]]).clone()

                        if i >2:
                            n_labeled = max(int(n_labeled / 2), 1)
                            n_unlabeled = max(int(n_unlabeled / 2), 1)
                        
                        cluster_indexer = F.one_hot(cluster_labels.long(), n_labeled +n_unlabeled ).float().T
                        
                        cluster_indexer = torch.cat([cluster_indexer,cluster_indexer],dim=1)
                        n_samples = torch.sum(cluster_indexer, dim=1).unsqueeze(1)
                        n_samples[n_samples == 0] = 1

                        distance = torch.cdist(backbone_feature, cluster_protos_list[i].float())
                        
                        cluster_radius_list[i]= (cluster_indexer*distance.T).sum(dim=1)/n_samples.squeeze()\
                                                *(1-cluster_momentum)+cluster_radius_list[i]* cluster_momentum

                        cluster_labels = torch.cat([cluster_labels,cluster_labels])

                        pair_dist = (pair_dist - pair_dist.min()) / (pair_dist.max() - pair_dist.min() + 1e-7)
                        pair_dist = pair_dist *2* (cluster_radius_list[i].max() - cluster_radius_list[i].min()) + cluster_radius_list[i].min()
                        
                        confusion_factor+=(pair_dist>2*cluster_radius_list[i][cluster_labels]).float()/2 ** i

                confusion_factor = (confusion_factor - confusion_factor.min()) / (
                        confusion_factor.max() - confusion_factor.min() + 0.0000001)
                confusion_factor = confusion_factor / confusion_factor.sum(dim=1)

                torch.cuda.empty_cache()

                contrastive_logits, contrastive_labels, similarity= info_nce_logits_for(features=backbone_feature, confusion_factor=confusion_factor,local_rank=local_rank, args=args)
                
                contrastive_loss = LabelSmoothingLoss()(contrastive_logits, contrastive_labels, similarity, unsupervised_smoothing)
                
                #memory
                concate_labels = class_labels.cuda(local_rank,non_blocking=True)
                sup_cls_x = cls_x[mask_lab].cuda(local_rank,non_blocking=True)
                
                known_label = concate_labels[mask_lab]
                known_memory = student_out[mask_lab]
                memory_loss = memory(known_memory,sup_cls_x,known_label)
                base_memory_loss= memory_loss.cuda(local_rank,non_blocking=True) +args.sharp_weight*loss_sharp
                
                pstr = ''
                pstr += f'cls_loss: {cls_loss.item():.4f} '
                pstr += f'cluster_loss: {cluster_loss.item():.4f} '
                # pstr += f'sharpen_loss: {loss_sharp.item():.4f} '
                pstr += f'contrastive_loss: {contrastive_loss.item():.4f} '
                pstr += f'memory_loss: {base_memory_loss.item():.4f}'
                
                loss = 0
                loss += (1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss
                loss += (1-args.sup_weight)*contrastive_loss + (args.sup_weight)*base_memory_loss
                
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
        args.writer.add_scalar('Backbone_LR', get_mean_lr(b_optimizer), epoch)
        args.writer.add_scalar('Head_LR', get_mean_lr(p_optimizer), epoch)

        args.logger.info('Testing on unlabelled examples in the training data...')

        if local_rank == 0 and epoch%1==0:
            # all_acc, old_acc,new_acc = test(label_meter,backbone,unlabelled_train_loader, epoch=epoch, save_name='Train ACC Unlabelled', args=args,cfg=cfg)

            # args.writer.add_scalars('sim_times',{'Old': old_acc, 'New': new_acc,'All': all_acc,}, epoch)
            # args.logger.info('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
            print("Kmeans result...")
            all_acc, old_acc, new_acc = test_kmeans(backbone,unlabelled_train_loader, epoch,"Test all acc", args,projection_head=None, Use_GPU=False)
            args.logger.info('kmeans acc: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))

    
        b_exp_lr_scheduler.step()
        p_exp_lr_scheduler.step()

        save_dict = {
            'backbone': backbone.state_dict(),
            'projector': projector.state_dict(),
            'b_optimizer': b_optimizer.state_dict(),
            'p_optimizer': p_optimizer.state_dict(),
            'epoch': epoch + 1,
        }
        if epoch % 1 ==0 and args.local_rank == 0:
            save_path = "{0}/model_epoch_{1}.pt".format(args.model_dir,epoch)
            
            torch.save(save_dict, save_path)
            args.logger.info("model saved to {}.".format(save_path))


def test_kmeans(model, test_loader, epoch,save_name, args,projection_head=None, Use_GPU=True):

    local_rank = 0
    model.eval()
    all_feats = []
    targets = []
    mask = np.array([])

    with torch.no_grad():
        for batch_idx, (images, label, uq_idx,_) in enumerate(tqdm(test_loader)):
            
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


def train_net(local_rank,nprocs, args):

    args.local_rank = local_rank #当前gpu的rank, such as 0, or 1 or 2...
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:23485', world_size=args.nprocs, rank=local_rank) # TODO!

    cfg = load_config(args)
        
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.interpolation = 3
    args.crop_pct = 0.875
    

    args = get_class_splits(args,cfg)
    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes

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
    
    train_dataset, test_dataset, unlabelled_train_examples_test, class_unlabel_dataset,class_label_dataset,class_merge_dataset,datasets = get_datasets(cfg,args.dataset_name,
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
    
    all_feat_memory = args.max_capacity
    
    if local_rank ==0:
        class_meter1 = TestMeter(
        len(class_loader.dataset.data)// (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TRAIN.NUM_SPATIAL_CROPS),
        cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TRAIN.NUM_SPATIAL_CROPS,
        cfg.MODEL.NUM_CLASSES,
        len(class_loader),
        cfg.DATA.MULTI_LABEL,
        cfg.DATA.ENSEMBLE_METHOD)

        memory1 = ClusterMemory(args.feat_dim, cfg.MODEL.NUM_CLASSES,local_rank,temp=args.temp,
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

        memory2 = ClusterMemory(args.feat_dim, cfg.MODEL.NUM_CLASSES,local_rank,temp=args.temp,
                            momentum=args.memory_momentum, use_hard=args.use_hard)
        memory = memory2
        class_meter = class_meter2
        
    if cfg.NUM_GPUS > 1:
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
    
            
    memory,links = class_vector(backbone,projector,memory,class_loader,class_meter,args,cfg)
   
    epoch_train(label_meter,backbone,projector,memory,train_loader,test_loader_unlabelled,class_merge_loader,merge_len,args,cfg)
    

def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    args.nprocs = torch.cuda.device_count()

    mp.spawn(train_net,nprocs=args.nprocs,args=(args.nprocs,args))


if __name__ == "__main__":
    main()