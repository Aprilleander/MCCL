import argparse
import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from util.general_utils import AverageMeter, init_experiment,get_mean_lr
from util.cluster_and_log_utils import log_accs_from_preds,split_cluster_acc_v2_kmeans
from util.k_means_utils import test_kmeans_semi_sup,test_kmeans
from kmeans_pytorch.kmeans_pytorch import kmeans
from sklearn.cluster import KMeans



from util.misc import launch_job
from util.parser import load_config, parse_args
from util.general_utils import AverageMeter, init_experiment

from torch.utils.tensorboard import SummaryWriter

from util.cluster_memory_utils import extract_features,generate_cluster_features

from config import exp_root
from model import DINOHead, info_nce_logits, SupConLoss, DistillLoss, ContrastiveLearningViewGenerator, get_params_groups
from backbone import mvit

import argparse
import random

from backbone.vit import vit_base_patch16_224 as vit_times
from backbone.vit import TimeSformer as TimeSformer
from backbone.Timesformer_defaults import get_timesformer_cfg
from backbone.Timesformer_defaults import _assert_and_infer_cfg,load_config
from backbone.checkpoint import load_simgcd_pretrained

from backbone import checkpoint as cu
from util import distributed as du
from util.cluster_memory_utils import ClusterMemory
from torch.nn import functional as F
from copy import deepcopy

from data.mixup import MixUp
from data.get_datasets import get_datasets, get_class_splits
from data import loader
from util.meters import TestMeter

import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data.distributed
import torch.multiprocessing as mp
from backbone_times.Spationtemporal import creat_spationtemporal,target_distribution

from torch.utils.data import Subset
import re
import csv

def test_kmeans(model, test_loader, epoch,save_name, args,projection_head=None, Use_GPU=True):

    local_rank = 0

    model.eval()
    all_feats = []

    targets = []
    mask = np.array([])

    # First extract all features
    with torch.no_grad():
        for batch_idx, (images, label, uq_idx,_) in enumerate(tqdm(test_loader)):
            
            images = images.cuda()
            
            backbone_feature,cls_x,x_space,x_time = model(images)
            # all_feats.append(backbone_feature.detach().cpu().numpy())
            all_feats.append(torch.nn.functional.normalize(backbone_feature,dim=-1).detach().cpu().numpy())
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

    print('kmeans Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
    # return all_acc, old_acc, new_acc


def test(backbone,projector,test_loader,test_meter,args,cfg,epoch,local_rank):

    backbone.eval()
    
    preds, targets = [], []
    new_list = []
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

            label = label.cuda(local_rank,non_blocking=True)
            uq_idxs = uq_idxs.cuda(local_rank,non_blocking=True)
            index = index.cuda(local_rank,non_blocking=True)


        with torch.no_grad():
            if isinstance(images, (list,)):
                images = images[0] 
            backbone_feature,cls_x,q,st_feature = backbone(images)
        
            _,logits = projector(backbone_feature)

            mask = torch.tensor(np.array([True if x.item() in args.train_classes else False for x in label]))

            cls_x = cls_x.cpu()

            test_meter.update_stats(
                cls_x, label, mask,index
            )
            
    all_acc, old_acc,new_acc = test_meter.finalize_metrics()
    all_b, old_b,new_b = test_meter.finalize_metrics_balanced()

    args.writer.add_scalars('sim_times',{'Old': old_acc, 'New': new_acc,'All': all_acc}, epoch)
    print('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
    args.writer.add_scalars('sim_times_balanced',{'Old_b': old_b, 'New_b': new_b,'All_b': all_b}, epoch)
    print('balanced Accuracies: All_b {:.4f} | Old_b {:.4f} | New_b {:.4f}'.format(all_b, old_b, new_b))
    
   
def run_tests_from_folder(backbone, projector, checkpoint_dir, test_loader, class_loader,label_meter, args, cfg,local_rank):

    pattern = re.compile(r'b_model_epoch_(\d+).pt')
    checkpoint_files = []
    for filename in os.listdir(checkpoint_dir):
        match = pattern.match(filename)
        if match:
            epoch = int(match.group(1))
            checkpoint_files.append((epoch, os.path.join(checkpoint_dir, filename)))
    checkpoint_files.sort()

    for epoch, checkpoint_path in checkpoint_files:
        backbone, projector, loaded_epoch = load_simgcd_pretrained(backbone, projector, checkpoint_path,multi_gpu=False)
        args.logger.info(f"Loading checkpoint: {checkpoint_path}")
        # print("cls_head result...")
        # test(backbone, projector, test_loader,label_meter, args, cfg, loaded_epoch,local_rank)
        print("Kmeans result...")
        test_kmeans(backbone, test_loader, epoch,"Test all acc", args,projection_head=None, Use_GPU=False)
            
def test_net(local_rank,nprocs, args):

    args.local_rank = local_rank #当前gpu的rank, such as 0, or 1 or 2...
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    cfg = load_config(args)

    args = get_class_splits(args,cfg)
    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args,cfg, runner_name=['simgcd'])
    args.logger.info(f'Using evaluation function {args.eval_funcs[0]} to print results')
    
    args.interpolation = 3
    args.crop_pct = 0.875

    if args.warmup_model_dir is not None:
        args.logger.info(f'Loading weights from {args.warmup_model_dir}')
        backbone.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))
    
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes

   
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
    
    train_dataset, test_dataset, unlabelled_train_examples_test, class_dataset,class_label_dataset,class_merge_dataset,datasets = get_datasets(cfg,args.dataset_name,
                                                                                         train_transform,
                                                                                         test_transform,
                                                                                         args)
                                                            

    test_loader_unlabelled = loader.construct_loader(cfg, "test",unlabelled_train_examples_test)
    class_loader = loader.construct_loader(cfg, "class",class_dataset)

    label_meter = TestMeter(
        len(test_loader_unlabelled.dataset.data)// (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TRAIN.NUM_SPATIAL_CROPS),
        cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TRAIN.NUM_SPATIAL_CROPS,
        cfg.MODEL.NUM_CLASSES,
        len(test_loader_unlabelled),
        cfg.DATA.MULTI_LABEL,
        cfg.DATA.ENSEMBLE_METHOD,
    )

    args.logger.info('Testing on unlabelled examples in the training data...')
    
    run_tests_from_folder(backbone, projector, cfg.OUTPUT_DIR+"log/checkpoints", test_loader_unlabelled,class_loader,label_meter, args, cfg,local_rank)


def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    args.nprocs = torch.cuda.device_count()

    # mp.spawn(test_net,nprocs=args.nprocs,args=(args.nprocs,args))
    test_net(0,args.nprocs, args)


if __name__ == "__main__":
    main()