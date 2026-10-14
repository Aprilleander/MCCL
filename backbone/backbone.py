import torch.nn as nn
"""
from slowfast.models import build_model
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.parser import load_config, parse_args
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
"""
import argparse

from model import vision_transformer as vits
from model import attribute_transformer as ats
from config import exp_root, km_label_path, subset_len_path, dino_base_pretrain_path, dino_small_pretrain_path, cub_root,mvit_pretrain_path
import torch
from backbone.mvit import DINOHead

from backbone.mvit import MViT
from backbone.MViT_defaults import get_cfg
from model import checkpoint as cu


"""
def Multi_video_train():
    
    args = parse_args()
    parser = argparse.ArgumentParser(
        description='dccl_ssts',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cfg', default='/mnt/newdisk/mnt/newdisk/code/DGC/DCCL-main/configs/Kinetics/t0_0.4_s4_0.6.yaml', type=str)
    cfg = load_config(args)
    model = build_model(cfg)

    # Load a checkpoint to resume training if applicable.
    cu.load_checkpoint(cfg.TRAIN.CHECKPOINT_FILE_PATH, model,data_parallel=False)
    
    return model
"""
    

def get_backbone(args,logger,labelled_train_examples):

    pretrain_path = args.pretrain_model
    num_attribute = 28
    device =args.device

    if args.base_model == 'base':
        if args.model_name == 'at':
            student = ats.__dict__['at_base'](pretrain_path)
        elif args.model_name == 'at2':
            student = ats.__dict__['at2_base'](pretrain_path, num_attribute=num_attribute,
                                               feat_channal=args.feat_channal, grad_from_block=args.grad_from_block)
        elif args.model_name == 'at3':
            student = ats.__dict__['at3_base'](pretrain_path, num_attribute=num_attribute,
                                               feat_channal=args.feat_channal, grad_from_block=args.grad_from_block)
        elif args.model_name == 'at4':
            student = ats.__dict__['at4_base'](pretrain_path, num_attribute=num_attribute,
                                               feat_channal=args.feat_channal, grad_from_block=args.grad_from_block)
        elif args.model_name == 'at5':
            student = ats.__dict__['at5_base'](pretrain_path, num_attribute=num_attribute,
                                               feat_channal=args.feat_channal, grad_from_block=args.grad_from_block)
        elif args.model_name == 'at6':
            student = ats.__dict__['at6_base'](pretrain_path, num_attribute=num_attribute,
                                               attribute_feat_channal=args.attribute_feat_channal,
                                               grad_from_block=args.grad_from_block)
        elif args.model_name == 'at7':
            student = ats.__dict__['at7_base'](pretrain_path, num_attribute=num_attribute,
                                               attribute_feat_channal=args.attribute_feat_channal,
                                               grad_from_block=args.grad_from_block)
        elif args.model_name == 'at8':
            student = ats.__dict__['at8_base'](pretrain_path, num_attribute=num_attribute,
                                               attribute_feat_channal=args.attribute_feat_channal,
                                               grad_from_block=args.grad_from_block)
        elif args.model_name == 'at9':
            student = ats.__dict__['at9_base'](pretrain_path, num_attribute=num_attribute,
                                               attribute_feat_channal=args.attribute_feat_channal,
                                               grad_from_block=args.grad_from_block)
        elif args.model_name == 'at10':
            student = ats.__dict__['at10_base'](pretrain_path, num_attribute=num_attribute,
                                                attribute_feat_channal=args.attribute_feat_channal,
                                                grad_from_block=args.grad_from_block)
            if args.use_meta_attribute:
                meta_learner = ats.__dict__['meta1_base'](pretrain_path, labelled_train_examples.dict_attribute,
                                                          grad_from_block=args.grad_from_block)
                logger(meta_learner)
                meta_learner.to(device)
            else:
                meta_learner = None
        elif args.model_name == 'at12':
            student = ats.__dict__['at12_base'](pretrain_path, num_attribute=num_attribute,
                                                attribute_feat_channal=args.attribute_feat_channal,
                                                grad_from_block=args.grad_from_block)
            if args.use_meta_attribute:
                meta_learner = ats.__dict__['meta2_base'](pretrain_path, labelled_train_examples.dict_attribute,
                                                          grad_from_block=args.grad_from_block)
                logger(meta_learner)
                meta_learner.to(device)
            else:
                meta_learner = None
        elif args.model_name == 'at13':
            student = ats.__dict__['at13_base'](pretrain_path, num_attribute=num_attribute,
                                                attribute_feat_channal=args.attribute_feat_channal,
                                                grad_from_block=args.grad_from_block)
            if args.use_meta_attribute:
                meta_learner = ats.__dict__['meta2_base'](pretrain_path, labelled_train_examples.dict_attribute,
                                                          grad_from_block=args.grad_from_block)
                logger(meta_learner)
                meta_learner.to(device)
            else:
                meta_learner = None

        elif args.model_name == 'at13_3D_video':

            MViT_cfg = get_cfg()
            student = MViT(MViT_cfg)
            
            if pretrain_path is not None:
                student.load_state_dict(torch.load(args.weight_path, map_location='cpu'))
            meta_learner = None

            
        elif args.model_name == 'at14':
            student = ats.__dict__['at14_base'](pretrain_path, num_attribute=num_attribute,
                                                attribute_feat_channal=args.attribute_feat_channal,
                                                grad_from_block=args.grad_from_block, device=device)
            if args.use_meta_attribute:
                meta_learner = ats.__dict__['meta2_base'](pretrain_path, labelled_train_examples.dict_attribute,
                                                          grad_from_block=args.grad_from_block)
                logger(meta_learner)
                meta_learner.to(device)
            else:
                meta_learner = None

        else:
            raise NotImplementedError
        # student = vits.__dict__['vit_base']()
        # student = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')

    elif args.base_model == 'small':
        if args.model_name == 'at':
            student = ats.__dict__['at_small'](pretrain_path)
        elif args.model_name == 'at2':
            student = ats.__dict__['at2_small'](pretrain_path, num_attribute=num_attribute,
                                                feat_channal=args.feat_channal, grad_from_block=args.grad_from_block)
        elif args.model_name == 'at3':
            student = ats.__dict__['at3_small'](pretrain_path, num_attribute=num_attribute,
                                                feat_channal=args.feat_channal, grad_from_block=args.grad_from_block)
        elif args.model_name == 'at4':
            student = ats.__dict__['at4_small'](pretrain_path, num_attribute=num_attribute,
                                                feat_channal=args.feat_channal, grad_from_block=args.grad_from_block)
        elif args.model_name == 'at5':
            student = ats.__dict__['at5_small'](pretrain_path, num_attribute=num_attribute,
                                                feat_channal=args.feat_channal, grad_from_block=args.grad_from_block)
        elif args.model_name == 'at6':
            student = ats.__dict__['at6_small'](pretrain_path, num_attribute=num_attribute,
                                                attribute_feat_channal=args.attribute_feat_channal,
                                                grad_from_block=args.grad_from_block)
        elif args.model_name == 'at7':
            student = ats.__dict__['at7_small'](pretrain_path, num_attribute=num_attribute,
                                                attribute_feat_channal=args.attribute_feat_channal,
                                                grad_from_block=args.grad_from_block)
        elif args.model_name == 'at8':
            student = ats.__dict__['at8_small'](pretrain_path, num_attribute=num_attribute,
                                                attribute_feat_channal=args.attribute_feat_channal,
                                                grad_from_block=args.grad_from_block)
        elif args.model_name == 'at9':
            student = ats.__dict__['at9_small'](pretrain_path, num_attribute=num_attribute,
                                                attribute_feat_channal=args.attribute_feat_channal,
                                                grad_from_block=args.grad_from_block)
        elif args.model_name == 'at10':
            student = ats.__dict__['at10_small'](pretrain_path, num_attribute=num_attribute,
                                                 attribute_feat_channal=args.attribute_feat_channal,
                                                 grad_from_block=args.grad_from_block)
            if args.use_meta_attribute:
                meta_learner = ats.__dict__['meta1_small'](pretrain_path, labelled_train_examples.dict_attribute,
                                                           grad_from_block=args.grad_from_block)
                logger(meta_learner)
                meta_learner.to(device)
            else:
                meta_learner = None
        elif args.model_name == 'at11':
            student = ats.__dict__['at11_small'](pretrain_path, num_attribute=num_attribute,
                                                 attribute_feat_channal=args.attribute_feat_channal,
                                                 grad_from_block=args.grad_from_block)
            if args.use_meta_attribute:
                meta_learner = ats.__dict__['meta2_small'](pretrain_path, labelled_train_examples.dict_attribute,
                                                           grad_from_block=args.grad_from_block)
                logger(meta_learner)
                meta_learner.to(device)
            else:
                meta_learner = None
        elif args.model_name == 'at12':
            student = ats.__dict__['at12_small'](pretrain_path, num_attribute=num_attribute,
                                                 attribute_feat_channal=args.attribute_feat_channal,
                                                 grad_from_block=args.grad_from_block)
            if args.use_meta_attribute:
                meta_learner = ats.__dict__['meta2_small'](pretrain_path, labelled_train_examples.dict_attribute,
                                                           grad_from_block=args.grad_from_block)
                logger(meta_learner)
                meta_learner.to(device)
            else:
                meta_learner = None
        elif args.model_name == 'at13':
            student = ats.__dict__['at13_small'](pretrain_path, num_attribute=num_attribute,
                                                 attribute_feat_channal=args.attribute_feat_channal,
                                                 grad_from_block=args.grad_from_block)
            if args.use_meta_attribute:
                meta_learner = ats.__dict__['meta2_small'](pretrain_path, labelled_train_examples.dict_attribute,
                                                           grad_from_block=args.grad_from_block)
                logger(meta_learner)
                meta_learner.to(device)
            else:
                meta_learner = None

        elif args.model_name == 'at14':
            student = ats.__dict__['at14_small'](pretrain_path, num_attribute=num_attribute,
                                                 attribute_feat_channal=args.attribute_feat_channal,
                                                 grad_from_block=args.grad_from_block, device=device)
            if args.use_meta_attribute:
                meta_learner = ats.__dict__['meta2_small'](pretrain_path, labelled_train_examples.dict_attribute,
                                                           grad_from_block=args.grad_from_block)
                logger(meta_learner)
        student.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))
    return student,meta_learner



def construct_backbone(args):

    if args.base_model == 'base':

        args.interpolation = 3
        args.crop_pct = 0.875
        if args.pretrain_model == 'dino':
            args.pretrain_path = dino_base_pretrain_path
        elif args.pretrain_model == 'mvit':
            #args.pretrain_path = mvit_pretrain_path
            args.pretrain_path = None
        else:
            raise NotImplementedError

        # NOTE: Hardcoded image size as we do not finetune the entire ViT model
        args.image_size = 224
        args.feat_dim = 768
        args.num_mlp_layers = 3
        args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes
        args.feat_channal = 197  # (13+1)*(13+1) + 1 class token
        args.attribute_feat_channal = 16  # (13+1)*(13+1) + 1 class token

    elif args.base_model == 'small':
        args.interpolation = 3
        args.crop_pct = 0.875
        if args.pretrain_model == 'dino':
            args.pretrain_path = dino_small_pretrain_path
        else:
            raise NotImplementedError

        # NOTE: Hardcoded image size as we do not finetune the entire ViT model
        args.image_size = 224
        args.feat_dim = 384
        args.num_mlp_layers = 3
        args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes
        args.feat_channal = 197  # (13+1)*(13+1) + 1 class token
        args.attribute_feat_channal = 16  # (13+1)*(13+1) + 1 class token
    else:
        raise NotImplementedError

    return args

def generate_projector(args):
    projectors = nn.ModuleDict()
    if args.use_expert:
        for expert_id in range(args.experts_num):
            projectors[f'expert_{expert_id + 1}'] = vits.__dict__['DINOHead'](in_dim=args.feat_dim,
                                                                              out_dim=args.mlp_out_dim,
                                                                              nlayers=args.num_mlp_layers).to(args.device)
    if args.use_global_con:
        projectors['global_contrastive'] = vits.__dict__['DINOHead'](in_dim=args.feat_dim,
                                                                     out_dim=args.mlp_out_dim,
                                                                     nlayers=args.num_mlp_layers).to(args.device)

    if args.use_attribute:
        projectors['attributes'] = vits.__dict__['Attribute_Classifier8ind'](labelled_train_examples.dict_attribute,
                                                                             in_dim=args.feat_dim,
                                                                             ).to(args.device)
    if args.use_cluster_head:
        projectors['cluster'] = vits.__dict__['DINOHead'](in_dim=args.feat_dim,
                                                          out_dim=args.feat_dim,
                                                          nlayers=args.num_mlp_layers).to(args.device)
    return projectors

