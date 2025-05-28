# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Argument parser functions."""

import argparse
import sys

import backbone.checkpoint as cu
from backbone.Timesformer_defaults import load_config,get_timesformer_cfg


def parse_args():
    """
    Parse the following arguments for a default parser for PySlowFast users.
    Args:
        shard_id (int): shard id for the current machine. Starts from 0 to
            num_shards - 1. If single machine is used, then set shard id to 0.
        num_shards (int): number of shards using by the job.
        init_method (str): initialization method to launch the job with multiple
            devices. Options includes TCP or shared file-system for
            initialization. details can be find in
            https://pytorch.org/docs/stable/distributed.html#tcp-initialization
        cfg (str): path to the config file.
        opts (argument): provide addtional options from the command line, it
            overwrites the config loaded from file.
    """
    parser = argparse.ArgumentParser(
        description="Provide SlowFast video training and testing pipeline."
    )
    parser.add_argument(
        "--shard_id",
        help="The shard id of current node, Starts from 0 to num_shards - 1",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--num_shards",
        help="Number of shards using by the job",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--init_method",
        help="Initialization method, includes TCP or shared file-system",
        default="tcp://localhost:9999",
        type=str,
    )

    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--num_frames', default=8, type=int)
    # parser.add_argument('--cfg_file', default='/mnt/newdisk/mnt/newdisk/code/fusion/nofor/Simc_init_ucf101/configs/ucf101/TimeSformer_ucf_8_224_even.yaml', type=str)
    parser.add_argument('--cfg_file', default='/mnt/code/final/MCCL/configs/ucf101/chi_ucf101_8_224.yaml', type=str)
    parser.add_argument('--dataset_name', type=str, default='ucf101', help='options: ucf101,ssv2,kinetcis400,vb100,ibc127')
    parser.add_argument('--opts', default=None, type=str)

    parser.add_argument('--unbalanced', default=False, type=bool)
    parser.add_argument('--gpu_id', default=None, type=str)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--temperature', type=float, default=1.0)
    
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--load_legancy', default=True, type=bool)
    
    parser.add_argument('--eval_funcs', help='Which eval functions to use', default=['v2'])
    parser.add_argument('--test_before_train',default=True, type=bool)

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--head_lr', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--p_weight_decay', type=float, default=1e-3)
    parser.add_argument('--epochs', default=15, type=int)

    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--sup_weight', type=float, default=0.45) #0.35
    parser.add_argument('--sharp_weight', type=float, default=0.5)
    
    
    parser.add_argument('--n_views', default=2, type=int)
    parser.add_argument('--use_contrastive_cluster', default=True, type=bool)
    parser.add_argument('--contrastive_cluster_weight', type=float, default=0.4)
    parser.add_argument('--contrastive_cluster_epochs', type=int, default=15)
    parser.add_argument('--temp', type=float, default=0.05,help="temperature for scaling contrastive loss")

    parser.add_argument('--use-hard', default=False)
    #vote
    parser.add_argument('--use_vote', type=bool, default=True)
    parser.add_argument('--new_vote', type=bool, default=False)
    
    parser.add_argument('--max_capacity', type=int, default=1600)
    parser.add_argument('--memory_momentum', type=float, default=0.1,
                        help="update momentum for the hybrid memory")
    # memory
    parser.add_argument('--use_cluster_head', type=bool, default=False,
                        help="learning rate")
    parser.add_argument('--memax_weight', type=float, default=2)
    
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float, help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float, help='Final value (after linear warmup)of the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=15, type=int, help='Number of warmup epochs for the teacher temperature.')

    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--print_freq', default=60, type=int)
    parser.add_argument('--exp_name', default='simc_dec', type=str)


    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def load_config(args):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_timesformer_cfg()
    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    # Inherit parameters from args.
    if hasattr(args, "num_shards") and hasattr(args, "shard_id"):
        cfg.NUM_SHARDS = args.num_shards
        cfg.SHARD_ID = args.shard_id
    if hasattr(args, "rng_seed"):
        cfg.RNG_SEED = args.rng_seed
    if hasattr(args, "output_dir"):
        cfg.OUTPUT_DIR = args.output_dir

    # Create the checkpoint dir.
    cu.make_checkpoint_dir(cfg.OUTPUT_DIR)
    return cfg
