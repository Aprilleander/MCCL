#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os

import pandas as pd

import random
from itertools import chain as chain
import torch
import numpy as np
import torch.utils.data

from . import utils as utils
from fvcore.common.file_io import PathManager
from torchvision.datasets.folder import default_loader

import json
import csv

class kinetics400(torch.utils.data.Dataset):
    """
    Charades video loader. Construct the Charades video loader, then sample
    clips from the videos. For training and validation, a single clip is randomly
    sampled from every video with random cropping, scaling, and flipping. For
    testing, multiple clips are uniformaly sampled from every video with uniform
    cropping. For uniform cropping, we take the left, center, and right crop if
    the width is larger than height, or take top, center, and bottom crop if the
    height is larger than the width.
    """

    def __init__(self,cfg=None,num_frames=8,train=True, labelled=True, target_transform=None, loader=default_loader, download=True,fraction=1.0):
        """
        Load Charades data (frame paths, labels, etc. ) to a given Dataset object.
        The dataset could be downloaded from Chrades official website
        (https://allenai.org/plato/charades/).
        Please see datasets/DATASET.md for more information about the data format.
        Args:
            dataset (Dataset): a Dataset object to load Charades data to.
            mode (string): 'train', 'val', or 'test'.
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        self.labelled = labelled
        self.target_transform = target_transform
        self.loader = loader
        self.train = train
        self.num_frames = num_frames
        self.cfg = cfg
        self.root = self.cfg.DATA.K400_PATH_TO_DATA_DIR
        self._num_retries = 10
        self.fraction = fraction
      
        if self.train:
            self.mode = 'train'
            self._num_clips = 1
            self.n_views = 1
        
        else:
            self.mode = 'test'
            
            self._num_clips = (
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )
            self.n_views = 1
        
        self._path_to_videos = []
        self._construct_loader()
        if self.fraction < 1.0:
            self._apply_fraction()
        self.uq_idxs = np.array(range(len(self)))

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        if self.mode == "train":
            if self.labelled == True:
                label_file_path = "label_train.json"
            else:
                label_file_path = "unlabel.json"
        else:
            label_file_path = "label_validation.json"


        label_file = os.path.join(self.cfg.DATA.K400_PATH_TO_DATA_DIR,label_file_path)
        
        with PathManager.open(label_file, "r") as f:
            label_json = json.load(f)

        self._video_names = []
        self._labels = []
        for video in label_json:
            video_name = video["id"]
            label = int(video["label"])
            self._video_names.append(video_name)
            self._labels.append(label)

        if self.mode == "train":
            if self.labelled == True:
                csv_file_path = "label_train.csv"
            else:
                csv_file_path = "unlabel.csv"
        else:
            csv_file_path = "label_validation.csv"
        
        path_to_file = os.path.join(
            self.cfg.DATA.K400_PATH_TO_DATA_DIR, csv_file_path
        )

        assert PathManager.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )

        (self._path_to_videos, _) = utils.load_image_lists(
            path_to_file, self.cfg.DATA.K400_PATH_PREFIX, return_list=True
        )

        # if self.mode != "train":
        #     # Form video-level labels from frame level annotations.
        #     self._labels = utils.convert_to_video_level_labels(self._labels)

        self._path_to_videos = list(
            chain.from_iterable(
                [[x] * self._num_clips for x in self._path_to_videos]
            )
        )
        self._labels = list(
            chain.from_iterable([[x] * self._num_clips for x in self._labels])
        )
        self._spatial_temporal_idx = list(
            chain.from_iterable(
                [range(self._num_clips) for _ in range(len(self._labels))]
            )
        )

        if self.mode == "train":
            if self.labelled == True:
                cub_file_path = "cub_label_train.csv"
            else:
                cub_file_path = "cub_unlabel_train.csv"
        else:
            cub_file_path = "cub_label_validation.csv"

        cub_csv_file = os.path.join(
            self.cfg.DATA.K400_PATH_TO_DATA_DIR, cub_file_path)

        # with open(cub_csv_file,'w',newline='',encoding='utf-8') as file:
        #     writer = csv.writer(file)
        #     writer.writerow(['img_id','filepath','target'])
        #     for video_name, path_list,label in zip(self._video_names, self._path_to_videos,self._labels):
        #         path = os.path.join(self.root,video_name)
        #         writer.writerow([video_name,path,label])
        
        # file.close()
    
        images = pd.read_csv(cub_csv_file)
        self.data = images.to_numpy()
        print(self.data.shape)
    
    def _apply_fraction(self):
        """
        Applies the fraction of data to the dataset. Only keeps a fraction of the samples.
        """
        total_size = len(self._path_to_videos)-1
        subset_size = int(total_size * self.fraction)
        
        # Randomly sample a subset of indices
        sampled_indices = np.random.choice(total_size, subset_size, replace=False)
        
        # Apply the subset sampling to both video names, labels, and paths
        self._video_names = [self._video_names[i] for i in sampled_indices]
        self._labels = [self._labels[i] for i in sampled_indices]
        self._path_to_videos = [self._path_to_videos[i] for i in sampled_indices]
        self.data = [self.data[i] for i in sampled_indices]

      
    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video frames can be fetched.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): the index of the video.
        """
        short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        if isinstance(index, tuple):
            index, short_cycle_idx = index

        if self.mode in ["train", "val"]:
            # -1 indicates random sampling.
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            if short_cycle_idx in [0, 1]:
                crop_size = int(
                    round(
                        self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]
                        * self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
            if self.cfg.MULTIGRID.DEFAULT_S > 0:
                # Decreasing the scale is equivalent to using a larger "span"
                # in a sampling grid.
                min_scale = int(
                    round(
                        float(min_scale)
                        * crop_size
                        / self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
        elif self.mode in ["test"]:
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                self._spatial_temporal_idx[index]
                % self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )
        
        label = self._labels[index]

        if self.n_views >1:
            frames = [self.transform_frame(index,spatial_sample_index,min_scale,max_scale,crop_size) for i in range(self.n_views)]
        else:
            frames = self.transform_frame(index,spatial_sample_index,min_scale,max_scale,crop_size)

        if frames is None:
            return None 

        return frames, label, self.uq_idxs[index],index        

    def __len__(self):
        #return len(self.data)
        return len(self.data)*self._num_clips

    def transform_frame(self,index,spatial_sample_index,min_scale,max_scale,crop_size):

        num_frames = self.cfg.DATA.NUM_FRAMES
        video_length = len(self._path_to_videos[index])


        seg_size = float(video_length - 1) / num_frames
        seq = []
        for i in range(num_frames):
            start = int(np.round(seg_size * i))
            end = int(np.round(seg_size * (i + 1)))
            if self.mode == "train":
                seq.append(random.randint(start, end))
            else:
                seq.append((start + end) // 2)

        frames = utils.retry_load_images(
                [self._path_to_videos[index][frame] for frame in seq],
                self._num_retries,)
        if frames is None:
            return None 

        frames = torch.as_tensor(frames)

        # Perform color normalization.
        frames = utils.tensor_normalize(
            frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
        )

        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)
        frames = utils.spatial_sampling(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
            random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
            inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
        )
        #if not self.cfg.RESFORMER.ACTIVE:
        if not self.cfg.MODEL.ARCH in ['vit']:
            frames = utils.pack_pathway_output(self.cfg, frames)
        else:
            # Perform temporal sampling from the fast pathway.
            frames = torch.index_select(
                 frames,
                 1,
                 torch.linspace(
                     0, frames.shape[1] - 1, self.cfg.DATA.NUM_FRAMES

                 ).long(),
            )
        
        return frames


def get_kinetics400_datasets(cfg,num_frames):
    
    train_dataset_labelled = kinetics400(cfg=cfg,num_frames=num_frames,train=True,labelled=True,download=False)
    train_dataset_unlabelled = kinetics400(cfg=cfg,num_frames=num_frames,train=True,labelled=False,download=False)
    class_dataset = kinetics400(cfg=cfg,num_frames=num_frames,train=True,labelled=False,download=False,fraction=0.03)
    class_label_dataset = kinetics400(cfg=cfg,num_frames=num_frames,train=True,labelled=True,download=False,fraction=0.03)

    test_dataset = kinetics400(cfg=cfg,num_frames=num_frames, train=False,labelled=True, download=False)
    val_dataset_labelled = None

    
    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': val_dataset_labelled,
        'test': test_dataset,
        'class':class_dataset,
        'class_label':class_label_dataset,
    }
    
    return all_datasets

