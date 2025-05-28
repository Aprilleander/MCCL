import os
import pandas as pd
import numpy as np
from copy import deepcopy

from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset

from data.data_utils import subsample_instances
import torch
import torch.nn.functional as F
import random

from . import transform as transform
import cv2
from util.env import pathmgr

from . import utils as utils
from itertools import chain as chain
import torch.utils.data
from fvcore.common.file_io import PathManager
import json
import csv
#from data_utils import ContrastiveLearningViewGenerator
"""
def nlen(self):
    return len(self.data)
"""

class ssv2(Dataset):

    def __init__(self, cfg=None,num_frames=8,train=True, labelled=True, target_transform=None, loader=default_loader, download=True,fraction=1.0):

        # self.root = os.path.expanduser(root)

        self.labelled = labelled
        self.target_transform = target_transform
        # self.base_folder = ssv2_base_frame
        self.loader = loader
        self.train = train
        self.num_frames = num_frames

        self._num_retries = 10
        self.cfg = cfg
        self.root = self.cfg.DATA.SSv2_PATH_TO_DATA_DIR
        self.fraction = fraction
        
        self._video_meta = {}

        if self.train:
            self.mode = 'train'
            self._num_clips = 1
            self.n_views = 1
        else:
            self.mode = 'test'
            
            self._num_clips = (
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )
            
            #self._num_clips = 1
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
        # Loading label names.
        with PathManager.open(
            os.path.join(
                self.cfg.DATA.SSv2_PATH_TO_DATA_DIR,
                "something-something-v2-labels.json",
            ),
            "r",
        ) as f:
            label_dict = json.load(f)

        # Loading labels.
        if self.mode == "train":
            if self.labelled == True:
                label_file_path = "something-something-v2-label_train_v15.json"
            else:
                label_file_path = "something-something-v2-unlabel_train_v15.json"
        else:
            label_file_path = "something-something-v2-label_validation_v15.json"


        label_file = os.path.join(self.cfg.DATA.SSv2_PATH_TO_DATA_DIR,label_file_path)

        with PathManager.open(label_file, "r") as f:
            label_json = json.load(f)

        self._video_names = []
        self._labels = []
        for video in label_json:
            video_name = video["id"]
            template = video["template"]
            template = template.replace("[", "")
            template = template.replace("]", "")
            label = int(label_dict[template])
            self._video_names.append(video_name)
            self._labels.append(label)

        if self.mode == "train":
            if self.labelled == True:
                csv_file_path = "label_train_v15.csv"
            else:
                csv_file_path = "unlabel_train_v15.csv"
        else:
            csv_file_path = "label_validation_v15.csv"

        path_to_file = os.path.join(
            self.cfg.DATA.SSv2_PATH_TO_DATA_DIR, csv_file_path
        )
        
        assert PathManager.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )

        self._path_to_videos, _ = utils.load_image_lists(
            path_to_file, self.cfg.DATA.SSv2_PATH_PREFIX
        )
        
        # From dict to list.
        new_paths, new_labels = [], []
        for index in range(len(self._video_names)):
            if self._video_names[index] in self._path_to_videos:
                new_paths.append(self._path_to_videos[self._video_names[index]])
                new_labels.append(self._labels[index])

        self._labels = new_labels
        self._path_to_videos = new_paths

        # Extend self when self._num_clips > 1 (during testing).
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
                [
                    range(self._num_clips)
                    for _ in range(len(self._path_to_videos))
                ]
            )
        )

        if self.mode == "train":
            if self.labelled == True:
                cub_file_path = "cub_label_train_v15.csv"
            else:
                cub_file_path = "cub_unlabel_train_v15.csv"
        else:
            cub_file_path = "cub_label_validation_v15.csv"

        cub_csv_file = os.path.join(
            self.cfg.DATA.SSv2_PATH_TO_DATA_DIR, cub_file_path)


        # with open(cub_csv_file,'w',newline='',encoding='utf-8') as file:
        #     writer = csv.writer(file)
        #     writer.writerow(['img_id','filepath','target'])
        #     for video_name, path_list,label in zip(self._video_names, self._path_to_videos,self._labels):
        #         path = os.path.join(self.cfg.DATA.SSv2_PATH_TO_DATA_DIR,video_name+'_frame')
        #         writer.writerow([video_name,path,label])
        
        # file.close()
    
        images = pd.read_csv(cub_csv_file)
        self.data = images.to_numpy()
        print(self.data.shape)

    def _apply_fraction(self):
        """
        Applies the fraction of data to the dataset. Only keeps a fraction of the samples.
        """
        total_size = len(self.data)
        subset_size = int(total_size * self.fraction)
        
        # Randomly sample a subset of indices
        sampled_indices = np.random.choice(total_size, subset_size, replace=False)
        
        # Apply the subset sampling to both video names, labels, and paths
        self._video_names = [self._video_names[i] for i in sampled_indices]
        self._labels = [self._labels[i] for i in sampled_indices]
        self._path_to_videos = [self._path_to_videos[i] for i in sampled_indices]
        self.data = [self.data[i] for i in sampled_indices]
        print("applying...",len(self.data))


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

        if self.mode in ["train", "val"]: #or self.cfg.MODEL.ARCH in ['resformer', 'vit']:
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
            if self.cfg.TEST.NUM_SPATIAL_CROPS == 1:
                spatial_sample_index = 1

            min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )

        label = self._labels[index]

        """
        frames =[]
        for i in range(self.n_views):
            frame = self.transform_frame(index)
            frames.append(frame)
        """
        if self.n_views >1:
            frames = [self.transform_frame(index,spatial_sample_index,min_scale,max_scale,crop_size) for i in range(self.n_views)]
        else:
            frames = self.transform_frame(index,spatial_sample_index,min_scale,max_scale,crop_size) 
        
        if frames is None:
            return None


        return frames, label, self.uq_idxs[index],index
    
    """
    def __len__(self):

        return len(self._path_to_videos)
    """

    def __len__(self):
        #return len(self.data)
        return len(self.data)*self._num_clips

    def __nlen__(self):
        return len(self._path_to_videos)


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

        frames = torch.as_tensor(
            utils.retry_load_images(
                [self._path_to_videos[index][frame] for frame in seq],
                self._num_retries,
            )
        )
        if frames is None:
            return None

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

        

def subsample_dataset(dataset, idxs):

    mask = np.zeros(len(dataset)).astype('bool')
    mask[idxs] = True

    dataset.data = dataset.data[mask]
    dataset.uq_idxs = dataset.uq_idxs[mask]

    return dataset


def subsample_classes(dataset, include_classes=range(160)):

    include_classes_cub = np.array(include_classes) + 1     # CUB classes are idxed 1 --> 200 instead of 0 --> 199
    # cls_idxs = [x for x, (_, r) in enumerate(dataset.data.iterrows()) if int(r['target']) in include_classes_cub]
    
    cls_idxs = [x for x, r in enumerate(dataset.data) if int(r[2]) in include_classes_cub]

    # TODO: For now have no target transform
    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset


def get_train_val_indices(train_dataset, val_split=0.2):

    # train_classes = np.unique(train_dataset.data['target'])
    train_classes = np.unique(train_dataset.data[:, 2])

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        # cls_idxs = np.where(train_dataset.data['target'] == cls)[0]
        cls_idxs = np.where(train_dataset.data[:, 2] == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs


def get_ssv2_datasets(cfg,num_frames):

    train_dataset_labelled = ssv2(cfg=cfg,num_frames=num_frames,train=True,labelled=True,download=False)
    train_dataset_unlabelled = ssv2(cfg=cfg,num_frames=num_frames,train=True,labelled=False,download=False)
    class_dataset = ssv2(cfg=cfg,num_frames=num_frames,train=True,labelled=False,download=False,fraction=0.075)
    class_label_dataset = ssv2(cfg=cfg,num_frames=num_frames,train=True,labelled=True,download=False,fraction=0.2)
    test_dataset = ssv2(cfg=cfg,num_frames=num_frames, train=False,labelled=True, download=False)
    val_dataset_labelled = None

    
    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': val_dataset_labelled,
        'test': test_dataset,
        'class': class_dataset,
        'class_label':class_label_dataset,
    }

    return all_datasets
