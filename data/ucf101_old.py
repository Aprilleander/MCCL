import os
import pandas as pd
import numpy as np
from copy import deepcopy

from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset

from data.data_utils import subsample_instances
from config import ucf101_root, ucf101_base_frame
import torch
import torch.nn.functional as F
import random

class ucf101(Dataset):

    def __init__(self, root,num_frames=8, train=True, transform=None, target_transform=None, loader=default_loader, download=True):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.base_folder =ucf101_base_frame
        self.loader = loader
        self.train = train
        self.num_frames = num_frames

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.uq_idxs = np.array(range(len(self)))

    def _load_metadata(self):

        images = pd.read_csv(os.path.join(self.root, 'dccl_labels', 'images.txt'), sep=' ', names=['img_id', 'filepath'])

        image_class_labels = pd.read_csv(os.path.join(self.root, 'dccl_labels', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'dccl_labels', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1].to_numpy()
        else:
            self.data = self.data[self.data.is_training_img == 0].to_numpy()
        for index, row in enumerate(self.data):
            self.data[index][1] = os.path.join(self.root, self.base_folder, row[1])


    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for row in self.data:
            filepath = row[1]
            #if not os.path.isfile(filepath):
            if not os.path.exists(filepath):
                print(filepath)
                return False
        return True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # sample = self.data.iloc[idx]
        sample = self.data[idx]
        if sample is None:
            return self.__getitem__(idx+1)
            
        path = sample[1]
        target = int(sample[2]) - 1  # Targets start at 1 by default, so shift to 0

        data_loader = self.loader
        img_dict = sample_ucf101_frames(path,data_loader,self.transform,self.num_frames)
        # attribute = sample[4:].to_numpy(dtype=np.int32)
        #if self.transform is not None:
            #img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img_dict, target, self.uq_idxs[idx]


def sample_ucf101_frames(images_folder,data_loader,transform,n):
    
    images_files = sorted([f for f in os.listdir(images_folder) if os.path.isfile(os.path.join(images_folder, f))])
    
    seq_len = len(images_files)
    seq = list(range(len(images_files)))

    if seq_len <=n:
        indices = list(range(seq_len))+[seq_len-1]*(n-seq_len)
    else:
        indices = random.sample(seq,n)

    indices = sorted(indices)

    sample_frames = [os.path.join(images_folder, images_files[i]) for i in indices]
    
    #img batch, 3,H,W
    #video batch, 3,T=16,H,W
    img_dict = None
    for img_path in sample_frames:
        #img = Image.open(fpath).convert('RGB')
        img = data_loader(img_path)
        if transform is not None:
            img = transform(img)
            if len(img)==2:
                img[0]=img[0].unsqueeze(1)
                img[1]=img[1].unsqueeze(1)
                if img_dict is None:
                    img_dict = img
                else:
                    img_dict[0] =torch.cat((img_dict[0],img[0]),dim=1)
                    img_dict[1] =torch.cat((img_dict[1],img[1]),dim=1)
            else:
                img = img.unsqueeze(1)
                if img_dict is None:
                    img_dict = img
                else:
                    img_dict =torch.cat((img_dict,img),dim=1)
    return img_dict


def subsample_dataset(dataset, idxs):

    mask = np.zeros(len(dataset)).astype('bool')
    mask[idxs] = True

    dataset.data = dataset.data[mask]
    dataset.uq_idxs = dataset.uq_idxs[mask]

    return dataset


def subsample_classes(dataset, include_classes=range(160)):

    include_classes_cub = np.array(include_classes) + 1     # CUB classes are indexed 1 --> 200 instead of 0 --> 199
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


def get_ucf101_datasets(num_frames,train_transform, test_transform, train_classes=range(160), prop_train_labels=0.8,
                    split_train_val=False, seed=0):

    np.random.seed(seed)

    # Init entire training set
    whole_training_set = ucf101(root=ucf101_root, num_frames=num_frames,transform=train_transform, train=True, download=False)

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    # Split into training and validation sets
    train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
    train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
    val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
    val_dataset_labelled_split.transform = test_transform

    # Get unlabelled data
    unlabelled_indices = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), np.array(list(unlabelled_indices)))

    # Get test set for all classes
    test_dataset = ucf101(root=ucf101_root, num_frames=num_frames,transform=test_transform, train=False, download=False)

    # Either split train into train and val or use test set as val
    train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
    val_dataset_labelled = val_dataset_labelled_split if split_train_val else None

    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': val_dataset_labelled,
        'test': test_dataset,
    }

    return all_datasets

def main():

    x = get_ucf101_datasets(None, None, split_train_val=False,
                         train_classes=range(16), prop_train_labels=0.5)

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labelled and unlabelled overlap...')
    print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))

    print(f'Num Labelled Classes: {len(set(x["train_labelled"].data["target"].values))}')
    print(f'Num Unabelled Classes: {len(set(x["train_unlabelled"].data["target"].values))}')
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')

if __name__ == '__main__':
    main()