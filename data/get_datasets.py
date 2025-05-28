from data.data_utils import MergedDataset

from data.ssv2 import get_ssv2_datasets
from data.ucf101 import get_ucf101_datasets
from data.kinetics400 import get_kinetics400_datasets
from data.vb100 import get_vb100_datasets
from data.ibc127 import get_ibc127_datasets


from copy import deepcopy
import pickle
import os

get_dataset_funcs = {
    'ssv2': get_ssv2_datasets,
    'ucf101': get_ucf101_datasets,
    'kinetics400': get_kinetics400_datasets,
    'vb100': get_vb100_datasets,
    'ibc127': get_ibc127_datasets
}


def get_datasets(cfg,dataset_name, train_transform, test_transform, args):

    """
    :return: train_dataset: MergedDataset which concatenates labelled and unlabelled
             test_dataset,
             unlabelled_train_examples_test,
             datasets
    """

    #
    if dataset_name not in get_dataset_funcs.keys():
        raise ValueError

    # Get datasets
    get_dataset_f = get_dataset_funcs[dataset_name]
    datasets = get_dataset_f(cfg=cfg,num_frames=args.num_frames)
    # Set target transforms:
    target_transform_dict = {}
    for i, cls in enumerate(list(args.train_classes) + list(args.unlabeled_classes)):
        target_transform_dict[cls] = i
    target_transform = lambda x: target_transform_dict[x]

    for dataset_name, dataset in datasets.items():
        if dataset is not None:
            dataset.target_transform = target_transform

    # Train split (labelled and unlabelled classes) for training
    train_dataset = MergedDataset(labelled_dataset=deepcopy(datasets['train_labelled']),
                                  unlabelled_dataset=deepcopy(datasets['train_unlabelled']))

    test_dataset = datasets['test']
    unlabelled_train_examples_test = deepcopy(datasets['train_unlabelled'])
    unlabelled_train_examples_test.transform = test_transform

    #test_dataset = deepcopy(datasets['test'])
    class_unlabel_dataset = datasets['class']
    class_label_dataset = datasets['class_label']
    
    class_merge_dataset = MergedDataset(labelled_dataset=deepcopy(datasets['class_label']),
                                  unlabelled_dataset=deepcopy(datasets['class']))

    return train_dataset, test_dataset, unlabelled_train_examples_test,class_unlabel_dataset,class_label_dataset,class_merge_dataset, datasets

def generate_and_separate(n):

    lst = list(range(0, n))
    odds = [num for num in lst if num % 2 != 0]
    evens = [num for num in lst if num % 2 == 0]
    
    return odds, evens

def get_class_splits(args,cfg):

    # For FGVC datasets, optionally return bespoke splits
    use_ssb_splits = False

    # -------------
    # GET CLASS SPLITS
    # -------------
    if args.dataset_name == 'ssv2':

        args.image_size = 224
        odds,evens = generate_and_separate(174)
        args.train_classes = evens
        args.unlabeled_classes = odds
        args.pse_train_classes = range(87)
        args.pse_unlabeled_classes = range(87,174)
    
    elif args.dataset_name == 'ucf101':
        args.image_size = 224
        odds,evens = generate_and_separate(101)
        args.train_classes = evens
        args.unlabeled_classes = odds
        args.pse_train_classes = range(51)
        args.pse_unlabeled_classes = range(51, 101)

    elif args.dataset_name == 'vb100':
        args.image_size = 224
        args.train_classes = range(50)
        args.unlabeled_classes = range(50, 100)
        args.pse_train_classes = range(50)
        args.pse_train_classes = range(50)
    
    elif args.dataset_name == 'ibc127':
        args.image_size = 224

        odds,evens = generate_and_separate(127)
        args.train_classes = evens
        args.unlabeled_classes = odds

        args.pse_train_classes = range(64)
        args.pse_unlabeled_classes = range(64,127)


    elif args.dataset_name == 'kinetics400':
        args.image_size = 224
        odds,evens = generate_and_separate(400)
        args.train_classes = evens
        args.unlabeled_classes = odds
        args.pse_train_classes = range(200)
        args.pse_unlabeled_classes = range(200,400)

    else:
        raise NotImplementedError

    return args
