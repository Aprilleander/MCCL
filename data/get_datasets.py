from data.data_utils import MergedDataset

from data.cifar import get_cifar_10_datasets, get_cifar_100_datasets
from data.herbarium_19 import get_herbarium_datasets
from data.stanford_cars import get_scars_datasets
from data.imagenet import get_imagenet_100_datasets, get_imagenet_1k_datasets
from data.cub import get_cub_datasets
from data.fgvc_aircraft import get_aircraft_datasets
from data.ssv2 import get_ssv2_datasets
from data.ucf101 import get_ucf101_datasets
from data.charades import get_Charades_datasets
from data.kinetics400 import get_kinetics400_datasets
from data.vb100 import get_vb100_datasets
from data.ibc127 import get_ibc127_datasets


from copy import deepcopy
import pickle
import os

from config import osr_split_dir


get_dataset_funcs = {
    'cifar10': get_cifar_10_datasets,
    'cifar100': get_cifar_100_datasets,
    'imagenet_100': get_imagenet_100_datasets,
    'imagenet_1k': get_imagenet_1k_datasets,
    'herbarium_19': get_herbarium_datasets,
    'cub': get_cub_datasets,
    'aircraft': get_aircraft_datasets,
    'scars': get_scars_datasets,
    'ssv2': get_ssv2_datasets,
    'ucf101': get_ucf101_datasets,
    'charades': get_Charades_datasets,
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
    datasets = get_dataset_f(cfg=cfg,num_frames=args.num_frames,train_transform=train_transform, test_transform=test_transform,
                            train_classes=args.train_classes,
                            prop_train_labels=args.prop_train_labels,
                            split_train_val=False)
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
    if args.dataset_name in ('scars', 'cub', 'aircraft'):
        if hasattr(args, 'use_ssb_splits'):
            use_ssb_splits = args.use_ssb_splits
        else:
            use_ssb_splits = False

    # -------------
    # GET CLASS SPLITS
    # -------------
    if args.dataset_name == 'cifar10':

        args.image_size = 32
        args.train_classes = range(5)
        args.unlabeled_classes = range(5, 10)

    elif args.dataset_name == 'cifar100':

        args.image_size = 32
        args.train_classes = range(80)
        args.unlabeled_classes = range(80, 100)

    elif args.dataset_name == 'herbarium_19':

        args.image_size = 224
        herb_path_splits = os.path.join(osr_split_dir, 'herbarium_19_class_splits.pkl')

        with open(herb_path_splits, 'rb') as handle:
            class_splits = pickle.load(handle)

        args.train_classes = class_splits['Old']
        args.unlabeled_classes = class_splits['New']

    elif args.dataset_name == 'imagenet_100':

        args.image_size = 224
        args.train_classes = range(50)
        args.unlabeled_classes = range(50, 100)

    elif args.dataset_name == 'ssv2':
        args.image_size = 224
        
        odds,evens = generate_and_separate(174)
        args.train_classes = evens
        args.unlabeled_classes = odds

        
    elif args.dataset_name == 'ucf101':
        args.image_size = 224
        odds,evens = generate_and_separate(101)
        args.train_classes = evens
        args.unlabeled_classes = odds

    elif args.dataset_name == 'vb100':
        args.image_size = 224
        args.train_classes = range(51)
        args.unlabeled_classes = range(51, 101)
    
    elif args.dataset_name == 'ibc127':
        args.image_size = 224

        odds,evens = generate_and_separate(127)
        args.train_classes = evens
        args.unlabeled_classes = odds


    elif args.dataset_name == 'kinetics400':
        args.image_size = 224
        
        odds,evens = generate_and_separate(400)
        args.train_classes = evens
        args.unlabeled_classes = odds




    elif args.dataset_name == 'imagenet_1k':

        args.image_size = 224
        args.train_classes = range(500)
        args.unlabeled_classes = range(500, 1000)
    
    elif args.dataset_name == 'scars':

        args.image_size = 224

        if use_ssb_splits:

            split_path = os.path.join(osr_split_dir, 'scars_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            args.unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

        else:

            args.train_classes = range(98)
            args.unlabeled_classes = range(98, 196)

    elif args.dataset_name == 'aircraft':

        args.image_size = 224
        if use_ssb_splits:

            split_path = os.path.join(osr_split_dir, 'aircraft_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            args.unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

        else:
            args.train_classes = range(50)
            args.unlabeled_classes = range(50, 100)

    elif args.dataset_name == 'cub':

        args.image_size = 224

        if use_ssb_splits:

            split_path = os.path.join(osr_split_dir, 'cub_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            args.unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

        else:

            args.train_classes = range(100)
            args.unlabeled_classes = range(100, 200)

    else:

        raise NotImplementedError

    return args
