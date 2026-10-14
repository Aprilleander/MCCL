# -----------------
# DATASET ROOTS
# -----------------
cifar_10_root = '${DATASET_DIR}/cifar10'
cifar_100_root = '${DATASET_DIR}/cifar100'
#cub_root = '${DATASET_DIR}/cub'
cub_root = '/mnt/newdisk/mnt/newdisk/dataset/DGC/DCCL/CUB/'
aircraft_root = '${DATASET_DIR}/fgvc-aircraft-2013b'
car_root = '${DATASET_DIR}/cars'
herbarium_dataroot = '${DATASET_DIR}/herbarium_19'
imagenet_root = '${DATASET_DIR}/ImageNet'

#video path
#ssv2_root = '/mnt/newdisk/mnt/newdisk/dataset/DGC/DCCL/ssv2/'
#ssv2_base_frame = 'ssv2_frame'

ssv2_root = '/mnt/newdisk/mnt/newdisk/dataset/action/ssv2_all/'
ssv2_base_frame = 'ssv2_frame_v5/'

ucf101_root = '/root/autodl-tmp/dataset/ucf101/UCF101_rgb/'
ucf101_base_frame = '/root/autodl-tmp/dataset/ucf101/label/v100/'

charades_root = '/mnt/newdisk/mnt/newdisk/dataset/DGC/DCCL/charades/v100/Charades_v1_all_frame/'
charades_base_frame = '/mnt/newdisk/mnt/newdisk/dataset/DGC/DCCL/charades/label/even/v100/'


# OSR Split dir
osr_split_dir = ''

# -----------------
# OTHER PATHS
# -----------------
exp_root = '/mnt/result/ucf101_test/ucf101_2gpus_3_28/' # All logs and checkpoints will be saved here