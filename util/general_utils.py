import os
import torch
import inspect

from datetime import datetime
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def init_experiment(args,cfg, runner_name=None, exp_id=None):
    # Get filepath of calling script
    root_dir = os.path.join(cfg.OUTPUT_DIR)

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    # Either generate a unique experiment ID, or use one which is passed
    if exp_id is None:
            log_dir = os.path.join(root_dir, 'log')
    else:

        log_dir = os.path.join(root_dir, 'log', f'{exp_id}')

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    #logger = Logger(os.path.join(args.log_dir, 'log_out.txt'))

    logger.add(os.path.join(log_dir, 'log.txt'))

    args.logger = logger
    args.log_dir = log_dir

    # Instantiate directory to save models to
    model_root_dir = os.path.join(args.log_dir, 'checkpoints')
    if not os.path.exists(model_root_dir):
        os.mkdir(model_root_dir)

    args.model_dir = model_root_dir
    

    print(f'Experiment saved to: {args.log_dir}')

    args.writer = SummaryWriter(log_dir=args.log_dir)

    hparam_dict = {}

    for k, v in vars(args).items():
        if isinstance(v, (int, float, str, bool, torch.Tensor)):
            hparam_dict[k] = v

    print(runner_name)
    print(args)

    return args

def get_mean_lr(optimizer):
    return torch.mean(torch.Tensor([param_group['lr'] for param_group in optimizer.param_groups])).item()


class DistributedWeightedSampler(torch.utils.data.distributed.DistributedSampler):

    def __init__(self, dataset, weights, num_samples, num_replicas=None, rank=None,
                 replacement=True, generator=None):
        super(DistributedWeightedSampler, self).__init__(dataset, num_replicas, rank)
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or \
                num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(num_samples))
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator
        self.weights = self.weights[self.rank::self.num_replicas]
        self.num_samples = self.num_samples // self.num_replicas

    def __iter__(self):
        rand_tensor = torch.multinomial(self.weights, self.num_samples, self.replacement, generator=self.generator)
        rand_tensor =  self.rank + rand_tensor * self.num_replicas
        yield from iter(rand_tensor.tolist())

    def __len__(self):
        return self.num_samples

class Logger:

    def __init__(self, log_file):
        '''/path/to/log_file.txt'''
        self.log_file = log_file

    def __call__(self, input):

        f.writelines(input+'\n')
        print(input)
