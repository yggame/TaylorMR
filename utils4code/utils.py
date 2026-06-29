import os
import csv
import json
import time
import random
import numpy as np
from datetime import datetime

from collections import OrderedDict


import torch

from torch.optim import SGD, Adam, RMSprop, AdamW
from tensorboardX import SummaryWriter

try:
    from yaml import CLoader as Loader, CDumper as Dumper
    import wandb
except ImportError:
    from yaml import Loader, Dumper

def load_json(path):
    with open(path, encoding='utf8') as f:
        target_dic = json.load(f)
    return target_dic

def save_json(path, save_dic):
    with open(path, 'w', encoding='utf8') as f:
        json.dump(save_dic, f, ensure_ascii=False, indent=2)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # sets the seed for cpu
    torch.cuda.manual_seed(seed) # Sets the seed for the current GPU.
    torch.cuda.manual_seed_all(seed) #  Sets the seed for the all GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark=False
    torch.set_deterministic(True)

def numpy_random_init(worker_id):
    process_seed = torch.initial_seed()
    base_seed    = process_seed - worker_id
    ss  = np.random.SeedSequence([worker_id, base_seed])
    np.random.seed(ss.generate_state(4))


def numpy_fix_init(worker_id):
    np.random.seed(2<<16 + worker_id)
    

numpy_init_dict = {
    "train": numpy_random_init,
    "val"  : numpy_fix_init,
    "test" : numpy_fix_init
}

class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v
    
class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

        # from UNI-Net
        self.vals = []
        self.std = 0
        self.stderr = 0
        self.median = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        # from UNI-Net
        self.vals.append(val)
        self.std = np.std(self.vals)
        self.stderr = self.std / np.sqrt(self.count)
        self.median = np.median(self.vals)


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)


_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    # print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


# def ensure_path(path, remove=True):
#     basename = os.path.basename(path.rstrip('/'))
#     if os.path.exists(path):
#         if remove and (basename.startswith('_')
#                 or input('{} exists, remove? (y/[n]): '.format(path)) == 'y'):
#             shutil.rmtree(path)
#             os.makedirs(path)
#     else:
#         os.makedirs(path)

def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')
     
def ensure_path(path, remove=True):
    # basename = os.path.basename(path.rstrip('/'))
    # if os.path.exists(path):
    #     if remove and (basename.startswith('_')
    #             or input('{} exists, remove? (y/[n]): '.format(path)) == 'y'):
    #         shutil.rmtree(path)
    #         os.makedirs(path)
    # else:
    #     os.makedirs(path)
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('Path already exists. Create new name: [{:s}]'.format(new_name))
        # os.rename(path, new_name)
        os.makedirs(new_name)
        return new_name
    else:
        os.makedirs(path)
        return path


def set_save_path(save_path, remove=True):
    save_path = ensure_path(save_path, remove=remove)
    set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    return log, writer, save_path


def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot
    
def make_optimizer(param_list, optimizer_spec, load_sd=False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam,
        'rmsprop':RMSprop,
        'adamw': AdamW
    }[optimizer_spec['name'].lower()]
    optimizer = Optimizer(param_list, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer

def update_summary(
        epoch,
        train_metrics,
        eval_metrics,
        filename,
        lr=None,
        write_header=False,
        log_wandb=False,
):
    rowd = OrderedDict(epoch=epoch)
    if train_metrics:
        rowd.update([('train_' + k, v) for k, v in train_metrics.items()])
    if eval_metrics:
        rowd.update([('eval_' + k, v) for k, v in eval_metrics.items()])
    if lr is not None:
        rowd['lr'] = lr
    if log_wandb:
        wandb.log(rowd)
    with open(filename, mode='a') as cf:
        dw = csv.DictWriter(cf, fieldnames=rowd.keys())
        if write_header:  # first iteration (epoch == 1 can't be used)
            dw.writeheader()
        dw.writerow(rowd)


def get_number_of_learnable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])