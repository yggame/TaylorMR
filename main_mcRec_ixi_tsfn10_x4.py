import os, sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR

from tqdm import tqdm
import numpy as np

import time
import yaml
import random
import argparse
from collections import OrderedDict

import datasets
import models
from utils4code import utils
from utils4code.losses import CharbonnierLoss, LapLoss, fftLoss
from utils4code.metrics import calculate_psnr, calculate_ssim, calculate_psnr_fastmri
from utils4code.lr_scheduler import MultiStepLR_Restart, CosineAnnealingLR_Restart

def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    # dataset = datasets.make(spec['dataset'])
    # dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    dataset = datasets.make(spec['dataset'])

    # log(f'{tag} dataset: size={len(dataset)}')
    # for k, v in dataset[0].items():
    #     if isinstance(v, float) or isinstance(v, int):
    #         log(f'  {k}: {v}')
    #     else:
    #         log(f'  {k}: shape={tuple(v.shape)}')
    
    loader = DataLoader(
                    dataset,
                    batch_size = spec['batch_size'],
                    shuffle = (tag == 'train'),
                    num_workers = spec['n_workers'],
                    pin_memory = True, 
                    worker_init_fn = utils.numpy_init_dict[tag]
                )
    return loader

def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag = 'train')
    val_loader = make_data_loader(config.get('val_dataset'), tag = 'val')
    return train_loader, val_loader

def prepare_training():
    if config.get('pre_train') is not None:
        print('loading pre_train model... ', config['pre_train'])
        model = models.make(config['model']).cuda()
        model_dict = model.state_dict()

        sv_file = torch.load(config['pre_train'])
        pretrained_dict = sv_file['model']['sd']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
            
    elif config.get('resume') is not None:
        sv_file = torch.load(config['resume'])
        model = models.make(sv_file['model'], load_sd=True).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), sv_file['optimizer'], load_sd=True)
        epoch_start = sv_file['epoch'] + 1
        state = sv_file['state']
        torch.set_rng_state(state)
        print(f'Resuming from epoch {epoch_start}...')
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        
        lr_scheduler.last_epoch = epoch_start - 1

    else:
        print('prepare_training from start')
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        elif config['multi_step_lr']['name'] == 'CosineAnnealingLR_Restart':
            lr_scheduler = CosineAnnealingLR_Restart(optimizer, **config['multi_step_lr']['args'])
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        for _ in range(epoch_start - 1):
            lr_scheduler.step()

    log(f'model: #params={utils.compute_num_params(model, text=True)}')
    log(f'model: #struct={model}')

    return model, optimizer, epoch_start, lr_scheduler


def main(config_, save_path):
    global config, log, writer
    config = config_
    log, writer, save_path = utils.set_save_path(save_path)

    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader = make_data_loaders()
    
    model, optimizer, epoch_start, lr_scheduler = prepare_training()

    # n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    # if n_gpus > 1:
    #     model = nn.parallel.DataParallel(model)
    
    device = torch.device ("cuda" if torch.cuda.is_available () else "cpu")
    loss_type = config['loss_pixel_criterion']
    if loss_type == 'l1':
        cri_pix = nn.L1Loss(reduction='mean').to(device)
    elif loss_type == 'l2':
        cri_pix = nn.MSELoss(reduction='sum').to(device)
    elif loss_type == 'cb':
        cri_pix = CharbonnierLoss().to(device)
    elif loss_type == 'lp':
        cri_pix = LapLoss(max_levels=5).to(device)

    loss_fn = nn.L1Loss()

    epoch_max = config['epoch_max']
    epoch_save = config.get('epoch_save')

    save_checkpoint_freq = config.get('save_checkpoint_freq')
    
    max_val_v = -1e18

    current_step = 0

    timer = utils.Timer()

    # begin training
    train_summary_flag = True

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = [f'epoch {epoch}/{epoch_max}']

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        log_info.append(f'lr:{optimizer.param_groups[0]["lr"]}')

        log(', '.join(log_info))

        optimizer.zero_grad()

        # train
        # for idx, batch in enumerate(tqdm(train_loader, leave=False, desc='train')):
        pbar = tqdm(train_loader, leave=False, desc='train_epoch_{}'.format(epoch))
        for batch in pbar:
            for k, v in batch.items():
                try:
                    batch[k] = v.cuda()
                except:
                    pass
            
            model.train()
            current_step += 1
            
            log_info = [f'iter {current_step}']

            end = time.time()

            # feed data
            target_img = batch['target_img']
            under_sample_target_img = batch['under_sample_img_target']

            reference_img = batch['reference_img']
            under_sample_reference_img = batch['under_sample_img_reference']

            # # 上述Kspace
            # target_mean = batch['target_mean']
            # target_std  =  batch['target_std']
            # target_slice = batch['target_slice_num']

            # reference_mean = batch['reference_mean']
            # reference_std  =  batch['reference_std']
            # reference_slice = batch['reference_slice_num']


            # with torch.cuda.amp.autocast(): 

            im1_lr = under_sample_target_img
            im2_gt = reference_img
            
            torch.cuda.synchronize()
            fake_H = model(im1_lr, im2_gt)

            loss = cri_pix(fake_H, target_img) # + cri_pix(fake_H_2, reference_img)
                
            optimizer.zero_grad()
            # loss.requires_grad_(True)
            loss.backward()
            optimizer.step()

            # train_batch_time_m.update(time.time() - end)
            
            pbar.set_postfix({'loss': loss.item(), 'time': time.time() - end})
            # log_info.append(f'loss={loss.item()}, time: {time.time() - end}')
                # 'loss={}, time: {}'.format((loss.item(), time.time() - end)))
        

            if lr_scheduler is not None:
                lr_scheduler.step()

            log_info.append(f'train: loss={loss.item():.4f}')
            writer.add_scalars('loss', {'train': loss.item()}, current_step)


            # model save
            # if n_gpus > 1:
            #     model_ = model.module
            # else:
            model_ = model

            model_spec = config['model']
            model_spec['sd'] = model_.state_dict()
            optimizer_spec = config['optimizer']
            optimizer_spec['sd'] = optimizer.state_dict()
            
            state = torch.get_rng_state()
            sv_file = {
                'model': model_spec,
                'optimizer': optimizer_spec,
                'current_step': current_step,
                'state': state
            }

            torch.save(sv_file, os.path.join(save_path, 'iter_last.pth'))

            if (save_checkpoint_freq is not None) and (current_step % save_checkpoint_freq == 0):
                torch.save(sv_file,
                os.path.join(save_path, f'iter_{current_step}.pth'))

            # Validation  ###  ###
            if current_step % config["val_freq"] == 0:
                
                avg_psnr_im1 = 0.0

                psnr_im1 = utils.AverageMeter()
                idx = 0

                # for val_data in tqdm(val_loader): 
                pbar_val = tqdm(val_loader, leave=False, desc='val_iter_{}'.format(current_step))
                for batch in pbar_val:
                    for k, v in batch.items():
                        try:
                            batch[k] = v.cuda()
                        except:
                            pass
                        
                    end = time.time()

                    # feed data
                    # if config.get('task_mode') == 'mcsr':
                    target_img = batch['target_img']
                    under_sample_target_img = batch['under_sample_img_target']

                    reference_img = batch['reference_img']
                    under_sample_reference_img = batch['under_sample_img_reference']
                    # # 上述Kspace

                    with torch.no_grad():
                        model.eval()

                        im1_lr = under_sample_target_img
                        im2_gt = reference_img

                        fake_H = model(im1_lr, im2_gt)

                    # pbar.set_postfix({'time': time.time() - end})

                    # visilization
                    out_dict = OrderedDict()
                    out_dict['im1_restore'] = fake_H.detach().cpu()
                    out_dict['im1_GT'] = target_img.detach().cpu() 
                    
                    img_num = out_dict["im1_GT"].shape[0]
                    for i in range(img_num):
                        sr_img_1 = out_dict["im1_restore"][i, 0, :, :]  # (1, w, h)
                        gt_img_1 = out_dict["im1_GT"][i, 0, :, :]  # (1, w, h) 
                                    
                        # calculate PSNR
                        which_dataset = config['val_dataset']['dataset']['name'].lower()
                        if 'ixi' in which_dataset or 'brats' in which_dataset or 'knee' in which_dataset:
                            cur_psnr_im1 = calculate_psnr(sr_img_1.numpy()*255., gt_img_1.numpy()*255.)
                        elif 'fastmri' in which_dataset:
                            cur_psnr_im1 = calculate_psnr_fastmri(gt_img_1.numpy(), sr_img_1.numpy())

                        avg_psnr_im1 += cur_psnr_im1
                        psnr_im1.update(cur_psnr_im1)

                    idx += img_num

                    pbar_val.set_postfix({'now_val_psnr': psnr_im1.avg, 'time': time.time() - end})

                avg_psnr_im1 = avg_psnr_im1 / idx

                log_info.append(f'val: psnr={avg_psnr_im1:.4f}')
                writer.add_scalars('psnr', {'val': avg_psnr_im1}, current_step)

                if avg_psnr_im1 > max_val_v:
                    max_val_v = avg_psnr_im1

                    torch.save(sv_file, os.path.join(save_path, 'iter_best.pth'))
            
            log(', '.join(log_info))

        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file,
                os.path.join(save_path, f'epoch_{epoch}.pth'))

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed = utils.time_text(t)
        t_all = utils.time_text(t / prog)
        log_info = [f'{t_epoch} {t_elapsed}/{t_all}']

        log(', '.join(log_info))
        writer.flush()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()  
    parser.add_argument('--config', default="./configs/train/train_ixi_mcRec_model_tsfn10_x4.yaml", type=str, help='config file path')
    parser.add_argument('--tag', default = None)
    # parser.add_argument('--gpu', default = '6')
    parser.add_argument('--resume', default = None)
    args = parser.parse_args()

    
    def setup_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed) # sets the seed for cpu
        torch.cuda.manual_seed(seed) # Sets the seed for the current GPU.
        torch.cuda.manual_seed_all(seed) #  Sets the seed for the all GPU.
        torch.backends.cudnn.benchmark = True
    setup_seed(2021)
    
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader = yaml.FullLoader)
        print('config loaded.')

    save_name = args.config.split('/')[-1][len('train-'):-len('.yaml')]
    if args.tag is not None:
        save_name += args.tag
    save_path = os.path.join('./save', save_name)

    if args.resume is None:
        config['resume'] = None

    main(config, save_path)
