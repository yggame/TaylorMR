from os.path import splitext
from os import listdir, path

from torch.utils.data import Dataset
import logging

import h5py
import pickle
from scipy.io import loadmat, savemat
import scipy.io as sio
import torch
import cv2
import numpy as np
import os

from .transforms_ixi import *

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import register

@register('ixi_dataset')
class IXI_dataset(Dataset):
    def __init__(self, split, modality1='T2', modality2='PD', list_dir=None, data_dir=None):  
        self.list_dir = list_dir #self.cfg['list_dir']
        self.split = split
        
        self.modality1 = modality1 # 'T2'
        self.modality2 = modality2 # 'PD'

        # print('load T2: ', self.load_T2)

        self.sample_list = open(os.path.join(self.list_dir, 'IXI_' + str(self.split) +'.txt')).readlines()
        self.data_dir = data_dir

        logging.info(f'Creating {self.split} dataset with {len(self.sample_list)} examples')
        print(f'Creating {self.split} dataset with {len(self.sample_list)} examples')
        
    def __len__(self):
        return len(self.sample_list)

    def ifft2(self, kspace_cplx):
        return np.absolute(np.fft.ifft2(kspace_cplx))[None, :, :]

    def fft2(self, img):
        return np.fft.fftshift(np.fft.fft2(img))

    def inverseFT(self, Kspace):
        Kspace = Kspace.permute(0, 2, 3, 1)  # last dimension=2
        img_cmplx = torch.ifft(Kspace, 2)
        img = torch.sqrt(img_cmplx[:, :, :, 0] ** 2 + img_cmplx[:, :, :, 1] ** 2)
        img = img[:, None, :, :]
        return img

    def __getitem__(self, i):

        fname = self.sample_list[i].strip('\n')
        
        full_file_path = path.join(self.data_dir, fname + '.h5')
        
        # 读取h5文件
        with h5py.File(full_file_path, 'r') as f:
            img_T2 = f['T2'][()]
            img_PD = f['PD'][()]

        # img_T2, meanT2, stdT2= normalize_instance(img_T2, eps=1e-11)
        # img_PD, meanPD, stdPD= normalize_instance(img_PD, eps=1e-11)

        # img_T2_height, img_T2_width = img_T2.shape
        # img_T2_matRotate = cv2.getRotationMatrix2D((img_T2_height * 0.5, img_T2_width * 0.5), 90, 1)
        # img_T2 = cv2.warpAffine(img_T2, img_T2_matRotate, (img_T2_height, img_T2_width))


        kspace_T2 = self.fft2(img_T2)  
        kspace_PD = self.fft2(img_PD)  

        ret = {
            'img_T2': img_T2,
            'img_PD': img_PD,
            'kspace_T2': torch.from_numpy(kspace_T2),
            'kspace_PD': torch.from_numpy(kspace_PD),
            'fname': fname,
        }
        return ret

@register('ixi_dataset_SR')
class IXI_dataset_SR(Dataset):
    def __init__(self, split, modality1='T2', modality2='PD', 
                 list_dir=None, data_dir=None,
                 img_size=256, scale=2.0):  
        self.list_dir = list_dir #self.cfg['list_dir']
        self.split = split
        
        self.modality1 = modality1 # 'T2'
        self.modality2 = modality2 # 'PD'

        self.img_size = img_size
        self.scale_factor = scale

        self.crop_size = int(img_size / self.scale_factor)

        self.transform = SuperResolutionTransform(img_size, self.scale_factor)

        # print('load T2: ', self.load_T2)

        self.sample_list = open(os.path.join(self.list_dir, 'IXI_' + str(self.split) +'.txt')).readlines()
        self.data_dir = data_dir

        logging.info(f'Creating {self.split} dataset with {len(self.sample_list)} examples')
        print(f'Creating {self.split} dataset with {len(self.sample_list)} examples')
        
    def __len__(self):
        return len(self.sample_list)

    def ifft2(self, kspace_cplx):
        return np.absolute(np.fft.ifft2(kspace_cplx))[None, :, :]

    def fft2(self, img):
        return np.fft.fftshift(np.fft.fft2(img))

    def inverseFT(self, Kspace):
        Kspace = Kspace.permute(0, 2, 3, 1)  # last dimension=2
        img_cmplx = torch.ifft(Kspace, 2)
        img = torch.sqrt(img_cmplx[:, :, :, 0] ** 2 + img_cmplx[:, :, :, 1] ** 2)
        img = img[:, None, :, :]
        return img

    def __getitem__(self, i):

        fname = self.sample_list[i].strip('\n')
        
        full_file_path = path.join(self.data_dir, fname + '.h5')
        
        # 读取h5文件
        with h5py.File(full_file_path, 'r') as f:
            img_T2 = f['T2'][()]
            img_PD = f['PD'][()]

        # img_T2, meanT2, stdT2= normalize_instance(img_T2, eps=1e-11)
        # img_PD, meanPD, stdPD= normalize_instance(img_PD, eps=1e-11)

        # img_T2_height, img_T2_width = img_T2.shape
        # img_T2_matRotate = cv2.getRotationMatrix2D((img_T2_height * 0.5, img_T2_width * 0.5), 90, 1)
        # img_T2 = cv2.warpAffine(img_T2, img_T2_matRotate, (img_T2_height, img_T2_width))


        kspace_T2 = self.fft2(img_T2)  
        kspace_PD = self.fft2(img_PD)  

        slice_full_img_T2, slice_full_img_T2_clpx, slice_full_img_T2_k, LR_T2_img, LR_T2_img_cplx, LR_T2_k, LR_T2_SR_zero_fill = self.transform(img_T2)      # LR, LR_cplx, LR_mask, LR_ori, LR_ori_cplx, LR_ori_k, kspace, HR, HR_cplx, HR_k (kspace=HR_k)
        slice_full_img_PD, slice_full_img_PD_clpx, slice_full_img_PD_k, LR_PD_img, LR_PD_img_cplx, LR_PD_k, LR_PD_SR_zero_fill = self.transform(img_PD)      # LR, LR_cplx, LR_mask, LR_ori, LR_ori_cplx, LR_ori_k, kspace, HR, HR_cplx, HR_k
        # kspace, HR, HR_cplx, HR_k, LR, LR_cplx, LR_k

        # if fname == 'IXI562-Guys-1131-070' or fname == 'IXI303-IOP-0968-050':
        #     print(fname)

        return {
            'target_Kspace': torch.from_numpy(slice_full_img_T2_k),
            'target_img': torch.from_numpy(slice_full_img_T2).float().unsqueeze(0),
            'LR_Kspace_target': torch.from_numpy(LR_T2_k), # 低分辨率 T2 K空间
            'LR_img_target': torch.from_numpy(LR_T2_img).float().unsqueeze(0),  # 欠采样img
            'SR_zero_fill_img_target': torch.from_numpy(LR_T2_SR_zero_fill).float().unsqueeze(0),  # 欠采样img

            'reference_Kspace': torch.from_numpy(slice_full_img_PD_k),
            'reference_img': torch.from_numpy(slice_full_img_PD).float().unsqueeze(0),
            'LR_Kspace_reference': torch.from_numpy(LR_PD_k), # 欠采样K
            'LR_img_reference': torch.from_numpy(LR_PD_img).float().unsqueeze(0),  # 欠采样img
            'SR_zero_fill_img_reference': torch.from_numpy(LR_PD_SR_zero_fill).float().unsqueeze(0),  # 欠采样img
            'fname': fname
        }

@register('ixi_dataset_Reconstruction')
class IXI_dataset_Reconstruction(Dataset):
    def __init__(self, split, modality1='T2', modality2='PD', 
                 list_dir=None, data_dir=None,
                 img_size=256, mask_type='random', acceleration_rate=4):  
        self.list_dir = list_dir #self.cfg['list_dir']
        self.split = split
        self.img_size = img_size

        if mask_type == 'random_uniform':
            if acceleration_rate == 4:
                # mask_name = str(self.crop_size) + str(mask_type) + '_acceleration' + str(acceleration_rate) + '_center_fraction' + str(0.08)    # 256random_uniform_acceleration4_center_fraction0.08.png
                mask_name4hr = str(self.img_size) + str(mask_type) + '_acceleration' + str(acceleration_rate) + '_center_fraction' + str(0.08)    # 256random_uniform_acceleration4_center_fraction0.08.png
            elif acceleration_rate == 8:
                # mask_name = str(self.crop_size) + str(mask_type) + '_acceleration' + str(acceleration_rate) + '_center_fraction' + str(0.04)    # 256random_uniform_acceleration8_center_fraction0.04.png
                mask_name4hr = str(self.img_size) + str(mask_type) + '_acceleration' + str(acceleration_rate) + '_center_fraction' + str(0.04)    # 256random_uniform_acceleration8_center_fraction0.04.png
        elif mask_type == 'equispaced_mask_type_uniform_frequency':
            # mask_name = str(self.crop_size) + str(mask_type) + '_acceleration' + str(acceleration_rate) + '_acs_lines' + str(self.cfg['acs_lines'])    # 256equispaced_mask_type_low_frequency_acceleration4_acs_lines16.mat
            mask_name4hr = str(self.img_size) + str(mask_type) + '_acceleration' + str(acceleration_rate) + '_acs_lines' + str(self.cfg['acs_lines'])    # 256equispaced_mask_type_low_frequency_acceleration4_acs_lines16.mat
        elif mask_type == 'equispaced_mask_type_low_frequency':
            # mask_name = str(self.crop_size) + str(mask_type) + '_acceleration' + str(acceleration_rate) + '_center_fraction' + str(self.cfg['center_fraction'])    # 256equispaced_mask_type_low_frequency_acceleration4_acs_lines16.mat
            mask_name4hr = str(self.img_size) + str(mask_type) + '_acceleration' + str(acceleration_rate) + '_center_fraction' + str(self.cfg['center_fraction'])    # 256equispaced_mask_type_low_frequency_acceleration4_acs_lines16.mat
        else:
            raise ValueError('Unknown mask type: {}'.format(mask_type))

        mask_path4hr = os.path.join('./mask/', mask_name4hr+'.mat')
        mask4hr = sio.loadmat(mask_path4hr)['mask']
        # mask = torch.from_numpy(mask).float()
        self.mask4hr = mask4hr

        self.transform = ReconstructionTransform(img_size)

        ###########################
        
        self.modality1 = modality1 # 'T2'
        self.modality2 = modality2 # 'PD'

        # print('load T2: ', self.load_T2)

        self.sample_list = open(os.path.join(self.list_dir, 'IXI_' + str(self.split) +'.txt')).readlines()
        self.data_dir = data_dir

        logging.info(f'Creating {self.split} dataset with {len(self.sample_list)} examples')
        print(f'Creating {self.split} dataset with {len(self.sample_list)} examples')
        
    def __len__(self):
        return len(self.sample_list)

    def ifft2(self, kspace_cplx):
        return np.absolute(np.fft.ifft2(kspace_cplx))[None, :, :]

    def fft2(self, img):
        return np.fft.fftshift(np.fft.fft2(img))

    def inverseFT(self, Kspace):
        Kspace = Kspace.permute(0, 2, 3, 1)  # last dimension=2
        img_cmplx = torch.ifft(Kspace, 2)
        img = torch.sqrt(img_cmplx[:, :, :, 0] ** 2 + img_cmplx[:, :, :, 1] ** 2)
        img = img[:, None, :, :]
        return img

    def __getitem__(self, i):

        fname = self.sample_list[i].strip('\n')
        
        full_file_path = path.join(self.data_dir, fname + '.h5')
        
        # 读取h5文件
        with h5py.File(full_file_path, 'r') as f:
            img_T2 = f['T2'][()]
            img_PD = f['PD'][()]

        # img_T2, meanT2, stdT2= normalize_instance(img_T2, eps=1e-11)
        # img_PD, meanPD, stdPD= normalize_instance(img_PD, eps=1e-11)

        # img_T2_height, img_T2_width = img_T2.shape
        # img_T2_matRotate = cv2.getRotationMatrix2D((img_T2_height * 0.5, img_T2_width * 0.5), 90, 1)
        # img_T2 = cv2.warpAffine(img_T2, img_T2_matRotate, (img_T2_height, img_T2_width))


        kspace_T2 = self.fft2(img_T2)  
        kspace_PD = self.fft2(img_PD)  

        target_img, target_img_clpx, target_img_k, UnderSample_target_img, UnderSample_target_img_cplx, UnderSample_target_img_k = self.transform(img_T2, self.mask4hr)       # image, target, mean, std, fname, slice_num
        reference_img, reference_img_clpx, reference_img_k, UnderSample_reference_img, UnderSample_reference_img_cplx, UnderSample_reference_img_k = self.transform(img_PD, self.mask4hr)   # image, target, mean, std, fname, slice_num


        return {
            "target_img": torch.from_numpy(target_img).float().unsqueeze(0),
            "target_img_cplx": torch.from_numpy(target_img_clpx),
            'target_Kspace': torch.from_numpy(target_img_k).unsqueeze(0),
            'under_sample_img_target': torch.from_numpy(UnderSample_target_img).float().unsqueeze(0),
            "under_sample_img_target_clpx": torch.from_numpy(UnderSample_target_img_cplx),
            'under_sample_Kspace_target': torch.from_numpy(UnderSample_target_img_k).unsqueeze(0), 

            "reference_img": torch.from_numpy(reference_img).float().unsqueeze(0),
            'reference_Kspace': torch.from_numpy(reference_img_k).unsqueeze(0),
            'under_sample_img_reference': torch.from_numpy(UnderSample_reference_img).float().unsqueeze(0),
            'under_sample_Kspace_reference': torch.from_numpy(UnderSample_reference_img_k).unsqueeze(0), 

            'mask_target': self.mask4hr,
            'mask_reference': self.mask4hr,

            "fname": fname
        }



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import argparse
    import torch 
    from torch.utils.data import DataLoader
    
    parser = argparse.ArgumentParser(description='DataLoader')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--img_size', type=int, default=256, help='size of image')
    
    
    args = parser.parse_args()
    
    cfg = {}
    
    cfg['scale'] = 4
    cfg['img_size'] = args.img_size
    cfg['mask_mode'] = 'random_uniform'
    cfg['list_dir'] = './IXI_dataset/list_file_IXI_T2_PD/'
    cfg['data_dir'] = './IXI_dataset/IXI-T2-PD-H5'
    
    
    # 测试数据集
    # list_dir = './IXI_dataset/list_file_IXI_T2_PD/IXI_train.txt'
    dataset_ixi = IXI_dataset(args=args,
                              cfg=cfg,
                              split='train'
    )
    print(len(dataset_ixi))
    
    dataloader_ixi = DataLoader(dataset_ixi, batch_size=1, shuffle=True, num_workers=0)
    
    for i, data in enumerate(dataloader_ixi):
        print(i, data['slice_LR_T2'].shape, data['target_img_T2'].shape)