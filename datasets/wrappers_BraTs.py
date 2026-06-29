import os
import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from .dataset_brats.transforms_Brats import *

from datasets import register

@register('BraTs_SuperResolution')
class BraTs_SuperResolution(Dataset):
    def __init__(self, dataset, img_size=256, scale_factor=2.0):
        self.dataset = dataset
        self.transform = SuperResolutionTransform(img_size, scale_factor)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample, slice_num = self.dataset[index]

        # pd_kspace, pd_mask, pd_target, attrs, pd_fname, pd_slice = pd_sample
        # pdfs_kspace, pdfs_mask, pdfs_target, attrs, pdfs_fname, pdfs_slice = pd_sample
        t2_data = sample['t2_data']
        t1_data = sample['t1_data']
        # t1ce_data = sample['t1ce_data']
        # flair_data = sample['flair_data']

        fname = sample['fname']

        target_img_t2 = t2_data
        reference_img_t1 = t1_data

        target_img, target_img_clpx, target_img_k, LR_target_img, LR_target_img_cplx, LR_target_img_k, LR_target_SR_zero_pad = self.transform(target_img_t2)       # LR_image, SR_zero_pad_image, target, mean, std, fname, slice_num
        reference_img, reference_img_clpx, reference_img_k, LR_reference_img, LR_reference_img_cplx, LR_reference_img_k, LR_reference_SR_zero_pad = self.transform(reference_img_t1)
        
        return {
            "target_img": torch.from_numpy(target_img).float().unsqueeze(0),
            'target_Kspace': torch.from_numpy(target_img_k).unsqueeze(0),
            'LR_img_target': torch.from_numpy(LR_target_img).float().unsqueeze(0),
            'LR_Kspace_target': torch.from_numpy(LR_target_img_k).unsqueeze(0), # 低分辨率 T2 K空间
            'SR_zero_pad_image_target': torch.from_numpy(LR_target_SR_zero_pad).float().unsqueeze(0),
            

            "reference_img": torch.from_numpy(reference_img).float().unsqueeze(0),
            'reference_Kspace': torch.from_numpy(reference_img_k).unsqueeze(0),
            'LR_img_reference': torch.from_numpy(LR_reference_img).float().unsqueeze(0),
            'LR_Kspace_reference': torch.from_numpy(LR_reference_img_k).unsqueeze(0), # 低分辨率 T2 K空间
            'SR_zero_pad_image_reference': torch.from_numpy(LR_reference_SR_zero_pad).float().unsqueeze(0),

            "fname": fname
        }

@register('BraTs_Reconstruction')
class BraTs_Reconstruction(Dataset):
    def __init__(self, dataset, img_size=256, mask_type='random', acceleration_rate=4):
        self.dataset = dataset
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
            mask_name4hr = str(self.img) + str(mask_type) + '_acceleration' + str(acceleration_rate) + '_center_fraction' + str(self.cfg['center_fraction'])    # 256equispaced_mask_type_low_frequency_acceleration4_acs_lines16.mat
        else:
            raise ValueError('Unknown mask type: {}'.format(mask_type))

        mask_path4hr = os.path.join('./mask/', mask_name4hr+'.mat')
        mask4hr = sio.loadmat(mask_path4hr)['mask']
        # mask = torch.from_numpy(mask).float()
        self.mask4hr = mask4hr

        self.transform = ReconstructionTransform(img_size)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample, slice_num = self.dataset[index]

        # pd_kspace, pd_mask, pd_target, attrs, pd_fname, pd_slice = pd_sample
        # pdfs_kspace, pdfs_mask, pdfs_target, attrs, pdfs_fname, pdfs_slice = pd_sample
        t2_data = sample['t2_data']
        t1_data = sample['t1_data']
        t1ce_data = sample['t1ce_data']
        flair_data = sample['flair_data']

        fname = sample['fname']

        target_img_t2 = t2_data
        reference_img_t1 = t1_data

        target_img, target_img_clpx, target_img_k, UnderSample_target_img, UnderSample_target_img_cplx, UnderSample_target_img_k = self.transform(target_img_t2, self.mask4hr)       # image, target, mean, std, fname, slice_num
        reference_img, reference_img_clpx, reference_img_k, UnderSample_reference_img, UnderSample_reference_img_cplx, UnderSample_reference_img_k = self.transform(reference_img_t1, self.mask4hr)   # image, target, mean, std, fname, slice_num

        # return {
        #     "target_sample": pd_sample,
        #     "reference_sample": pdfs_sample,
        #     "id": id
        # }
        return {
            "target_img": torch.from_numpy(target_img).float().unsqueeze(0),
            'target_Kspace': torch.from_numpy(target_img_k).unsqueeze(0),
            'under_sample_img_target': torch.from_numpy(UnderSample_target_img).float().unsqueeze(0),
            'under_sample_Kspace_target': torch.from_numpy(UnderSample_target_img_k).unsqueeze(0), 

            "reference_img": torch.from_numpy(reference_img).float().unsqueeze(0),
            'reference_Kspace': torch.from_numpy(reference_img_k).unsqueeze(0),
            'under_sample_img_reference': torch.from_numpy(UnderSample_reference_img).float().unsqueeze(0),
            'under_sample_Kspace_reference': torch.from_numpy(UnderSample_reference_img_k).unsqueeze(0), 

            "fname": fname
        }
    
@register('BraTs_JointSrReconstruction')
class BraTs_JointSrReconstruction(Dataset):
    def __init__(self, dataset, img_size=256, scale=4, mask_type='random', acceleration_rate=4):
        self.dataset = dataset
        self.img_size = img_size

        self.crop_size = int(img_size / scale)

        if mask_type == 'random_uniform':
            if acceleration_rate == 4:
                mask_name = str(self.crop_size) + str(mask_type) + '_acceleration' + str(acceleration_rate) + '_center_fraction' + str(0.08)    # 256random_uniform_acceleration4_center_fraction0.08.png
                mask_name4hr = str(self.img_size) + str(mask_type) + '_acceleration' + str(acceleration_rate) + '_center_fraction' + str(0.08)    # 256random_uniform_acceleration4_center_fraction0.08.png
            elif acceleration_rate == 8:
                mask_name = str(self.crop_size) + str(mask_type) + '_acceleration' + str(acceleration_rate) + '_center_fraction' + str(0.04)    # 256random_uniform_acceleration8_center_fraction0.04.png
                mask_name4hr = str(self.img_size) + str(mask_type) + '_acceleration' + str(acceleration_rate) + '_center_fraction' + str(0.04)    # 256random_uniform_acceleration8_center_fraction0.04.png
        elif mask_type == 'equispaced_mask_type_uniform_frequency':
            mask_name = str(self.crop_size) + str(mask_type) + '_acceleration' + str(acceleration_rate) + '_acs_lines' + str(self.cfg['acs_lines'])    # 256equispaced_mask_type_low_frequency_acceleration4_acs_lines16.mat
            mask_name4hr = str(self.img_size) + str(mask_type) + '_acceleration' + str(acceleration_rate) + '_acs_lines' + str(self.cfg['acs_lines'])    # 256equispaced_mask_type_low_frequency_acceleration4_acs_lines16.mat
        elif mask_type == 'equispaced_mask_type_low_frequency':
            mask_name = str(self.crop_size) + str(mask_type) + '_acceleration' + str(acceleration_rate) + '_center_fraction' + str(self.cfg['center_fraction'])    # 256equispaced_mask_type_low_frequency_acceleration4_acs_lines16.mat
            mask_name4hr = str(self.img) + str(mask_type) + '_acceleration' + str(acceleration_rate) + '_center_fraction' + str(self.cfg['center_fraction'])    # 256equispaced_mask_type_low_frequency_acceleration4_acs_lines16.mat
        else:
            raise ValueError('Unknown mask type: {}'.format(mask_type))
        
        mask_path = os.path.join('./mask', mask_name+'.mat')
        mask = sio.loadmat(mask_path)['mask']
        # mask = torch.from_numpy(mask).float()
        self.mask = mask

        mask_path4hr = os.path.join('./mask/', mask_name4hr+'.mat')
        mask4hr = sio.loadmat(mask_path4hr)['mask']
        # mask = torch.from_numpy(mask).float()
        self.mask4hr = mask4hr
        
        self.transform = JointSrReconstructionTransform(img_size, scale)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample, slice_num = self.dataset[index]

        # pd_kspace, pd_mask, pd_target, attrs, pd_fname, pd_slice = pd_sample
        # pdfs_kspace, pdfs_mask, pdfs_target, attrs, pdfs_fname, pdfs_slice = pd_sample
        t2_data = sample['t2_data']
        t1_data = sample['t1_data']
        t1ce_data = sample['t1ce_data']
        flair_data = sample['flair_data']

        fname = sample['fname']

        target_img_t2 = t2_data
        reference_img_t1 = t1_data

        target_img, target_img_clpx, target_img_k, LR_UnderSample_target_img, LR_UnderSample_target_img_cplx, LR_UnderSample_target_img_k, LR_UnderSample_target_SR_zero_pad = self.transform(target_img_t2, self.mask)       # image, target, mean, std, fname, slice_num
        reference_img, reference_img_clpx, reference_img_k, LR_UnderSample_reference_img, LR_UnderSample_reference_img_cplx, LR_UnderSample_reference_img_k, LR_UnderSample_reference_SR_zero_pad = self.transform(reference_img_t1, self.mask)   # image, reference, mean, std, fname, slice_num

        # return {
        #     "target_sample": pd_sample,
        #     "reference_sample": pdfs_sample,
        #     "id": id
        # }
        return {
            "target_img": torch.from_numpy(target_img).float().unsqueeze(0),
            'target_Kspace': torch.from_numpy(target_img_k).unsqueeze(0),
            'LR_UnderSample_img_target': torch.from_numpy(LR_UnderSample_target_img).float().unsqueeze(0),
            'LR_UnderSample_Kspace_target': torch.from_numpy(LR_UnderSample_target_img_k).unsqueeze(0), # 低分辨率 T2 K空间
            'SR_zero_pad_image_target': torch.from_numpy(LR_UnderSample_target_SR_zero_pad).unsqueeze(0),
            

            "reference_img": torch.from_numpy(reference_img).float().unsqueeze(0),
            'reference_Kspace': torch.from_numpy(reference_img_k).unsqueeze(0),
            'LR_UnderSample_img_reference': torch.from_numpy(LR_UnderSample_reference_img).float().unsqueeze(0),
            'LR_UnderSample_Kspace_reference': torch.from_numpy(LR_UnderSample_reference_img_k).unsqueeze(0), # 低分辨率 T2 K空间
            'SR_zero_pad_image_reference': torch.from_numpy(LR_UnderSample_reference_SR_zero_pad).unsqueeze(0),

            "fname": fname
        }
    
# @register('BraTs_Denoise')
# class BraTs_Reconstruction(Dataset):
#     def __init__(self, dataset, size, noise_rate):
#         self.dataset = dataset
#         self.transform = DenoiseDataTransform(size, noise_rate)
    
#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, index):
#         pd_sample, pdfs_sample, id = self.dataset[index]

#         pd_sample = self.transform(*pd_sample)       # noise_image, target, mean, std, fname, slice_num
#         pdfs_sample = self.transform(*pdfs_sample)   # noise_image, target, mean, std, fname, slice_num

#         # return {
#         #     "target_sample": pd_sample,
#         #     "reference_sample": pdfs_sample,
#         #     "id": id
#         # }
#         return {
#             "target_img": pd_sample[1],
#             'noise_img_target': pd_sample[0],
#             'target_mean': pd_sample[2],
#             'target_std': pd_sample[3],
#             'target_fname': pd_sample[4],
#             'target_slice_num': pd_sample[5],

#             "reference_img": pdfs_sample[1],
#             "noise_img_reference": pdfs_sample[0],
#             "reference_mean": pdfs_sample[2],
#             "reference_std": pdfs_sample[3],
#             "reference_fname": pdfs_sample[4],
#             "reference_slice_num": pdfs_sample[5],

#             "id": id
#         }

