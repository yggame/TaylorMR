import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from .transforms_fastMRI import *
from .subsample_fastMRI import create_mask_for_mask_type

from datasets import register

@register('fastMRI_SuperResolution')
class fastMRI_SuperResolution(Dataset):
    def __init__(self, dataset, which_challenge="singlecoil", scale_factor=2.0):
        self.dataset = dataset
        self.transform = SuperResolutionTransform(which_challenge, scale_factor)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pd_sample, pdfs_sample, id = self.dataset[index]

        # pd_kspace, pd_mask, pd_target, attrs, pd_fname, pd_slice = pd_sample
        # pdfs_kspace, pdfs_mask, pdfs_target, attrs, pdfs_fname, pdfs_slice = pd_sample

        pd_sample = self.transform(*pd_sample)       # LR_image, SR_zero_pad_image, target, mean, std, fname, slice_num
        pdfs_sample = self.transform(*pdfs_sample)   # LR_image, SR_zero_pad_image, target, mean, std, fname, slice_num

        # return {
        #     "target_sample": pd_sample,
        #     "reference_sample": pdfs_sample,
        #     "id": id
        # }
        return {
            "target_img": pd_sample[2],
            'LR_img_target': pd_sample[0],
            'SR_zero_pad_image_target': pd_sample[1],
            'target_mean': pd_sample[3],
            'target_std': pd_sample[4],
            'target_fname': pd_sample[5],
            'target_slice_num': pd_sample[6],

            "reference_img": pdfs_sample[2],
            "LR_img_reference": pdfs_sample[0],
            'SR_zero_pad_image_reference': pdfs_sample[1],
            "reference_mean": pdfs_sample[3],
            "reference_std": pdfs_sample[4],
            "reference_fname": pdfs_sample[5],
            "reference_slice_num": pdfs_sample[6],

            'target_fname': os.path.basename(pd_sample[5]).split('.')[0] + '_'+ str(pd_sample[6]),
            'reference_fname': os.path.basename(pdfs_sample[5]).split('.')[0] + '_' + str(pdfs_sample[6]),
            "id": id
        }

@register('fastMRI_Reconstruction')
class fastMRI_Reconstruction(Dataset):
    def __init__(self, dataset, which_challenge, MASKTYPE='random', CENTER_FRACTIONS=0.08, ACCELERATIONS=4, use_seed=True, mode="train"):
        self.dataset = dataset

        if mode == "train":
            mask_func = create_mask_for_mask_type(
                    MASKTYPE, CENTER_FRACTIONS, ACCELERATIONS,
                )
            self.transform = ReconstructionTransform(which_challenge, mask_func, use_seed=use_seed)
        
        else:
            self.transform = ReconstructionTransform(which_challenge)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pd_sample, pdfs_sample, id = self.dataset[index]

        pd_sample = self.transform(*pd_sample)       # image, target, mean, std, fname, slice_num
        pdfs_sample = self.transform(*pdfs_sample)   # image, target, mean, std, fname, slice_num

        # return {
        #     "target_sample": pd_sample,
        #     "reference_sample": pdfs_sample,
        #     "id": id
        # }
        return {
            "target_img": pd_sample[1],
            'under_sample_img_target': pd_sample[0],
            'target_mean': pd_sample[2],
            'target_std': pd_sample[3],
            'target_fname': pd_sample[4],
            'target_slice_num': pd_sample[5],

            "reference_img": pdfs_sample[1],
            "under_sample_img_reference": pdfs_sample[0],
            "reference_mean": pdfs_sample[2],
            "reference_std": pdfs_sample[3],
            "reference_fname": pdfs_sample[4],
            "reference_slice_num": pdfs_sample[5],

            "id": id
        }
    
@register('fastMRI_JointSrReconstruction')
class fastMRI_JointSrReconstruction(Dataset):
    def __init__(self, dataset, which_challenge, MASKTYPE='random', CENTER_FRACTIONS=0.08, ACCELERATIONS=4, scale_factor=4, use_seed=True, mode="train"):
        self.dataset = dataset

        if mode == "train":
            mask_func = create_mask_for_mask_type(
                    MASKTYPE, CENTER_FRACTIONS, ACCELERATIONS,
                )
            self.transform = JointSrReconstructionTransform(which_challenge, mask_func, scale_factor, use_seed=use_seed)
        
        else:
            self.transform = JointSrReconstructionTransform(which_challenge)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pd_sample, pdfs_sample, id = self.dataset[index]

        pd_sample = self.transform(*pd_sample)       # UnderSample_LR_image, SR_zero_pad_image, target, mean, std, fname, slice_num
        pdfs_sample = self.transform(*pdfs_sample)   # UnderSample_LR_image, SR_zero_pad_image, target, mean, std, fname, slice_num

        # return {
        #     "target_sample": pd_sample,
        #     "reference_sample": pdfs_sample,
        #     "id": id
        # }
        return {
            "target_img": pd_sample[2],
            'under_sample_LR_img_target': pd_sample[0],
            'SR_zero_pad_image_target': pd_sample[1],
            'target_mean': pd_sample[3],
            'target_std': pd_sample[4],
            'target_fname': pd_sample[5],
            'target_slice_num': pd_sample[6],

            "reference_img": pdfs_sample[2],
            "under_sample_LR_img_reference": pdfs_sample[0],
            'SR_zero_pad_image_reference': pdfs_sample[1],
            "reference_mean": pdfs_sample[3],
            "reference_std": pdfs_sample[4],
            "reference_fname": pdfs_sample[5],
            "reference_slice_num": pdfs_sample[6],

            "id": id
        }
    
@register('fastMRI_Denoise')
class fastMRI_Reconstruction(Dataset):
    def __init__(self, dataset, size, noise_rate):
        self.dataset = dataset
        self.transform = DenoiseDataTransform(size, noise_rate)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pd_sample, pdfs_sample, id = self.dataset[index]

        pd_sample = self.transform(*pd_sample)       # noise_image, target, mean, std, fname, slice_num
        pdfs_sample = self.transform(*pdfs_sample)   # noise_image, target, mean, std, fname, slice_num

        # return {
        #     "target_sample": pd_sample,
        #     "reference_sample": pdfs_sample,
        #     "id": id
        # }
        return {
            "target_img": pd_sample[1],
            'noise_img_target': pd_sample[0],
            'target_mean': pd_sample[2],
            'target_std': pd_sample[3],
            'target_fname': pd_sample[4],
            'target_slice_num': pd_sample[5],

            "reference_img": pdfs_sample[1],
            "noise_img_reference": pdfs_sample[0],
            "reference_mean": pdfs_sample[2],
            "reference_std": pdfs_sample[3],
            "reference_fname": pdfs_sample[4],
            "reference_slice_num": pdfs_sample[5],

            "id": id
        }

