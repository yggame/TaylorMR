import os
import sys
import glob
import SimpleITK as sitk
import random
import scipy.io as sio

import torch
from torch.utils.data import Dataset

from .transforms_Brats import *

from datasets import register

@register('Brats2019_dataset')
class BraTs_dataset(Dataset):
    def __init__(self, data_dir='./BraTs/BraTs-2019', mode='train', sample_rate=1,):
        self.data_dir = data_dir
        # self.transform = transform
        self.files = os.listdir(data_dir)

        if mode == 'train':
            self.t2_files_path = glob.glob(os.path.join(data_dir, 'MICCAI_BraTS_2019_Data_Training/*/*/*t2.nii.gz'))
            # self.t1_files_path = glob.glob(os.path.join(data_dir, 'MICCAI_BraTS_2019_Data_Training/*/*/*t1.nii'))
            # self.t1ce_files_path = glob.glob(os.path.join(data_dir, 'MICCAI_BraTS_2019_Data_Training/*/*/*t1ce.nii'))
            # self.flair_files_path = glob.glob(os.path.join(data_dir, 'MICCAI_BraTS_2019_Data_Training/*/*/*flair.nii'))
            self.t1_files_path = [t2_fname.replace('t2.nii.gz', 't1.nii.gz') for t2_fname in self.t2_files_path]
            self.t1ce_files_path = [t2_fname.replace('t2.nii.gz', 't1ce.nii.gz') for t2_fname in self.t2_files_path]
            self.flair_files_path = [t2_fname.replace('t2.nii.gz', 'flair.nii.gz') for t2_fname in self.t2_files_path]

        else:
            self.t2_files_path = glob.glob(os.path.join(data_dir, 'MICCAI_BraTS_2019_Data_Validation/*/*t2.nii.gz'))
            # self.t1_files_path = glob.glob(os.path.join(data_dir, 'MICCAI_BraTS_2019_Data_Validation/*/*t1.nii.gz'))
            # self.t1ce_files_path = glob.glob(os.path.join(data_dir, 'MICCAI_BraTS_2019_Data_Validation/*/*t1ce.nii.gz'))
            # self.flair_files_path = glob.glob(os.path.join(data_dir, 'MICCAI_BraTS_2019_Data_Validation/*/*flair.nii.gz'))
            self.t1_files_path = [t2_fname.replace('t2.nii.gz', 't1.nii.gz') for t2_fname in self.t2_files_path]
            self.t1ce_files_path = [t2_fname.replace('t2.nii.gz', 't1ce.nii.gz') for t2_fname in self.t2_files_path]
            self.flair_files_path = [t2_fname.replace('t2.nii.gz', 'flair.nii.gz') for t2_fname in self.t2_files_path]

        slice_id = [60, 65, 70, 75, 80, 85, 90, 95, 100, 105]
        
        self.examples = []

        self.data_dir = data_dir

        for t2_fname, t1_fname, t1ce_fname, flair_fname in zip(self.t2_files_path, self.t1_files_path, self.t1ce_files_path, self.flair_files_path):
            # t2_metadata, t1_metadata, t1ce_metadata, flair_metadata = self._retrieve_metadata(t2_fname)
            for slice_num in slice_id:
                self.examples.append((t2_fname, t1_fname, t1ce_fname, flair_fname, slice_num))

        if sample_rate < 1.0:
            random.shuffle(self.examples)
            num_examples = round(len(self.examples) * sample_rate)

            self.examples = self.examples[0:num_examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        t2_fname, t1_fname, t1ce_fname, flair_fname, slice_num = self.examples[idx]
        
        fname = os.path.basename(os.path.dirname(t2_fname)) + '_slice_{}'.format(slice_num)

        t2_data = sitk.ReadImage(t2_fname)
        t2_data = sitk.GetArrayFromImage(t2_data)

        t2_data = (t2_data-t2_data.min()) / (t2_data.max()-t2_data.min())
        t2_data_255 = (255*t2_data).astype(int)

        t1_data = sitk.ReadImage(t1_fname)
        t1_data = sitk.GetArrayFromImage(t1_data)

        t1_data = (t1_data-t1_data.min()) / (t1_data.max()-t1_data.min())
        t1_data_255 = (255*t1_data).astype(int)

        # t1ce_data = sitk.ReadImage(t1ce_fname)
        # t1ce_data = sitk.GetArrayFromImage(t1ce_data)

        # t1ce_data = (t1ce_data-t1ce_data.min()) / (t1ce_data.max()-t1ce_data.min())
        # t1ce_data_255 = (255*t1ce_data).astype(int)

        # flair_data = sitk.ReadImage(flair_fname)
        # flair_data = sitk.GetArrayFromImage(flair_data)

        # flair_data = (flair_data-flair_data.min()) / (flair_data.max()-flair_data.min())
        # flair_data_255 = (255*flair_data).astype(int)

        t2_data = t2_data[slice_num]
        t1_data = t1_data[slice_num]
        # t1ce_data = t1ce_data[slice_num]
        # flair_data = flair_data[slice_num]

        sample = {'t2_data': t2_data, 
                  't1_data': t1_data, 
                #   't1ce_data': t1ce_data, 
                #   'flair_data': flair_data,
                  'fname': fname}

        return sample, slice_num


@register('Brats2019_dataset_SR')
class BraTs_dataset_SR(Dataset):
    def __init__(self, data_dir='./BraTs/BraTs-2019', 
                 mode='train', sample_rate=1, img_size=256, scale_factor=2.0):
        self.data_dir = data_dir
        self.transform = SuperResolutionTransform(img_size, scale_factor)

        self.files = os.listdir(data_dir)

        if mode == 'train':
            self.t2_files_path = glob.glob(os.path.join(data_dir, 'MICCAI_BraTS_2019_Data_Training/*/*/*t2.nii.gz'))
            # self.t1_files_path = glob.glob(os.path.join(data_dir, 'MICCAI_BraTS_2019_Data_Training/*/*/*t1.nii'))
            # self.t1ce_files_path = glob.glob(os.path.join(data_dir, 'MICCAI_BraTS_2019_Data_Training/*/*/*t1ce.nii'))
            # self.flair_files_path = glob.glob(os.path.join(data_dir, 'MICCAI_BraTS_2019_Data_Training/*/*/*flair.nii'))
            self.t1_files_path = [t2_fname.replace('t2.nii.gz', 't1.nii.gz') for t2_fname in self.t2_files_path]
            self.t1ce_files_path = [t2_fname.replace('t2.nii.gz', 't1ce.nii.gz') for t2_fname in self.t2_files_path]
            self.flair_files_path = [t2_fname.replace('t2.nii.gz', 'flair.nii.gz') for t2_fname in self.t2_files_path]

        else:
            self.t2_files_path = glob.glob(os.path.join(data_dir, 'MICCAI_BraTS_2019_Data_Validation/*/*t2.nii.gz'))
            # self.t1_files_path = glob.glob(os.path.join(data_dir, 'MICCAI_BraTS_2019_Data_Validation/*/*t1.nii.gz'))
            # self.t1ce_files_path = glob.glob(os.path.join(data_dir, 'MICCAI_BraTS_2019_Data_Validation/*/*t1ce.nii.gz'))
            # self.flair_files_path = glob.glob(os.path.join(data_dir, 'MICCAI_BraTS_2019_Data_Validation/*/*flair.nii.gz'))
            self.t1_files_path = [t2_fname.replace('t2.nii.gz', 't1.nii.gz') for t2_fname in self.t2_files_path]
            self.t1ce_files_path = [t2_fname.replace('t2.nii.gz', 't1ce.nii.gz') for t2_fname in self.t2_files_path]
            self.flair_files_path = [t2_fname.replace('t2.nii.gz', 'flair.nii.gz') for t2_fname in self.t2_files_path]

        slice_id = [60, 65, 70, 75, 80, 85, 90, 95, 100, 105]
        
        self.examples = []

        self.data_dir = data_dir

        for t2_fname, t1_fname, t1ce_fname, flair_fname in zip(self.t2_files_path, self.t1_files_path, self.t1ce_files_path, self.flair_files_path):
            # t2_metadata, t1_metadata, t1ce_metadata, flair_metadata = self._retrieve_metadata(t2_fname)
            for slice_num in slice_id:
                self.examples.append((t2_fname, t1_fname, t1ce_fname, flair_fname, slice_num))

        if sample_rate < 1.0:
            random.shuffle(self.examples)
            num_examples = round(len(self.examples) * sample_rate)

            self.examples = self.examples[0:num_examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        t2_fname, t1_fname, t1ce_fname, flair_fname, slice_num = self.examples[idx]
        
        fname = os.path.basename(os.path.dirname(t2_fname)) + '_slice_{}'.format(slice_num)

        t2_data = sitk.ReadImage(t2_fname)
        t2_data = sitk.GetArrayFromImage(t2_data)

        t2_data = (t2_data-t2_data.min()) / (t2_data.max()-t2_data.min())
        t2_data_255 = (255*t2_data).astype(int)

        t1_data = sitk.ReadImage(t1_fname)
        t1_data = sitk.GetArrayFromImage(t1_data)

        t1_data = (t1_data-t1_data.min()) / (t1_data.max()-t1_data.min())
        t1_data_255 = (255*t1_data).astype(int)

        # t1ce_data = sitk.ReadImage(t1ce_fname)
        # t1ce_data = sitk.GetArrayFromImage(t1ce_data)

        # t1ce_data = (t1ce_data-t1ce_data.min()) / (t1ce_data.max()-t1ce_data.min())
        # t1ce_data_255 = (255*t1ce_data).astype(int)

        # flair_data = sitk.ReadImage(flair_fname)
        # flair_data = sitk.GetArrayFromImage(flair_data)

        # flair_data = (flair_data-flair_data.min()) / (flair_data.max()-flair_data.min())
        # flair_data_255 = (255*flair_data).astype(int)

        t2_data = t2_data[slice_num]
        t1_data = t1_data[slice_num]
        # t1ce_data = t1ce_data[slice_num]
        # flair_data = flair_data[slice_num]

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
    

@register('Brats2019_dataset_Reconstruction')
class BraTs_dataset_Reconstruction(Dataset):
    def __init__(self, data_dir='./BraTs/BraTs-2019', 
                 mode='train', sample_rate=1,
                 img_size=240, mask_type='random_uniform', acceleration_rate=4):
        self.data_dir = data_dir
        # self.transform = transform
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
        
        ##############################################################

        self.files = os.listdir(data_dir)

        if mode == 'train':
            self.t2_files_path = glob.glob(os.path.join(data_dir, 'MICCAI_BraTS_2019_Data_Training/*/*/*t2.nii.gz'))
            # self.t1_files_path = glob.glob(os.path.join(data_dir, 'MICCAI_BraTS_2019_Data_Training/*/*/*t1.nii'))
            # self.t1ce_files_path = glob.glob(os.path.join(data_dir, 'MICCAI_BraTS_2019_Data_Training/*/*/*t1ce.nii'))
            # self.flair_files_path = glob.glob(os.path.join(data_dir, 'MICCAI_BraTS_2019_Data_Training/*/*/*flair.nii'))
            self.t1_files_path = [t2_fname.replace('t2.nii.gz', 't1.nii.gz') for t2_fname in self.t2_files_path]
            self.t1ce_files_path = [t2_fname.replace('t2.nii.gz', 't1ce.nii.gz') for t2_fname in self.t2_files_path]
            self.flair_files_path = [t2_fname.replace('t2.nii.gz', 'flair.nii.gz') for t2_fname in self.t2_files_path]

        else:
            self.t2_files_path = glob.glob(os.path.join(data_dir, 'MICCAI_BraTS_2019_Data_Validation/*/*t2.nii.gz'))
            # self.t1_files_path = glob.glob(os.path.join(data_dir, 'MICCAI_BraTS_2019_Data_Validation/*/*t1.nii.gz'))
            # self.t1ce_files_path = glob.glob(os.path.join(data_dir, 'MICCAI_BraTS_2019_Data_Validation/*/*t1ce.nii.gz'))
            # self.flair_files_path = glob.glob(os.path.join(data_dir, 'MICCAI_BraTS_2019_Data_Validation/*/*flair.nii.gz'))
            self.t1_files_path = [t2_fname.replace('t2.nii.gz', 't1.nii.gz') for t2_fname in self.t2_files_path]
            self.t1ce_files_path = [t2_fname.replace('t2.nii.gz', 't1ce.nii.gz') for t2_fname in self.t2_files_path]
            self.flair_files_path = [t2_fname.replace('t2.nii.gz', 'flair.nii.gz') for t2_fname in self.t2_files_path]

        slice_id = [60, 65, 70, 75, 80, 85, 90, 95, 100, 105]
        
        self.examples = []

        self.data_dir = data_dir

        for t2_fname, t1_fname, t1ce_fname, flair_fname in zip(self.t2_files_path, self.t1_files_path, self.t1ce_files_path, self.flair_files_path):
            # t2_metadata, t1_metadata, t1ce_metadata, flair_metadata = self._retrieve_metadata(t2_fname)
            for slice_num in slice_id:
                self.examples.append((t2_fname, t1_fname, t1ce_fname, flair_fname, slice_num))

        if sample_rate < 1.0:
            random.shuffle(self.examples)
            num_examples = round(len(self.examples) * sample_rate)

            self.examples = self.examples[0:num_examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        t2_fname, t1_fname, t1ce_fname, flair_fname, slice_num = self.examples[idx]
        
        fname = os.path.basename(os.path.dirname(t2_fname)) + '_slice_{}'.format(slice_num)

        t2_data = sitk.ReadImage(t2_fname)
        t2_data = sitk.GetArrayFromImage(t2_data)

        t2_data = (t2_data-t2_data.min()) / (t2_data.max()-t2_data.min())
        t2_data_255 = (255*t2_data).astype(int)

        t1_data = sitk.ReadImage(t1_fname)
        t1_data = sitk.GetArrayFromImage(t1_data)

        t1_data = (t1_data-t1_data.min()) / (t1_data.max()-t1_data.min())
        t1_data_255 = (255*t1_data).astype(int)

        # t1ce_data = sitk.ReadImage(t1ce_fname)
        # t1ce_data = sitk.GetArrayFromImage(t1ce_data)

        # t1ce_data = (t1ce_data-t1ce_data.min()) / (t1ce_data.max()-t1ce_data.min())
        # t1ce_data_255 = (255*t1ce_data).astype(int)

        # flair_data = sitk.ReadImage(flair_fname)
        # flair_data = sitk.GetArrayFromImage(flair_data)

        # flair_data = (flair_data-flair_data.min()) / (flair_data.max()-flair_data.min())
        # flair_data_255 = (255*flair_data).astype(int)

        t2_data = t2_data[slice_num]
        t1_data = t1_data[slice_num]
        # t1ce_data = t1ce_data[slice_num]
        # flair_data = flair_data[slice_num]

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
    


    

