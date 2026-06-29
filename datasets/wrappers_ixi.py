import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

from io import BytesIO
from PIL import Image
import numpy as np
import scipy.io as sio

import cv2

from datasets import register

def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, Image.BICUBIC)(
            transforms.ToPILImage()(img)))

def JPEGcompression(image, qf=10):
    # qf = random.randrange(10, 75)
    outputIoStream = BytesIO()
    image.save(outputIoStream, "JPEG", quality=qf, optimice=True)
    outputIoStream.seek(0)
    return Image.open(outputIoStream)

def resize_compress_fn(img, size):
    return transforms.ToTensor()(
            JPEGcompression(
                transforms.Resize(size, Image.BICUBIC)(transforms.ToPILImage()(img))
            )
        )

@register('mr_super_resolution')
class SR_Paired_Dataset(Dataset):
    def __init__(self, dataset, img_size=256, scale=4):
        self.dataset = dataset
        self.img_size = img_size

        self.crop_size = int(img_size / scale)

        # self.mask = mask

        # self.files = os.listdir(self.root)

    def __getitem__(self, index):
        data_t2_pd = self.dataset[index]

        t2_data = data_t2_pd['img_T2']
        pd_data = data_t2_pd['img_PD']
        kspace_T2 = data_t2_pd['kspace_T2']
        kspace_PD = data_t2_pd['kspace_PD']

        slice_full_T2_Kspace, slice_full_img_T2, slice_full_img_T2_clpx, slice_full_img_T2_k, LR_T2_img, LR_T2_img_cplx, LR_T2_k, LR_T2_SR_zero_fill = self.slice_preprocess(
            kspace_T2)      # LR, LR_cplx, LR_mask, LR_ori, LR_ori_cplx, LR_ori_k, kspace, HR, HR_cplx, HR_k (kspace=HR_k)
        slice_full_PD_Kspace, slice_full_img_PD, slice_full_img_PD_clpx, slice_full_img_PD_k, LR_PD_img, LR_PD_img_cplx, LR_PD_k, LR_PD_SR_zero_fill = self.slice_preprocess(
            kspace_PD)      # LR, LR_cplx, LR_mask, LR_ori, LR_ori_cplx, LR_ori_k, kspace, HR, HR_cplx, HR_k
        # kspace, HR, HR_cplx, HR_k, LR, LR_cplx, LR_k

        return {
            'target_Kspace': torch.from_numpy(slice_full_img_T2_k),
            'target_img': torch.from_numpy(slice_full_img_T2).float(),
            'LR_Kspace_target': torch.from_numpy(LR_T2_k), # 低分辨率 T2 K空间
            'LR_img_target': torch.from_numpy(LR_T2_img).float(),  # 欠采样img
            'SR_zero_fill_img_target': torch.from_numpy(LR_T2_SR_zero_fill).float().unsqueeze(0),  # 欠采样img

            'reference_Kspace': torch.from_numpy(slice_full_img_PD_k),
            'reference_img': torch.from_numpy(slice_full_img_PD).float(),
            'LR_Kspace_reference': torch.from_numpy(LR_PD_k), # 欠采样K
            'LR_img_reference': torch.from_numpy(LR_PD_img).float(),  # 欠采样img
            'SR_zero_fill_img_reference': torch.from_numpy(LR_PD_SR_zero_fill).float().unsqueeze(0),  # 欠采样img
            'fname': data_t2_pd['fname']
        }

    def __len__(self):
        return len(self.dataset)
    
    def crop_toshape(self, kspace_cplx):
        if kspace_cplx.shape[0] == self.img_size:
            return kspace_cplx
        if kspace_cplx.shape[0] % 2 == 1:
            kspace_cplx = kspace_cplx[:-1, :-1]
        crop = int((kspace_cplx.shape[0] - self.img_size) / 2)
        kspace_cplx = kspace_cplx[crop:-crop, crop:-crop]
        return kspace_cplx
    

    def slice_preprocess(self, kspace_cplx):  # 256,256
        # crop to fix size
        kspace_cplx = self.crop_toshape(kspace_cplx)  # 256,256
        # split to real and imaginary channels
        kspace = torch.zeros((self.img_size, self.img_size, 2))  # 256,256,2
        kspace[:, :, 0] = torch.real(kspace_cplx)
        kspace[:, :, 1] = torch.imag(kspace_cplx)
        # target image:
        image = self.ifft2(kspace_cplx)  # 256,256===1,256,256
        # HWC to CHW
        kspace = kspace.permute(2, 0, 1)  # 2,256,256
        # print('kspace:',kspace.shape)
        HR, HR_cplx, HR_k = self.getHR(image)
        LR, LR_cplx, LR_k, SR_zero_fill  = self.getLR(image)

        return kspace, HR, HR_cplx, HR_k, LR, LR_cplx, LR_k, SR_zero_fill
    
    def getHR(self, hr_data):

        imgfft = self.fft2(hr_data)

        imgfft = self.center_crop(imgfft, (self.img_size, self.img_size))
        imgifft = np.fft.ifft2(imgfft)
        
        img_out = abs(imgifft)
        
        # imgfft_mask = imgfft * self.mask4hr
        # img_under_sample = np.fft.ifft2(imgfft_mask)
        # img_out_under_sample = abs(img_under_sample)

        return img_out, imgifft, imgfft # , img_out_under_sample, img_under_sample, imgfft_mask

    def getLR(self, hr_data):
        # imgfft = np.fft.fft2(hr_data)
        #
        imgfft = self.fft2(hr_data)

        imgfft = self.center_crop(imgfft, (self.crop_size, self.crop_size))

        LR_ori_k = imgfft

        # imgfft_mask = imgfft * self.mask

        # img_out_cplx = np.fft.ifft2(imgfft_mask)
        # img_out = abs(img_out_cplx)

        LR_ori_cplx = np.fft.ifft2(LR_ori_k)
        LR_ori = abs(LR_ori_cplx)

        # zero pad
        SR_zero_fill_fft = np.pad(imgfft[0, :, :], ((self.img_size - self.crop_size) // 2, (self.img_size - self.crop_size) // 2))
        SR_zero_fill = np.fft.ifft2(SR_zero_fill_fft)
        SR_zero_fill = abs(SR_zero_fill)
        
        # cv2.imwrite('SR_zero_fill.png', SR_zero_fill) # 256,256 ok

        return LR_ori, LR_ori_cplx, LR_ori_k, SR_zero_fill
    
    def ifft2(self, kspace_cplx):
        return np.absolute(np.fft.ifft2(kspace_cplx))[None, :, :]

    def fft2(self, img):
        return np.fft.fftshift(np.fft.fft2(img))
    
    def center_crop(self, data, shape):
        """
        Apply a center crop to the input real image or batch of real images.

        Args:
            data (torch.Tensor): The input tensor to be center cropped. It should have at
                least 2 dimensions and the cropping is applied along the last two dimensions.
            shape (int, int): The output shape. The shape should be smaller than the
                corresponding dimensions of data.

        Returns:
            torch.Tensor: The center cropped image
        """
        # print(data.shape)
        # print(data.shape[-2],data.shape[-1],data.shape[0],data.shape[1])
        assert 0 < shape[0] <= data.shape[-2], 'Error: shape: {}, data.shape: {}'.format(shape, data.shape)  # 556...556
        assert 0 < shape[1] <= data.shape[-1]  # 640...640
        w_from = (data.shape[-2] - shape[0]) // 2
        h_from = (data.shape[-1] - shape[1]) // 2
        w_to = w_from + shape[0]
        h_to = h_from + shape[1]
        return data[..., w_from:w_to, h_from:h_to]
    
@register('mr_reconstruction')
class Rec_Paired_Dataset(Dataset):
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

        # self.files = os.listdir(self.root)

    def __getitem__(self, index):
        data_t2_pd = self.dataset[index]

        t2_data = data_t2_pd['img_T2']
        pd_data = data_t2_pd['img_PD']
        kspace_T2 = data_t2_pd['kspace_T2']
        kspace_PD = data_t2_pd['kspace_PD']

        slice_full_T2_Kspace, slice_full_img_T2, slice_full_img_T2_clpx, slice_full_img_T2_k, U_T2_img, U_T2_img_cplx, U_T2_k, mask_T2 = self.slice_preprocess(
            kspace_T2)      # LR, LR_cplx, LR_mask, LR_ori, LR_ori_cplx, LR_ori_k, kspace, HR, HR_cplx, HR_k (kspace=HR_k)
        slice_full_PD_Kspace, slice_full_img_PD, slice_full_img_PD_clpx, slice_full_img_PD_k, U_PD_img, U_PD_img_cplx, U_PD_k, mask_PD = self.slice_preprocess(
            kspace_PD)      # LR, LR_cplx, LR_mask, LR_ori, LR_ori_cplx, LR_ori_k, kspace, HR, HR_cplx, HR_k

        return {
            'target_Kspace': torch.from_numpy(slice_full_img_T2_k),
            'target_img': torch.from_numpy(slice_full_img_T2).float(),
            'under_sample_Kspace_target': torch.from_numpy(U_T2_k), # 欠采样K
            'under_sample_img_target': torch.from_numpy(U_T2_img).float(),  # 欠采样img

            'reference_Kspace': torch.from_numpy(slice_full_img_PD_k),
            'reference_img': torch.from_numpy(slice_full_img_PD).float(),
            'under_sample_Kspace_reference': torch.from_numpy(U_PD_k), # 欠采样K
            'under_sample_img_reference': torch.from_numpy(U_PD_img).float(),  # 欠采样img

            'mask_target': mask_T2,
            'mask_reference': mask_PD,
            'fname': data_t2_pd['fname']
        }

    def __len__(self):
        return len(self.dataset)

    def slice_preprocess(self, kspace_cplx):  # 256,256
        # crop to fix size
        kspace_cplx = self.crop_toshape(kspace_cplx)  # 256,256
        # split to real and imaginary channels
        kspace = torch.zeros((self.img_size, self.img_size, 2))  # 256,256,2
        kspace[:, :, 0] = torch.real(kspace_cplx)
        kspace[:, :, 1] = torch.imag(kspace_cplx)
        # target image:
        image = self.ifft2(kspace_cplx)  # 256,256===1,256,256
        # HWC to CHW
        kspace = kspace.permute(2, 0, 1)  # 2,256,256
        # print('kspace:',kspace.shape)
        HR, HR_cplx, HR_k, U_img, U_img_cplx, U_k, mask = self.getHR(image)
        # LR, LR_cplx, LR_mask, LR_ori, LR_ori_cplx, LR_ori_k = self.getLR(image)


        return kspace, HR, HR_cplx, HR_k, U_img, U_img_cplx, U_k, mask
    
    def crop_toshape(self, kspace_cplx):
        if kspace_cplx.shape[0] == self.img_size:
            return kspace_cplx
        if kspace_cplx.shape[0] % 2 == 1:
            kspace_cplx = kspace_cplx[:-1, :-1]
        crop = int((kspace_cplx.shape[0] - self.img_size) / 2)
        kspace_cplx = kspace_cplx[crop:-crop, crop:-crop]
        return kspace_cplx
    
    def getHR(self, hr_data):

        imgfft = self.fft2(hr_data)

        imgfft = self.center_crop(imgfft, (self.img_size, self.img_size))
        imgifft = np.fft.ifft2(imgfft)
        
        img_out = abs(imgifft)
        
        imgfft_mask = imgfft * self.mask4hr
        img_under_sample = np.fft.ifft2(imgfft_mask)
        img_out_under_sample = abs(img_under_sample)

        return img_out, imgifft, imgfft, img_out_under_sample, img_under_sample, imgfft_mask, self.mask4hr

    # def getLR(self, hr_data):
    #     # imgfft = np.fft.fft2(hr_data)
    #     #
    #     imgfft = self.fft2(hr_data)

    #     imgfft = self.center_crop(imgfft, (self.crop_size, self.crop_size))

    #     LR_ori_k = imgfft

    #     imgfft_mask = imgfft * self.mask

    #     img_out_cplx = np.fft.ifft2(imgfft_mask)
    #     img_out = abs(img_out_cplx)

    #     LR_ori_cplx = np.fft.ifft2(LR_ori_k)
    #     LR_ori = abs(LR_ori_cplx)

    #     # SR_zero_filled_fft = np.pad(imgfft, (0, 0), 'constant')

    #     return img_out, img_out_cplx, imgfft_mask, LR_ori, LR_ori_cplx, LR_ori_k
    
    def ifft2(self, kspace_cplx):
        return np.absolute(np.fft.ifft2(kspace_cplx))[None, :, :]

    def fft2(self, img):
        return np.fft.fftshift(np.fft.fft2(img))
    
    def center_crop(self, data, shape):
        """
        Apply a center crop to the input real image or batch of real images.

        Args:
            data (torch.Tensor): The input tensor to be center cropped. It should have at
                least 2 dimensions and the cropping is applied along the last two dimensions.
            shape (int, int): The output shape. The shape should be smaller than the
                corresponding dimensions of data.

        Returns:
            torch.Tensor: The center cropped image
        """
        # print(data.shape)
        # print(data.shape[-2],data.shape[-1],data.shape[0],data.shape[1])
        assert 0 < shape[0] <= data.shape[-2], 'Error: shape: {}, data.shape: {}'.format(shape, data.shape)  # 556...556
        assert 0 < shape[1] <= data.shape[-1]  # 640...640
        w_from = (data.shape[-2] - shape[0]) // 2
        h_from = (data.shape[-1] - shape[1]) // 2
        w_to = w_from + shape[0]
        h_to = h_from + shape[1]
        return data[..., w_from:w_to, h_from:h_to]

@register('mr_joint_super_resolution_and_reconstruction')
class SR_Rec_Paired_Dataset(Dataset):
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


    def __getitem__(self, index):
        data_t2_pd = self.dataset[index]

        t2_data = data_t2_pd['img_T2']
        pd_data = data_t2_pd['img_PD']
        kspace_T2 = data_t2_pd['kspace_T2']
        kspace_PD = data_t2_pd['kspace_PD']

        slice_full_T2_Kspace, slice_full_img_T2, slice_full_img_T2_clpx, slice_full_img_T2_k, LR_UnderSample_T2_img, LR_UnderSample_T2_img_cplx, LR_UnderSample_T2_k, LR_UnderSample_T2_SR_zero_fill, mask_T2 = self.slice_preprocess(
            kspace_T2)      # LR, LR_cplx, LR_mask, LR_ori, LR_ori_cplx, LR_ori_k, kspace, HR, HR_cplx, HR_k (kspace=HR_k)
        slice_full_PD_Kspace, slice_full_img_PD, slice_full_img_PD_clpx, slice_full_img_PD_k, LR_UnderSample_PD_img, LR_UnderSample_PD_img_cplx, LR_UnderSample_PD_k, LR_UnderSample_PD_SR_zero_fill, mask_PD = self.slice_preprocess(
            kspace_PD)      # LR, LR_cplx, LR_mask, LR_ori, LR_ori_cplx, LR_ori_k, kspace, HR, HR_cplx, HR_k

        return {
            'target_Kspace': torch.from_numpy(slice_full_img_T2_k),
            'target_img': torch.from_numpy(slice_full_img_T2).float(),
            'LR_UnderSample_Kspace_target': torch.from_numpy(LR_UnderSample_T2_k), # 低分辨率 T2 K空间
            'LR_UnderSample_img_target': torch.from_numpy(LR_UnderSample_T2_img).float(),  # 欠采样img
            'SR_zero_fill_img_target': torch.from_numpy(LR_UnderSample_T2_SR_zero_fill).float().unsqueeze(0),  # 欠采样img

            'reference_Kspace': torch.from_numpy(slice_full_img_PD_k),
            'reference_img': torch.from_numpy(slice_full_img_PD).float(),
            'LR_UnderSample_Kspace_reference': torch.from_numpy(LR_UnderSample_PD_k), # 低分辨率 T2 K空间
            'LR_UnderSample_img_reference': torch.from_numpy(LR_UnderSample_PD_img).float(),  # 欠采样img
            'SR_zero_fill_img_reference': torch.from_numpy(LR_UnderSample_PD_SR_zero_fill).float().unsqueeze(0),  # 欠采样img

            'mask_target': mask_T2,
            'mask_reference': mask_PD,
            'fname': data_t2_pd['fname']
        }

    def __len__(self):
        return len(self.dataset)
    

    def crop_toshape(self, kspace_cplx):
        if kspace_cplx.shape[0] == self.img_size:
            return kspace_cplx
        if kspace_cplx.shape[0] % 2 == 1:
            kspace_cplx = kspace_cplx[:-1, :-1]
        crop = int((kspace_cplx.shape[0] - self.img_size) / 2)
        kspace_cplx = kspace_cplx[crop:-crop, crop:-crop]
        return kspace_cplx


    def slice_preprocess(self, kspace_cplx):  # 256,256
        # crop to fix size
        kspace_cplx = self.crop_toshape(kspace_cplx)  # 256,256
        # split to real and imaginary channels
        kspace = torch.zeros((self.img_size, self.img_size, 2))  # 256,256,2
        kspace[:, :, 0] = torch.real(kspace_cplx)
        kspace[:, :, 1] = torch.imag(kspace_cplx)
        # target image:
        image = self.ifft2(kspace_cplx)  # 256,256===1,256,256
        # HWC to CHW
        kspace = kspace.permute(2, 0, 1)  # 2,256,256
        # print('kspace:',kspace.shape)
        HR, HR_cplx, HR_k = self.getHR(image)
        LR, LR_cplx, LR_k, SR_zero_fill, mask  = self.getLR(image)


        return kspace, HR, HR_cplx, HR_k, LR, LR_cplx, LR_k, SR_zero_fill, mask
    
    def getHR(self, hr_data):

        imgfft = self.fft2(hr_data)

        imgfft = self.center_crop(imgfft, (self.img_size, self.img_size))
        imgifft = np.fft.ifft2(imgfft)
        
        img_out = abs(imgifft)
        
        # imgfft_mask = imgfft * self.mask4hr
        # img_under_sample = np.fft.ifft2(imgfft_mask)
        # img_out_under_sample = abs(img_under_sample)

        return img_out, imgifft, imgfft # , img_out_under_sample, img_under_sample, imgfft_mask

    def getLR(self, hr_data):
        # imgfft = np.fft.fft2(hr_data)
        #
        imgfft = self.fft2(hr_data)

        imgfft = self.center_crop(imgfft, (self.crop_size, self.crop_size))

        LR_ori_k = imgfft

        imgfft_mask = imgfft * self.mask

        img_out_cplx = np.fft.ifft2(imgfft_mask)
        img_out = abs(img_out_cplx)

        # LR_ori_cplx = np.fft.ifft2(LR_ori_k)
        # LR_ori = abs(LR_ori_cplx)


        SR_zero_fill_fft = np.pad(imgfft_mask[0, :, :], ((self.img_size - self.crop_size) // 2, (self.img_size - self.crop_size) // 2))
        SR_zero_fill = np.fft.ifft2(SR_zero_fill_fft)
        SR_zero_fill = abs(SR_zero_fill)
        
        # cv2.imwrite('SR_zero_fill.png', SR_zero_fill) # 256,256 ok

        return img_out, img_out_cplx, imgfft_mask, SR_zero_fill, self.mask
    
    def ifft2(self, kspace_cplx):
        return np.absolute(np.fft.ifft2(kspace_cplx))[None, :, :]

    def fft2(self, img):
        return np.fft.fftshift(np.fft.fft2(img))
    
    def center_crop(self, data, shape):
        """
        Apply a center crop to the input real image or batch of real images.

        Args:
            data (torch.Tensor): The input tensor to be center cropped. It should have at
                least 2 dimensions and the cropping is applied along the last two dimensions.
            shape (int, int): The output shape. The shape should be smaller than the
                corresponding dimensions of data.

        Returns:
            torch.Tensor: The center cropped image
        """
        # print(data.shape)
        # print(data.shape[-2],data.shape[-1],data.shape[0],data.shape[1])
        assert 0 < shape[0] <= data.shape[-2], 'Error: shape: {}, data.shape: {}'.format(shape, data.shape)  # 556...556
        assert 0 < shape[1] <= data.shape[-1]  # 640...640
        w_from = (data.shape[-2] - shape[0]) // 2
        h_from = (data.shape[-1] - shape[1]) // 2
        w_to = w_from + shape[0]
        h_to = h_from + shape[1]
        return data[..., w_from:w_to, h_from:h_to]
