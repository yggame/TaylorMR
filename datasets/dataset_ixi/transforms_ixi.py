from typing import Dict, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

def ifft2( kspace_cplx):
    return np.absolute(np.fft.ifft2(kspace_cplx))[None, :, :]

def fft2(img):
    return np.fft.fftshift(np.fft.fft2(img))

def center_crop(data, shape):
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

# def crop_toshape(kspace_cplx, img_size):
#     if kspace_cplx.shape[0] == img_size:
#         return kspace_cplx
#     if kspace_cplx.shape[0] % 2 == 1:
#         kspace_cplx = kspace_cplx[:-1, :-1]
#     crop = int((kspace_cplx.shape[0] - img_size) / 2)
#     kspace_cplx = kspace_cplx[crop:-crop, crop:-crop]
#     return kspace_cplx


def getLR2UnderSample(hr_data, crop_size, mask):
    img_size = hr_data.shape[-1]

    imgfft = fft2(hr_data)

    imgfft = center_crop(imgfft, (crop_size, crop_size))

    LR_ori_k = imgfft

    imgfft_mask = imgfft * mask

    img_out_cplx = np.fft.ifft2(imgfft_mask)
    img_out = abs(img_out_cplx)

    # LR_ori_cplx = np.fft.ifft2(LR_ori_k)
    # LR_ori = abs(LR_ori_cplx)

    SR_zero_fill_fft = np.pad(imgfft_mask[0, :, :], ((img_size - crop_size) // 2, (img_size - crop_size) // 2))
    SR_zero_fill = np.fft.ifft2(SR_zero_fill_fft)
    SR_zero_fill = abs(SR_zero_fill)
    
    # cv2.imwrite('SR_zero_fill.png', SR_zero_fill) # 256,256 ok

    return img_out, img_out_cplx, imgfft_mask, SR_zero_fill

    
class SuperResolutionTransform(object):
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self, img_size, scale_factor=2.0):
        """
        Args:
            which_challenge (str): Either "singlecoil" or "multicoil" denoting
                the dataset.
            mask_func (fastmri.data.subsample.MaskFunc): A function that can
                create a mask of appropriate shape.
            use_seed (bool): If true, this class computes a pseudo random
                number generator seed from the filename. This ensures that the
                same mask is used for all the slices of a given volume every
                time.
        """
        self.img_size = img_size
        self.scale_factor = scale_factor

        self.crop_size = int(img_size / scale_factor)

    def __call__(self, image):
        """
        Args:
            kspace (numpy.array): Input k-space of shape (num_coils, rows,
                cols, 2) for multi-coil data or (rows, cols, 2) for single coil
                data.
            mask (numpy.array): Mask from the test dataset.
            target (numpy.array): Target image.
            attrs (dict): Acquisition related information stored in the HDF5
                object.
            fname (str): File name.
            slice_num (int): Serial number of the slice.

        Returns:
            (tuple): tuple containing:
                image (torch.Tensor): Zero-filled input image.
                target (torch.Tensor): Target image converted to a torch
                    Tensor.
                mean (float): Mean value used for normalization.
                std (float): Standard deviation value used for normalization.
                fname (str): File name.
                slice_num (int): Serial number of the slice.
        """
        # kspace_cplx = self.crop_toshape(kspace_cplx, self.img_size)

        # kspace = torch.zeros((self.img_size, self.img_size, 2))  # 256,256,2
        # kspace[:, :, 0] = torch.real(kspace_cplx)
        # kspace[:, :, 1] = torch.imag(kspace_cplx)
        # # target image:
        # image = ifft2(kspace_cplx)  # 256,256===1,256,256
        # HWC to CHW
        # kspace = kspace.permute(2, 0, 1)  # 2,256,256
        # print('kspace:',kspace.shape)
        HR, HR_cplx, HR_k = self.getHR(image)

        # crop_size = (self.img_size // self.scale_factor, self.img_size // self.scale_factor)
        LR, LR_cplx, LR_k, SR_zero_fill  = self.getLR(image)

        return HR, HR_cplx, HR_k, LR, LR_cplx, LR_k, SR_zero_fill
    
    def crop_toshape(self, kspace_cplx):
        if kspace_cplx.shape[0] == self.img_size:
            return kspace_cplx
        if kspace_cplx.shape[0] % 2 == 1:
            kspace_cplx = kspace_cplx[:-1, :-1]
        crop = int((kspace_cplx.shape[0] - self.img_size) / 2)
        kspace_cplx = kspace_cplx[crop:-crop, crop:-crop]
        return kspace_cplx
    
    def getHR(self, hr_data):

        imgfft = fft2(hr_data)

        imgfft = center_crop(imgfft, (self.img_size, self.img_size))
        imgifft = np.fft.ifft2(imgfft)
        
        img_out = abs(imgifft)

        return img_out, imgifft, imgfft

    def getLR(self, hr_data):

        imgfft = fft2(hr_data)

        imgfft = center_crop(imgfft, (self.crop_size, self.crop_size))

        LR_ori_k = imgfft

        LR_ori_cplx = np.fft.ifft2(LR_ori_k)
        LR_ori = abs(LR_ori_cplx)

        # zero pad
        SR_zero_fill_fft = np.pad(imgfft[:, :], ((self.img_size - self.crop_size) // 2, (self.img_size - self.crop_size) // 2))
        SR_zero_fill = np.fft.ifft2(SR_zero_fill_fft)
        SR_zero_fill = abs(SR_zero_fill)
        
        # cv2.imwrite('SR_zero_fill.png', SR_zero_fill) # 256,256 ok

        return LR_ori, LR_ori_cplx, LR_ori_k, SR_zero_fill

# class DenoiseDataTransform(object):
#     def __init__(self, size, noise_rate):
#         super(DenoiseDataTransform, self).__init__()
#         self.size = (size, size)
#         self.noise_rate = noise_rate

#     def __call__(self, kspace, mask, target, attrs, fname, slice_num):
#         max_value = attrs["max"]

#         #target
#         target = to_tensor(target)
#         target = center_crop(target, self.size)
#         target, mean, std = normalize_instance(target, eps=1e-11)
#         target = target.clamp(-6, 6)

#         #image
#         kspace = to_tensor(kspace)
#         complex_image = ifft2c(kspace)     #complex_image
#         image = complex_center_crop(complex_image, self.size)
#         noise_image = self.rician_noise(image, max_value)
#         noise_image = complex_abs(noise_image)

#         noise_image = normalize(noise_image, mean, std, eps=1e-11)
#         noise_image = noise_image.clamp(-6, 6)

#         return noise_image, target, mean, std, fname, slice_num


#     def rician_noise(self, X, noise_std):
#         #Add rician noise with variance sampled uniformly from the range 0 and 0.1
#         noise_std = random.uniform(0, noise_std*self.noise_rate)
#         Ir = X + noise_std * torch.randn(X.shape)
#         Ii = noise_std*torch.randn(X.shape)
#         In = torch.sqrt(Ir ** 2 + Ii ** 2)
#         return In

class ReconstructionTransform(object):
    """
       Data Transformer for training U-Net models.
       """

    def __init__(self, img_size, apply_mask=None, mask_func=None, use_seed=True):
        """
        Args:
            which_challenge (str): Either "singlecoil" or "multicoil" denoting
                the dataset.
            mask_func (fastmri.data.subsample.MaskFunc): A function that can
                create a mask of appropriate shape.
            use_seed (bool): If true, this class computes a pseudo random
                number generator seed from the filename. This ensures that the
                same mask is used for all the slices of a given volume every
                time.
        """

        self.img_size = img_size

        # self.apply_mask = apply_mask
        # self.mask_func = mask_func
        # self.use_seed = use_seed

    def __call__(self, image, mask):
       
        # kspace_cplx = self.crop_toshape(kspace_cplx)  # 256,256
        # # split to real and imaginary channels
        # kspace = torch.zeros((self.img_size, self.img_size, 2))  # 256,256,2
        # kspace[:, :, 0] = torch.real(kspace_cplx)
        # kspace[:, :, 1] = torch.imag(kspace_cplx)
        # target image:
        # image = ifft2(kspace_cplx)  # 256,256===1,256,256
        # # HWC to CHW
        # kspace = kspace.permute(2, 0, 1)  # 2,256,256
        # print('kspace:',kspace.shape)

        # if self.apply_mask:
        #     kspace, mask, seed = apply_mask(
        #         kspace,
        #         self.mask_func,
        #         seed=seed,
        #         padding=(0, 0, 0, 0),
        #     )

        HR, HR_cplx, HR_k, U_img, U_img_cplx, U_k = self.getHR(image, mask)
        # LR, LR_cplx, LR_mask, LR_ori, LR_ori_cplx, LR_ori_k = self.getLR(image)


        return HR, HR_cplx, HR_k, U_img, U_img_cplx, U_k

    def crop_toshape(self, kspace_cplx):
        if kspace_cplx.shape[0] == self.img_size:
            return kspace_cplx
        if kspace_cplx.shape[0] % 2 == 1:
            kspace_cplx = kspace_cplx[:-1, :-1]
        crop = int((kspace_cplx.shape[0] - self.img_size) / 2)
        kspace_cplx = kspace_cplx[crop:-crop, crop:-crop]
        return kspace_cplx
    
    # def getHR_wo_mask(self, hr_data):

    #     imgfft = fft2(hr_data)

    #     imgfft = center_crop(imgfft, (self.img_size, self.img_size))
    #     imgifft = np.fft.ifft2(imgfft)
        
    #     img_out = abs(imgifft)
        
    #     imgfft_mask = imgfft * mask
    #     img_under_sample = np.fft.ifft2(imgfft_mask)
    #     img_out_under_sample = abs(img_under_sample)

    #     return img_out, imgifft, imgfft, img_out_under_sample, img_under_sample, imgfft_mask
    
    def getHR(self, hr_data, mask):

        imgfft = fft2(hr_data)

        imgfft = center_crop(imgfft, (self.img_size, self.img_size))
        imgifft = np.fft.ifft2(imgfft)
        
        img_out = abs(imgifft)
        
        imgfft_mask = imgfft * mask
        img_under_sample = np.fft.ifft2(imgfft_mask)
        img_out_under_sample = abs(img_under_sample)

        return img_out, imgifft, imgfft, img_out_under_sample, img_under_sample, imgfft_mask

class JointSrReconstructionTransform(object):
    """
       Data Transformer for training U-Net models.
       """

    def __init__(self, img_size, scale_factor=4.0, mask_func=None, use_seed=True):
        self.img_size = img_size
        self.scale_factor = scale_factor

        self.crop_size = int(img_size / scale_factor)

        # self.mask_func = mask_func
        # self.use_seed = use_seed

    def __call__(self, image, mask):
        # kspace_cplx = self.crop_toshape(kspace_cplx)  # 256,256
        # # split to real and imaginary channels
        # kspace = torch.zeros((self.img_size, self.img_size, 2))  # 256,256,2
        # kspace[:, :, 0] = torch.real(kspace_cplx)
        # kspace[:, :, 1] = torch.imag(kspace_cplx)
        # # target image:
        # image = ifft2(kspace_cplx)  # 256,256===1,256,256
        # # HWC to CHW
        # kspace = kspace.permute(2, 0, 1)  # 2,256,256
        # print('kspace:',kspace.shape)
        HR, HR_cplx, HR_k = self.getHR(image)
        LR, LR_cplx, LR_k, SR_zero_fill  = self.getLR(image, mask)


        return HR, HR_cplx, HR_k, LR, LR_cplx, LR_k, SR_zero_fill
    
    def crop_toshape(self, kspace_cplx):
        if kspace_cplx.shape[0] == self.img_size:
            return kspace_cplx
        if kspace_cplx.shape[0] % 2 == 1:
            kspace_cplx = kspace_cplx[:-1, :-1]
        crop = int((kspace_cplx.shape[0] - self.img_size) / 2)
        kspace_cplx = kspace_cplx[crop:-crop, crop:-crop]
        return kspace_cplx
    
    def getHR(self, hr_data):

        imgfft = fft2(hr_data)

        imgfft = center_crop(imgfft, (self.img_size, self.img_size))
        imgifft = np.fft.ifft2(imgfft)
        
        img_out = abs(imgifft)

        return img_out, imgifft, imgfft 

    def getLR(self, hr_data, mask):
        # imgfft = np.fft.fft2(hr_data)
        #
        imgfft = fft2(hr_data)

        imgfft = center_crop(imgfft, (self.crop_size, self.crop_size))

        LR_ori_k = imgfft

        imgfft_mask = imgfft * mask

        img_out_cplx = np.fft.ifft2(imgfft_mask)
        img_out = abs(img_out_cplx)

        # LR_ori_cplx = np.fft.ifft2(LR_ori_k)
        # LR_ori = abs(LR_ori_cplx)


        SR_zero_fill_fft = np.pad(imgfft_mask[0, :, :], ((self.img_size - self.crop_size) // 2, (self.img_size - self.crop_size) // 2))
        SR_zero_fill = np.fft.ifft2(SR_zero_fill_fft)
        SR_zero_fill = abs(SR_zero_fill)
        
        # cv2.imwrite('SR_zero_fill.png', SR_zero_fill) # 256,256 ok

        return img_out, img_out_cplx, imgfft_mask, SR_zero_fill