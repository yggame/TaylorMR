from typing import Dict, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from .subsample_fastMRI_brain import MaskFunc, create_mask_for_mask_type
from .math_fastMRI_brain import fft2c_new as fft2c
from .math_fastMRI_brain import ifft2c_new as ifft2c
from .math_fastMRI_brain import complex_abs

def ifft2( kspace_cplx):
    return np.absolute(np.fft.ifft2(kspace_cplx))

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

def rss(data, dim=0):
    """
    Compute the Root Sum of Squares (RSS).

    RSS is computed assuming that dim is the coil dimension.

    Args:
        data (torch.Tensor): The input tensor
        dim (int): The dimensions along which to apply the RSS transform

    Returns:
        torch.Tensor: The RSS value.
    """
    return torch.sqrt((data ** 2).sum(dim))

def to_tensor(data: np.ndarray) -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor.

    For complex arrays, the real and imaginary parts are stacked along the last
    dimension.

    Args:
        data: Input numpy array.

    Returns:
        PyTorch version of data.
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)

    return torch.from_numpy(data)

def tensor_to_complex_np(data: torch.Tensor) -> np.ndarray:
    """
    Converts a complex torch tensor to numpy array.

    Args:
        data: Input data to be converted to numpy.

    Returns:
        Complex numpy version of data.
    """
    return torch.view_as_complex(data).numpy()

def apply_mask(
    data: torch.Tensor,
    mask_func: MaskFunc,
    offset: Optional[int] = None,
    seed: Optional[Union[int, Tuple[int, ...]]] = None,
    padding: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Subsample given k-space by multiplying with a mask.

    Args:
        data: The input k-space data. This should have at least 3 dimensions,
            where dimensions -3 and -2 are the spatial dimensions, and the
            final dimension has size 2 (for complex values).
        mask_func: A function that takes a shape (tuple of ints) and a random
            number seed and returns a mask.
        seed: Seed for the random number generator.
        padding: Padding value to apply for mask.

    Returns:
        tuple containing:
            masked data: Subsampled k-space data.
            mask: The generated mask.
            num_low_frequencies: The number of low-resolution frequency samples
                in the mask.
    """
    shape = (1,) * len(data.shape[:-3]) + tuple(data.shape[-3:])
    mask, num_low_frequencies = mask_func(shape, offset, seed)
    if padding is not None:
        mask[..., : padding[0], :] = 0
        mask[..., padding[1] :, :] = 0  # padding value inclusive on right of zeros

    masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros

    return masked_data, mask, num_low_frequencies

def mask_center(x: torch.Tensor, mask_from: int, mask_to: int) -> torch.Tensor:
    """
    Initializes a mask with the center filled in.

    Args:
        mask_from: Part of center to start filling.
        mask_to: Part of center to end filling.

    Returns:
        A mask with the center filled.
    """
    mask = torch.zeros_like(x)
    mask[:, :, :, mask_from:mask_to] = x[:, :, :, mask_from:mask_to]

    return mask

def center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop to the input real image or batch of real images.

    Args:
        data: The input tensor to be center cropped. It should
            have at least 2 dimensions and the cropping is applied along the
            last two dimensions.
        shape: The output shape. The shape should be smaller
            than the corresponding dimensions of data.

    Returns:
        The center cropped image.
    """
    if not (0 < shape[0] <= data.shape[-2] and 0 < shape[1] <= data.shape[-1]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to]


def complex_center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop to the input image or batch of complex images.

    Args:
        data: The complex input tensor to be center cropped. It should have at
            least 3 dimensions and the cropping is applied along dimensions -3
            and -2 and the last dimensions should have a size of 2.
        shape: The output shape. The shape should be smaller than the
            corresponding dimensions of data.

    Returns:
        The center cropped image
    """
    if not (0 < shape[0] <= data.shape[-3] and 0 < shape[1] <= data.shape[-2]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-3] - shape[0]) // 2
    h_from = (data.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to, :]

def center_crop_to_smallest(
    x: torch.Tensor, y: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply a center crop on the larger image to the size of the smaller.

    The minimum is taken over dim=-1 and dim=-2. If x is smaller than y at
    dim=-1 and y is smaller than x at dim=-2, then the returned dimension will
    be a mixture of the two.

    Args:
        x: The first image.
        y: The second image.

    Returns:
        tuple of tensors x and y, each cropped to the minimim size.
    """
    smallest_width = min(x.shape[-1], y.shape[-1])
    smallest_height = min(x.shape[-2], y.shape[-2])
    x = center_crop(x, (smallest_height, smallest_width))
    y = center_crop(y, (smallest_height, smallest_width))

    return x, y


def normalize(
    data: torch.Tensor,
    mean: Union[float, torch.Tensor],
    stddev: Union[float, torch.Tensor],
    eps: Union[float, torch.Tensor] = 0.0,
) -> torch.Tensor:
    """
    Normalize the given tensor.

    Applies the formula (data - mean) / (stddev + eps).

    Args:
        data: Input data to be normalized.
        mean: Mean value.
        stddev: Standard deviation.
        eps: Added to stddev to prevent dividing by zero.

    Returns:
        Normalized tensor.
    """
    # return (data - mean) / (stddev + eps)
    data = (data - mean) / (stddev + eps)
    # 极大极小值归一化
    return (data - data.min()) / (data.max() - data.min())


def normalize_instance(
    data: torch.Tensor, eps: Union[float, torch.Tensor] = 0.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize the given tensor  with instance norm/

    Applies the formula (data - mean) / (stddev + eps), where mean and stddev
    are computed from the data itself.

    Args:
        data: Input data to be normalized
        eps: Added to stddev to prevent dividing by zero.

    Returns:
        torch.Tensor: Normalized tensor
    """
    mean = data.mean()
    std = data.std()

    return normalize(data, mean, std, eps), mean, std

class ReconstructionTransform(object):
    """
       Data Transformer for training U-Net models.
       """

    def __init__(self, mode, which_challenge, MASKTYPE=None, CENTER_FRACTIONS=None, ACCELERATIONS=None, use_seed=True):
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
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')

        self.which_challenge = which_challenge
        self.use_seed = use_seed

        if mode == 'train':
            self.mask_func = create_mask_for_mask_type(
                MASKTYPE, CENTER_FRACTIONS, ACCELERATIONS,
            )
        elif mode == 'val':
            self.mask_func = create_mask_for_mask_type(
                MASKTYPE, CENTER_FRACTIONS, ACCELERATIONS,
            )
        else:
            self.mask_func = create_mask_for_mask_type(
                MASKTYPE, CENTER_FRACTIONS, ACCELERATIONS,
            )

    def __call__(self, target):
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
        kspace = fft2(target)

        # apply mask
        if self.mask_func:
            seed = None if not self.use_seed else 0
            masked_kspace, mask, num_low_frequencies = apply_mask(torch.from_numpy(kspace), self.mask_func, seed=seed)
        else:
            masked_kspace = kspace

        # inverse Fourier transform to get zero filled solution
        image_cplx = np.fft.ifft2(masked_kspace) 

        # crop input to correct size
        # if target is not None:
        # crop_size = (target.shape[-2], target.shape[-1])
        # # else:
        # #     crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        # # check for sFLAIR 203
        # if image.shape[-2] < crop_size[1]:
        #     crop_size = (image.shape[-2], image.shape[-2])

        # image_cplx = complex_center_crop(image, crop_size)
        # print('image',image.shape)
        # absolute value
        image = np.absolute(image_cplx)

        # apply Root-Sum-of-Squares if multicoil data
        if self.which_challenge == "multicoil":
            image = rss(image)

        # normalize input
        image, mean, std = normalize_instance(image, eps=1e-11)
        # image = image.clamp(-6, 6)

        # normalize target
        if target is not None:
            # target = to_tensor(target)
            # target = center_crop(target, crop_size)
            target = normalize(target, mean, std, eps=1e-11)
            target = target.clamp(-6, 6)
        else:
            target = torch.Tensor([0])

        # apply mask
        if self.mask_func:
            return torch.from_numpy(image).float() , target.float(), mean, std, image_cplx, masked_kspace, mask
        return torch.from_numpy(image).float() , target.float(), mean, std, image_cplx, masked_kspace

# def normalize_instance(data, eps=1e-11):
#     mean = data.mean()
#     std = data.std()

#     return normalize(data, mean, std, eps), mean

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
    
def build_transforms(WORK_TYPE = 'sr', mode = 'train', 
                     img_size=None, scale_factor=None, 
                     CHALLENGE=None, MASKTYPE=None, CENTER_FRACTIONS=None, ACCELERATIONS=None,):

    if WORK_TYPE == 'sr':

        if mode == 'train':
            return SuperResolutionTransform(img_size=img_size, scale_factor=scale_factor)
        elif mode == 'val':
            return SuperResolutionTransform(img_size=img_size, scale_factor=scale_factor)
        else:
            return SuperResolutionTransform(img_size=img_size, scale_factor=scale_factor)

    elif WORK_TYPE == 'denoise':
        return DenoiseDataTransform(INPUT_SIZE, NOISE_RATE)

    else:
        if mode == 'train':
            mask_func = create_mask_for_mask_type(
                MASKTYPE, CENTER_FRACTIONS, ACCELERATIONS,
            )
            return ReconstructionTransform(CHALLENGE, mask_func, use_seed=False)
        elif mode == 'val':
            mask_func = create_mask_for_mask_type(
                MASKTYPE, CENTER_FRACTIONS, ACCELERATIONS,
            )
            return ReconstructionTransform(CHALLENGE, mask_func)
        else:
            return ReconstructionTransform(CHALLENGE)