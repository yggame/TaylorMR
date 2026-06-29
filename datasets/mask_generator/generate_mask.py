import torch
import numpy as np
from numpy.lib.stride_tricks import as_strided
import contextlib
import os
import cv2
import scipy.io as scio

def chk_path(cpath):
    if not os.path.exists(cpath):
        os.makedirs(cpath)
    return

###################     https://github.com/lpcccc-cv/MC-CDic/blob/main/data/generate_mask_random.py
###################     https://github.com/Aboriginer/HFS-SDE/blob/master/utils/generate_mask.py

@contextlib.contextmanager
def temp_seed(rng, seed):
    state = rng.get_state()
    rng.seed(seed)
    try:
        yield
    finally:
        rng.set_state(state)
def mask_func_random_unique(image_size, acc = 4, seed=42):
    """
    Args:
    shape:[320, 320, 2]

    Return:
    [1, 320, 1]非0即1的tensor
    """
    # if len(shape) < 3:
    #     raise ValueError("Shape should have 3 or more dimensions")
    
    rng = np.random
    with temp_seed(rng, seed):
        # num_cols = shape[-2]
        num_cols = image_size
        if acc == 4:
            center_fraction, acceleration = 0.08, 4#中心采样比例，加速比
        elif acc == 8:
            center_fraction, acceleration = 0.04, 8#中心采样比例，加速比
        else:
            assert('accelerate rate is not implmented')

        # create the mask
        num_low_freqs = int(round(num_cols * center_fraction))
        #
        #(采样条数-中心采样条数) / 所有未采样条数 计算每一行的采样概率
        #
        
        # prob = (num_cols / acceleration - num_low_freqs) / (
        #     num_cols - num_low_freqs)
        # mask = rng.uniform(size=num_cols) < prob
        # # print(np.sum(mask==True))
        # pad = (num_cols - num_low_freqs + 1) // 2
        # mask[pad: pad + num_low_freqs] = True

        # # reshape the mask
        # mask_shape = [1 for _ in shape]
        # mask_shape[-2] = num_cols
        # mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32)) # mask.shape=[col, 1]
        # mask = mask.repeat(shape[0], 1, 1)   
        
        # mask_final = mask[:, :, 0]
        # mask_datadict = {"mask": np.squeeze(mask_final)}
        center_line_idx = np.arange(
            (num_cols - num_low_freqs) // 2, (num_cols + num_low_freqs) // 2
        )
        outer_line_idx = np.setdiff1d(np.arange(num_cols), center_line_idx)
        np.random.shuffle(outer_line_idx)
        print(sorted(outer_line_idx))
        
        lines_num = int(num_cols / acc) - num_low_freqs
        random_line_idx = outer_line_idx[0:lines_num]
        print(sorted(random_line_idx))

        mask = np.zeros((num_cols))
        mask[center_line_idx] = 1.0
        mask[random_line_idx] = 1.0

        mask = np.repeat(mask[np.newaxis, :], num_cols, axis=0)
        
        mask_final = mask
        mask_datadict = {"mask": np.squeeze(mask)}  
        
        mask_result_filename = str(num_cols)+"random_uniform_acceleration" + str(acceleration) + '_center_fraction' + str(center_fraction)  # + "_acs_lines" + str(acs_lines)
        scio.savemat(os.path.join("mask", mask_result_filename + ".mat"), mask_datadict)
        
        # mask_numpy = torch.from_numpy(mask_final)
        save_mask_folder = "./mask/mask2png/"
        chk_path(save_mask_folder)
        cv2.imwrite(os.path.join(save_mask_folder, mask_result_filename+'.png'), mask_final*255)
    return mask_final

def normal_pdf(length, sensitivity):
    return np.exp(-sensitivity * (np.arange(length) - length / 2) ** 2)

def cartesian_mask(shape, acc, sample_n):
    """
    Sampling density estimated from implementation of kt FOCUSS
    shape: tuple - of form (..., nx, ny)
    acc: float - doesn't have to be integer 4, 8, etc..
    """
    N, Nx, Ny = int(np.prod(shape[:-2])), shape[-2], shape[-1]
    pdf_x = normal_pdf(Nx, 0.5 / (Nx / 10.0) ** 2)
    lmda = Nx / (2.0 * acc)
    n_lines = int(Nx / acc)

    # add uniform distribution
    pdf_x += lmda * 1.0 / Nx

    if sample_n:
        pdf_x[Nx // 2 - sample_n // 2 : Nx // 2 + sample_n // 2] = 0
        pdf_x /= np.sum(pdf_x)
        n_lines -= sample_n

    mask = np.zeros((N, Nx))
    for i in range(N):
        idx = np.random.choice(Nx, n_lines, False, pdf_x)
        mask[i, idx] = 1

    if sample_n:
        mask[:, Nx // 2 - sample_n // 2 : Nx // 2 + sample_n // 2] = 1

    size = mask.itemsize
    mask = as_strided(mask, (N, Nx, Ny), (size * Nx, size, 0))

    mask = mask.reshape(shape)

    return mask

def get_cartesian_mask(acceleration=4, acs_lines=24, image_size=256):
    shape = (1, image_size, image_size)
    mask = cartesian_mask(shape, acceleration, sample_n=acs_lines)
    mask = np.transpose(mask, (0, 2, 1))

    # mask_result_file = os.path.join(
    #     "mask", "cartesian_acc" + str(acceleration) + "_acs" + str(acs_lines) + ".mat"
    # )
    mask_datadict = {"mask": np.squeeze(mask)}
    
    mask_result_filename = str(image_size)+"cartesian_acceleration" + str(acceleration) + '_acs_lines' + str(acs_lines)  # + "_acs_lines" + str(acs_lines)
    scio.savemat(os.path.join("./mask", mask_result_filename + ".mat"), mask_datadict)
    
    # mask_numpy = torch.from_numpy(mask_final)
    save_mask_folder = "./mask/mask2png/"
    chk_path(save_mask_folder)
    cv2.imwrite(os.path.join(save_mask_folder, mask_result_filename+'.png'), np.squeeze(mask)*255)
    
    # scio.savemat(mask_result_file, mask_datadict)
    print("generate cartesian mask, acc =", acceleration)
    
    return np.squeeze(mask)

def get_equispaced_mask(mask_type='low_frequency', acceleration=4, acs_lines=16, image_size=256):
    center_line_idx = np.arange(
        (image_size - acs_lines) // 2, (image_size + acs_lines) // 2
    )
    outer_line_idx = np.setdiff1d(np.arange(image_size), center_line_idx)

    random_line_idx = outer_line_idx[::acceleration]
    print(random_line_idx)

    mask = np.zeros((image_size))
    mask[center_line_idx] = 1.0
    if mask_type == "low_frequency":
        mask[random_line_idx] = 0.0
    else:
        mask[random_line_idx] = 1.0

    mask = np.repeat(mask[np.newaxis, :], image_size, axis=0)
    # if mask_type == "low_frequency":
    #     mask_result_file = os.path.join(
    #         "mask", "low_frequency_acs" + str(acs_lines) + ".mat"
    #     )
    # else:
    #     mask_result_file = os.path.join(
    #         "mask", "uniform_acc" + str(acceleration) + "_acs" + str(acs_lines) + ".mat"
    #     )
    mask_datadict = {"mask": np.squeeze(mask)}
    # scio.savemat(mask_result_file, mask_datadict)
    
    mask_result_filename = str(image_size)+"equispaced" + "_mask_type_" + mask_type + "_acceleration" + str(acceleration) + '_acs_lines' + str(acs_lines)  # + "_acs_lines" + str(acs_lines)
    scio.savemat(os.path.join("./mask", mask_result_filename + ".mat"), mask_datadict)
    
    # mask_numpy = torch.from_numpy(mask_final)
    save_mask_folder = "./mask/mask2png/"
    chk_path(save_mask_folder)
    cv2.imwrite(os.path.join(save_mask_folder, mask_result_filename+'.png'), np.squeeze(mask)*255)

    return np.squeeze(mask)

def get_blur_mask(image_size, length):
    # mask = torch.zeros([1, 1, image_size, image_size], dtype=torch.complex128)
    mask = torch.zeros([1, 1, image_size, image_size])
    x_start = int((image_size - length) / 2)
    x_end = int((image_size + length) / 2)
    mask[:, :, x_start:x_end, x_start:x_end] = 1.0

    # mask_result_file = os.path.join("mask", "center_length" + str(length) + ".mat")
    mask_datadict = {"mask": np.squeeze(mask.numpy())}
    # scio.savemat(mask_result_file, mask_datadict)
    
    mask_result_filename = str(image_size)+"blur" + 'center_length' + str(length)  # + "_acs_lines" + str(acs_lines)
    scio.savemat(os.path.join("./mask", mask_result_filename + ".mat"), mask_datadict)
    
    # mask_numpy = torch.from_numpy(mask_final)
    save_mask_folder = "./mask/mask2png/"
    chk_path(save_mask_folder)
    cv2.imwrite(os.path.join(save_mask_folder, mask_result_filename+'.png'), np.squeeze(mask.numpy())*255)

    return np.squeeze(mask.numpy())


def get_sr_mask(image_size, sr_ratio):
    # mask = torch.zeros([1, 1, image_size, image_size], dtype=torch.complex128)
    # 断言sr_ratio是否为整数
    assert isinstance(sr_ratio, (int)), "sr_ratio must be an integer or float"
    # assert 0 <= sr_ratio <= 1, "sr_ratio must be between 0 and 1"
    length = int(image_size * sr_ratio)
    mask = torch.zeros([1, 1, image_size, image_size])
    x_start = int((image_size - length) / 2)
    x_end = int((image_size + length) / 2)
    mask[:, :, x_start:x_end, x_start:x_end] = 1.0

    # mask_result_file = os.path.join("mask", "center_length" + str(length) + ".mat")
    mask_datadict = {"mask": np.squeeze(mask.numpy())}
    # scio.savemat(mask_result_file, mask_datadict)
    
    mask_result_filename = str(image_size)+"sr_mask" + '_ratio' + str(sr_ratio)  # + "_acs_lines" + str(acs_lines)
    scio.savemat(os.path.join("./mask", mask_result_filename + ".mat"), mask_datadict)
    
    # mask_numpy = torch.from_numpy(mask_final)
    save_mask_folder = "./mask/mask2png/"
    chk_path(save_mask_folder)
    cv2.imwrite(os.path.join(save_mask_folder, mask_result_filename+'.png'), np.squeeze(mask.numpy())*255)

    return np.squeeze(mask.numpy())


# https://github.com/HJ-harry/DiffusionMBIR/blob/main/utils.py#L235
def get_mask(img, size, batch_size, type='gaussian2d', acc_factor=8, center_fraction=0.04, fix=False):
    mux_in = size ** 2
    if type.endswith('2d'):
        Nsamp = mux_in // acc_factor
    elif type.endswith('1d'):
        Nsamp = size // acc_factor
    if type == 'gaussian2d':
        mask = torch.zeros_like(img)
        cov_factor = size * (1.5 / 128)
        mean = [size // 2, size // 2]
        cov = [[size * cov_factor, 0], [0, size * cov_factor]]
        if fix:
          samples = np.random.multivariate_normal(mean, cov, int(Nsamp))
          int_samples = samples.astype(int)
          int_samples = np.clip(int_samples, 0, size - 1)
          mask[..., int_samples[:, 0], int_samples[:, 1]] = 1
        else:
          for i in range(batch_size):
            # sample different masks for batch
            samples = np.random.multivariate_normal(mean, cov, int(Nsamp))
            int_samples = samples.astype(int)
            int_samples = np.clip(int_samples, 0, size - 1)
            mask[i, :, int_samples[:, 0], int_samples[:, 1]] = 1
    elif type == 'uniformrandom2d':
        mask = torch.zeros_like(img)
        if fix:
          mask_vec = torch.zeros([1, size * size])
          samples = np.random.choice(size * size, int(Nsamp))
          mask_vec[:, samples] = 1
          mask_b = mask_vec.view(size, size)
          mask[:, ...] = mask_b
        else:
          for i in range(batch_size):
            # sample different masks for batch
            mask_vec = torch.zeros([1, size * size])
            samples = np.random.choice(size * size, int(Nsamp))
            mask_vec[:, samples] = 1
            mask_b = mask_vec.view(size, size)
            mask[i, ...] = mask_b
    elif type == 'gaussian1d':
        mask = torch.zeros_like(img)
        mean = size // 2
        std = size * (15.0 / 128)
        Nsamp_center = int(size * center_fraction)
        if fix:
          samples = np.random.normal(loc=mean, scale=std, size=int(Nsamp * 1.2))
          int_samples = samples.astype(int)
          int_samples = np.clip(int_samples, 0, size - 1)
          mask[... , int_samples] = 1
          c_from = size // 2 - Nsamp_center // 2
          mask[... , c_from:c_from + Nsamp_center] = 1
        else:
          for i in range(batch_size):
            samples = np.random.normal(loc=mean, scale=std, size=int(Nsamp*1.2))
            int_samples = samples.astype(int)
            int_samples = np.clip(int_samples, 0, size - 1)
            mask[i, :, :, int_samples] = 1
            c_from = size // 2 - Nsamp_center // 2
            mask[i, :, :, c_from:c_from + Nsamp_center] = 1
    elif type == 'uniform1d':
        mask = torch.zeros_like(img)
        if fix:
          Nsamp_center = int(size * center_fraction)
          samples = np.random.choice(size, int(Nsamp - Nsamp_center))
          mask[..., samples] = 1
          # ACS region
          c_from = size // 2 - Nsamp_center // 2
          mask[..., c_from:c_from + Nsamp_center] = 1
        else:
          for i in range(batch_size):
            Nsamp_center = int(size * center_fraction)
            samples = np.random.choice(size, int(Nsamp - Nsamp_center))
            mask[i, :, :, samples] = 1
            # ACS region
            c_from = size // 2 - Nsamp_center // 2
            mask[i, :, :, c_from:c_from+Nsamp_center] = 1
    else:
        NotImplementedError(f'Mask type {type} is currently not supported.')

    mask_datadict = {"mask": np.squeeze(mask.numpy())}
    mask_result_filename = str(size)+"type_" + str(type) + '_acceleration' + str(acc_factor) + '_center_fraction' + str(center_fraction)  # + "_acs_lines" + str(acs_lines)
    scio.savemat(os.path.join("./mask", mask_result_filename + ".mat"), mask_datadict)
    
    # mask_numpy = torch.from_numpy(mask_final)
    save_mask_folder = "./mask/mask2png/"
    chk_path(save_mask_folder)
    cv2.imwrite(os.path.join(save_mask_folder, mask_result_filename+'.png'), np.squeeze(mask.numpy())*255)

    # return np.squeeze(mask.numpy())
    return np.squeeze(mask.numpy())



## https://github.com/MarkQuanHaoyu/SuperResolutionProject/blob/12859c7d6bbdae57622c9ea624032f3ec4c2210a/fastmri_brain.py#L110  感觉不如前面的###
# def uniformly_cartesian_mask(img_size, acceleration_rate, acs_percentage: float = 0.2, randomly_return: bool = False):

#     ny = img_size

#     ACS_START_INDEX = (ny // 2) - (int(ny * acs_percentage * (2 / acceleration_rate)) // 2)
#     ACS_END_INDEX = (ny // 2) + (int(ny * acs_percentage * (2 / acceleration_rate)) // 2)

#     if ny % 2 == 0:
#         ACS_END_INDEX -= 1

#     mask = np.zeros(shape=(acceleration_rate,) + (img_size, img_size), dtype=np.float32)
#     mask[..., ACS_START_INDEX: (ACS_END_INDEX + 1)] = 1

#     for i in range(ny):
#         for j in range(acceleration_rate):
#             if i % acceleration_rate == j:
#                 mask[j, ..., i] = 1

#     if randomly_return:
#         mask = mask[np.random.randint(0, acceleration_rate)]
#     else:
#         mask = mask[0]
    
#     mask_datadict = {"mask": np.squeeze(mask)}
#     mask_result_filename = str(img_size)+"cartesian2_acceleration" + str(acceleration_rate) + '_acs_percentage' + str(acs_percentage)  # + "_acs_lines" + str(acs_lines)
#     scio.savemat(os.path.join("./mask", mask_result_filename + ".mat"), mask_datadict)
    
#     # mask_numpy = torch.from_numpy(mask_final)
#     save_mask_folder = "./mask/mask2png/"
#     chk_path(save_mask_folder)
#     cv2.imwrite(os.path.join(save_mask_folder, mask_result_filename+'.png'), np.squeeze(mask)*255)
    
#     # scio.savemat(mask_result_file, mask_datadict)
#     print("generate cartesian mask, acc =", acceleration_rate)
    
#     return mask

if __name__ == '__main__':
    mask_sample_type = 'random_uniform'
    img_size = 320
    acceleration_factor = 4
    if mask_sample_type == 'random_uniform':
        mask = mask_func_random_unique(image_size = img_size, acc = acceleration_factor, seed=42)
    elif mask_sample_type == 'cartesian':
        mask = get_cartesian_mask(image_size=img_size, acceleration=acceleration_factor, acs_lines=24)
    elif mask_sample_type == 'equispaced':
        mask = get_equispaced_mask(image_size=img_size, mask_type='uniform_frequency', acceleration=4, acs_lines=16)
    elif mask_sample_type == 'blur':
        mask = get_blur_mask(image_size=img_size, length=16)
    # elif mask_sample_type == 'cartesian2':
    #     mask = uniformly_cartesian_mask(img_size=256, acceleration_rate=4, acs_percentage= 0.1, randomly_return= False)
    elif mask_sample_type == 'gaussian2d':
        mask = get_mask(torch.zeros(1, 1, img_size, img_size), size=img_size, batch_size=1, type='gaussian2d', acc_factor=8, center_fraction=0.04, fix=False)
    elif mask_sample_type == 'uniformrandom2d':
        mask = get_mask(torch.zeros(1, 1, img_size, img_size), size=img_size, batch_size=1, type='uniformrandom2d', acc_factor=8, center_fraction=0.04, fix=False)
    elif mask_sample_type == 'gaussian1d':
        mask = get_mask(torch.zeros(1, 1, img_size, img_size), size=img_size, batch_size=1, type='gaussian1d', acc_factor=8, center_fraction=0.04, fix=False)
    elif mask_sample_type == 'uniform1d':
        mask = get_mask(torch.zeros(1, 1, img_size, img_size), size=img_size, batch_size=1, type='uniform1d', acc_factor=8, center_fraction=0.04, fix=False)
    elif mask_sample_type == 'sr':
        mask = get_sr_mask(image_size=img_size, sr_ratio=acceleration_factor)
    print(mask.shape)