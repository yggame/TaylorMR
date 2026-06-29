import torch
import numpy as np
import contextlib
import os
import cv2
import scipy.io as scio

def chk_path(cpath):
    if not os.path.exists(cpath):
        os.makedirs(cpath)
    return

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

  return mask

@contextlib.contextmanager
def temp_seed(rng, seed):
    state = rng.get_state()
    rng.seed(seed)
    try:
        yield
    finally:
        rng.set_state(state)
def mask_func_random_unique(shape, acc = 4, seed=42):
    """
    Args:
    shape:[320, 320, 2]

    Return:
    [1, 320, 1]非0即1的tensor
    """
    if len(shape) < 3:
        raise ValueError("Shape should have 3 or more dimensions")
    
    rng = np.random
    with temp_seed(rng, seed):
        num_cols = shape[-2]
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
        print(outer_line_idx)
        
        lines_num = int(num_cols / acc) - num_low_freqs
        random_line_idx = outer_line_idx[0:lines_num]
        print(random_line_idx)

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

mask = mask_func_random_unique([128, 128, 2])
print(mask.shape)
# cv2.imwrite('./mask_x8_brain.png', mask.numpy()*255)