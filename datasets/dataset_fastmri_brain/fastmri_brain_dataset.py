import os
import sys
import csv
import cv2
import yaml
import h5py
import random
import pathlib
import scipy.io as sio

import numpy as np
import xml.etree.ElementTree as etree
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

from .transforms_fastMRI_brain import *
from .subsample_fastMRI_brain import create_mask_for_mask_type

import torch
from torch.utils.data import Dataset

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from datasets import register

def center_crop(data, shape):
    if shape[0] <= data.shape[-2]:
        w_from = (data.shape[-2] - shape[0]) // 2
        w_to = w_from + shape[0]
        data = data[..., w_from:w_to, :]
    else:
        w_before = (shape[0] - data.shape[-2]) // 2
        w_after = shape[0] - data.shape[-2] - w_before
        pad = [(0, 0)] * data.ndim
        pad[-2] = (w_before, w_after)
        data = np.pad(data, pad_width=pad, mode='constant', constant_values=0)
    if shape[1] <= data.shape[-1]:
        h_from = (data.shape[-1] - shape[1]) // 2
        h_to = h_from + shape[1]
        data = data[..., :, h_from:h_to]
    else:
        h_before = (shape[1] - data.shape[-1]) // 2
        h_after = shape[1] - data.shape[-1] - h_before
        pad = [(0, 0)] * data.ndim
        pad[-1] = (h_before, h_after)
        data = np.pad(data, pad_width=pad, mode='constant', constant_values=0)
    return data

@register('fastmri_brain_dataset')
class fastMRI_brain_dataset(Dataset):
    def __init__(self, 
                 list_dir='./fastMRI/',
                 data_dir= './fastMRI/',
                 challenge='singlecoil',
                 sample_rate=1,
                 mode='train'):
        
        self.mode = mode

        self.examples = []

        self.cur_path = list_dir
        self.csv_file = os.path.join(self.cur_path, "t1_t2_paired_6875_" + self.mode + ".csv")

        # self.data_dir = os.path.join(data_dir, self.mode)

        # 读取CSV
        with open(self.csv_file, 'r') as f:
            reader = csv.reader(f)

            id = 0

            for row in reader:
                t1_num_slices = self._retrieve_metadata(os.path.join(data_dir, row[0]))

                t2_num_slices = self._retrieve_metadata(os.path.join(data_dir, row[1]))

                for slice_id in range(min(t1_num_slices, t2_num_slices)):
                    self.examples.append(
                        (os.path.join(data_dir, row[0]), os.path.join(data_dir, row[1])
                         , slice_id, id))
                id += 1

        if sample_rate < 1.0:
            random.shuffle(self.examples)
            num_examples = round(len(self.examples) * sample_rate)

            self.examples = self.examples[0:num_examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # 读取pd
        t1_fname, t2_fname, slice, id = self.examples[idx]

        with h5py.File(t1_fname, "r") as hf1:        # hf.keys(): <KeysViewHDF5 ['ismrmrd_header', 'kspace', 'reconstruction_esc', 'reconstruction_rss']> 单线圈用reconstruction_esc， 多线圈用reconstruction_rss
            target_img_t1 = hf1["image"][:][slice]

        with h5py.File(t2_fname, "r") as hf2:
            target_img_t2 = hf2["image"][:][slice]

        target_img_t1 = (target_img_t1-target_img_t1.min()) / (target_img_t1.max()-target_img_t1.min())
        # target_img_t1 = (255*target_img_t1).astype(int)

        target_img_t2 = (target_img_t2-target_img_t2.min()) / (target_img_t2.max()-target_img_t2.min())
        # target_img_t2 = (255*target_img_t2).astype(int)

        target_img_t1, target_img_t2 = map(lambda x: center_crop(x, [self.img_size]*2), (target_img_t1, target_img_t2))

        return {'t1_data': target_img_t1, 
                't2_data': target_img_t2, 
                'id': id}
        # return {
        #     "t1_sample": t1_sample,
        #     "t2_sample": t2_sample,
        #     "id": id
        # }

    def _retrieve_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            num_slices = hf["image"].shape[0]
        return num_slices
    
def fetch_dir(key, data_config_file=pathlib.Path("fastmri_dirs.yaml")):
    """
    Data directory fetcher.

    This is a brute-force simple way to configure data directories for a
    project. Simply overwrite the variables for `knee_path` and `brain_path`
    and this function will retrieve the requested subsplit of the data for use.

    Args:
        key (str): key to retrieve path from data_config_file.
        data_config_file (pathlib.Path,
            default=pathlib.Path("fastmri_dirs.yaml")): Default path config
            file.

    Returns:
        pathlib.Path: The path to the specified directory.
    """
    if not data_config_file.is_file():
        default_config = dict(
            knee_path="/home/jc3/Data/",
            brain_path="/home/jc3/Data/",
        )
        with open(data_config_file, "w") as f:
            yaml.dump(default_config, f)

        raise ValueError(f"Please populate {data_config_file} with directory paths.")

    with open(data_config_file, "r") as f:
        data_dir = yaml.safe_load(f)[key]

    data_dir = pathlib.Path(data_dir)

    if not data_dir.exists():
        raise ValueError(f"Path {data_dir} from {data_config_file} does not exist.")

    return data_dir


def et_query(
        root: etree.Element,
        qlist: Sequence[str],
        namespace: str = "http://www.ismrm.org/ISMRMRD",
) -> str:
    """
    ElementTree query function.
    This can be used to query an xml document via ElementTree. It uses qlist
    for nested queries.
    Args:
        root: Root of the xml to search through.
        qlist: A list of strings for nested searches, e.g. ["Encoding",
            "matrixSize"]
        namespace: Optional; xml namespace to prepend query.
    Returns:
        The retrieved data as a string.
    """
    s = "."
    prefix = "ismrmrd_namespace"

    ns = {prefix: namespace}

    for el in qlist:
        s = s + f"//{prefix}:{el}"

    value = root.find(s, ns)
    if value is None:
        raise RuntimeError("Element not found")

    return str(value.text)


@register('fastmri_brain_dataset_SR')
class fastMRI_brain_dataset_SR(Dataset):
    def __init__(self, 
                 list_dir='./fastMRI/',
                 data_dir= './fastMRI/',
                 challenge='singlecoil',
                 sample_rate=1,
                 mode='train',
                 img_size=320,
                 scale_factor=2.0):
        
        self.mode = mode
        self.img_size = img_size

        # # challenge
        # if challenge not in ("singlecoil", "multicoil"):
        #     raise ValueError('challenge should be either "singlecoil" or "multicoil"')
        # self.recons_key = (
        #     "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"      # 单线圈用reconstruction_esc， 多线圈用reconstruction_rss
        # )

        self.transform = SuperResolutionTransform(img_size=img_size, scale_factor=scale_factor)

        self.examples = []

        self.cur_path = list_dir
        self.csv_file = os.path.join(self.cur_path, "t1_t2_paired_6875_" + self.mode + ".csv")

        # self.data_dir = os.path.join(data_dir, self.mode)

        # 读取CSV
        with open(self.csv_file, 'r') as f:
            reader = csv.reader(f)

            id = 0

            for row in reader:
                t1_num_slices = self._retrieve_metadata(os.path.join(data_dir, row[0]))

                t2_num_slices = self._retrieve_metadata(os.path.join(data_dir, row[1]))

                for slice_id in range(min(t1_num_slices, t2_num_slices)):
                    self.examples.append(
                        (os.path.join(data_dir, row[0]), os.path.join(data_dir, row[1])
                         , slice_id, id))
                id += 1

        if sample_rate < 1.0:
            random.shuffle(self.examples)
            num_examples = round(len(self.examples) * sample_rate)

            self.examples = self.examples[0:num_examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # 读取pd
        t1_fname, t2_fname, slice, id = self.examples[idx]

        with h5py.File(t1_fname, "r") as hf1:        # hf.keys(): <KeysViewHDF5 ['ismrmrd_header', 'kspace', 'reconstruction_esc', 'reconstruction_rss']> 单线圈用reconstruction_esc， 多线圈用reconstruction_rss
            target_img_t1 = hf1["image"][:][slice]

        with h5py.File(t2_fname, "r") as hf2:
            target_img_t2 = hf2["image"][:][slice]

        target_img_t1 = (target_img_t1-target_img_t1.min()) / (target_img_t1.max()-target_img_t1.min())
        # target_img_t1 = (255*target_img_t1).astype(int)

        target_img_t2 = (target_img_t2-target_img_t2.min()) / (target_img_t2.max()-target_img_t2.min())
        # target_img_t2 = (255*target_img_t2).astype(int)

        target_img_t1, target_img_t2 = map(lambda x: center_crop(x, [self.img_size]*2), (target_img_t1, target_img_t2))
        

        t1_sample = self.transform(target_img_t1)       # HR, HR_cplx, HR_k, LR, LR_cplx, LR_k, SR_zero_fill
        t2_sample = self.transform(target_img_t2)   # HR, HR_cplx, HR_k, LR, LR_cplx, LR_k, SR_zero_fill

        lr_sx = cv2.Sobel(t1_sample[3], cv2.CV_64F, 1, 0)
        lr_sy  = cv2.Sobel(t1_sample[3], cv2.CV_64F, 0, 1)
        lr_gradient = cv2.addWeighted(lr_sx, 0.5, lr_sy, 0.5, 0)
        # return {
        #     "target_sample": t1_sample,
        #     "reference_sample": t2_sample,
        #     "id": id
        # }
        return {
            "target_img": torch.from_numpy(t1_sample[0]).float(),
            'target_Kspace': torch.from_numpy(t1_sample[2]).unsqueeze(0),
            'LR_img_target': torch.from_numpy(t1_sample[3]).float(),
            'LR_Kspace_target': torch.from_numpy(t1_sample[5]).unsqueeze(0), #
            'SR_zero_pad_image_target': torch.from_numpy(t1_sample[6]).float(),

            'LR_img_target_gradient': torch.from_numpy(lr_gradient).float(),

            "reference_img": torch.from_numpy(t2_sample[0]).float(),
            'reference_Kspace': torch.from_numpy(t2_sample[2]).unsqueeze(0),
            'LR_img_reference': torch.from_numpy(t2_sample[3]).float(),
            'LR_Kspace_reference': torch.from_numpy(t2_sample[5]).unsqueeze(0), #
            'SR_zero_pad_image_reference': torch.from_numpy(t2_sample[6]).float(),

            # 'target_fname': os.path.basename(t1_sample[5]).split('.')[0] + '_'+ str(t1_sample[6]),
            # 'reference_fname': os.path.basename(t2_sample[5]).split('.')[0] + '_' + str(t2_sample[6]),
            "id": id
        }

    def _retrieve_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            num_slices = hf["image"].shape[0]
        return num_slices
    

@register('fastmri_brain_dataset_Reconstruction')
class fastMRI_brain_dataset_Reconstruction(Dataset):
    def __init__(self, 
                 list_dir='./fastMRI/',
                 data_dir= './fastMRI/',
                 challenge='singlecoil',
                 sample_rate=1,
                 mode='train',
                 MASKTYPE='random', 
                 CENTER_FRACTIONS=0.08, 
                 ACCELERATIONS=4, 
                 img_size=320,
                 use_seed=True):
        
        self.mode = mode
        self.img_size = img_size

        # # challenge
        # if challenge not in ("singlecoil", "multicoil"):
        #     raise ValueError('challenge should be either "singlecoil" or "multicoil"')
        # self.recons_key = (
        #     "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"      # 单线圈用reconstruction_esc， 多线圈用reconstruction_rss
        # )

        # if mode == "train": # # knee中 train中没有，官网测试集中有，而brain都没有，因此train 和val 需要mask.

        # transform
        self.transform = ReconstructionTransform(mode, challenge, MASKTYPE, CENTER_FRACTIONS, ACCELERATIONS, use_seed)
        
        # else:
        #     self.transform = ReconstructionTransform(challenge)

        self.examples = []

        self.cur_path = list_dir
        self.csv_file = os.path.join(self.cur_path, "t1_t2_paired_6875_" + self.mode + ".csv")

        # self.data_dir = os.path.join(data_dir, self.mode)

        # 读取CSV
        with open(self.csv_file, 'r') as f:
            reader = csv.reader(f)

            id = 0

            for row in reader:
                t1_num_slices = self._retrieve_metadata(os.path.join(data_dir, row[0]))

                t2_num_slices = self._retrieve_metadata(os.path.join(data_dir, row[1]))

                for slice_id in range(min(t1_num_slices, t2_num_slices)):
                    self.examples.append(
                        (os.path.join(data_dir, row[0]), os.path.join(data_dir, row[1])
                         , slice_id, id))
                id += 1

        if sample_rate < 1.0:
            random.shuffle(self.examples)
            num_examples = round(len(self.examples) * sample_rate)

            self.examples = self.examples[0:num_examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # 读取pd
        t1_fname, t2_fname, slice, id = self.examples[idx]

        with h5py.File(t1_fname, "r") as hf1:        # hf.keys(): <KeysViewHDF5 ['ismrmrd_header', 'kspace', 'reconstruction_esc', 'reconstruction_rss']> 单线圈用reconstruction_esc， 多线圈用reconstruction_rss
            target_img_t1 = hf1["image"][:][slice]

        with h5py.File(t2_fname, "r") as hf2:
            target_img_t2 = hf2["image"][:][slice]

        target_img_t1 = (target_img_t1-target_img_t1.min()) / (target_img_t1.max()-target_img_t1.min())
        # target_img_t1 = (255*target_img_t1).astype(int)

        target_img_t2 = (target_img_t2-target_img_t2.min()) / (target_img_t2.max()-target_img_t2.min())
        # target_img_t2 = (255*target_img_t2).astype(int)

        target_img_t1, target_img_t2 = map(lambda x: center_crop(x, [self.img_size]*2), (target_img_t1, target_img_t2))
        # else:
        #     t2_sample = self.transform(pdfs_kspace, pdfs_mask, pdfs_target, attrs, t2_fname, slice)

        # vis_data(t2_sample[0], pdfs_target[0], t1_fname, t2_fname, slice, 'vis_noise')

        t1_sample = self.transform(torch.from_numpy(target_img_t1).unsqueeze_(0))       # image, target, mean, std, image_cplx
        t2_sample = self.transform(torch.from_numpy(target_img_t2).unsqueeze_(0))   # image, target, mean, std, image_cplx

        # return {
        #     "target_sample": t1_sample,
        #     "reference_sample": t2_sample,
        #     "id": id
        # }
        target_fname = os.path.basename(os.path.dirname(t1_fname)) + f'_silce_{slice}'
        reference_fname = os.path.basename(os.path.dirname(t2_fname)) + f'_silce_{slice}'
        return {
            "target_img": t1_sample[1],
            'under_sample_img_target': t1_sample[0],
            'target_mean': t1_sample[2],
            'target_std': t1_sample[3],
            'target_fname': target_fname,
            # 'under_sample_img_target_clpx': t1_sample[6],
            'under_sample_img_target_clpx': t1_sample[4].squeeze(0),
            'under_sample_Kspace_target': t1_sample[5],

            "reference_img": t2_sample[1],
            "under_sample_img_reference": t2_sample[0],
            "reference_mean": t2_sample[2],
            "reference_std": t2_sample[3],
            "reference_fname": reference_fname,
            "under_sample_img_reference_clpx": t2_sample[4].squeeze(0),
            'under_sample_Kspace_reference': t2_sample[5],

            "id": id
        }

    def _retrieve_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            num_slices = hf["image"].shape[0]
        return num_slices
    

@register('fastmri_brain_dataset_SR2')
class fastMRI_brain_dataset_SR2(Dataset):
    def __init__(self, 
                 list_dir='./fastMRI/',
                 data_dir= './fastMRI/',
                 challenge='singlecoil',
                 sample_rate=1,
                 mode='train',
                 img_size=320,
                 scale_factor=2.0):
        
        self.mode = mode
        self.img_size = img_size

        # # challenge
        # if challenge not in ("singlecoil", "multicoil"):
        #     raise ValueError('challenge should be either "singlecoil" or "multicoil"')
        # self.recons_key = (
        #     "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"      # 单线圈用reconstruction_esc， 多线圈用reconstruction_rss
        # )

        self.transform = SuperResolutionTransform(img_size=img_size, scale_factor=scale_factor)

        self.examples = []

        self.cur_path = list_dir
        self.csv_file = os.path.join(self.cur_path, "t1_t2_paired_6875_" + self.mode + ".csv")

        # self.data_dir = os.path.join(data_dir, self.mode)

        # 读取CSV
        with open(self.csv_file, 'r') as f:
            reader = csv.reader(f)

            id = 0

            for row in reader:
                t1_num_slices = self._retrieve_metadata(os.path.join(data_dir, row[0]))

                t2_num_slices = self._retrieve_metadata(os.path.join(data_dir, row[1]))

                for slice_id in range(min(t1_num_slices, t2_num_slices)):
                    self.examples.append(
                        (os.path.join(data_dir, row[0]), os.path.join(data_dir, row[1])
                         , slice_id, id))
                id += 1

        if sample_rate < 1.0:
            random.shuffle(self.examples)
            num_examples = round(len(self.examples) * sample_rate)

            self.examples = self.examples[0:num_examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # 读取pd
        t1_fname, t2_fname, slice, id = self.examples[idx]

        with h5py.File(t1_fname, "r") as hf1:        # hf.keys(): <KeysViewHDF5 ['ismrmrd_header', 'kspace', 'reconstruction_esc', 'reconstruction_rss']> 单线圈用reconstruction_esc， 多线圈用reconstruction_rss
            target_img_t1 = hf1["image"][:][slice]

        with h5py.File(t2_fname, "r") as hf2:
            target_img_t2 = hf2["image"][:][slice]

        target_img_t1 = (target_img_t1-target_img_t1.min()) / (target_img_t1.max()-target_img_t1.min())
        # target_img_t1 = (255*target_img_t1).astype(int)

        target_img_t2 = (target_img_t2-target_img_t2.min()) / (target_img_t2.max()-target_img_t2.min())
        # target_img_t2 = (255*target_img_t2).astype(int)

        target_img_t1, target_img_t2 = map(lambda x: center_crop(x, [self.img_size]*2), (target_img_t1, target_img_t2))
        

        t1_sample = self.transform(target_img_t1)       # HR, HR_cplx, HR_k, LR, LR_cplx, LR_k, SR_zero_fill
        t2_sample = self.transform(target_img_t2)   # HR, HR_cplx, HR_k, LR, LR_cplx, LR_k, SR_zero_fill

        lr_sx = cv2.Sobel(t2_sample[3], cv2.CV_64F, 1, 0)
        lr_sy  = cv2.Sobel(t2_sample[3], cv2.CV_64F, 0, 1)
        lr_gradient = cv2.addWeighted(lr_sx, 0.5, lr_sy, 0.5, 0)
        # return {
        #     "target_sample": t1_sample,
        #     "reference_sample": t2_sample,
        #     "id": id
        # }
        target_fname = os.path.basename(os.path.dirname(t2_fname)) + f'_silce_{slice}'
        reference_fname = os.path.basename(os.path.dirname(t1_fname)) + f'_silce_{slice}'
        return {
            "target_img": torch.from_numpy(t2_sample[0]).float(),
            'target_Kspace': torch.from_numpy(t2_sample[2]).unsqueeze(0),
            'LR_img_target': torch.from_numpy(t2_sample[3]).float(),
            'LR_Kspace_target': torch.from_numpy(t2_sample[5]).unsqueeze(0), #
            'SR_zero_pad_image_target': torch.from_numpy(t2_sample[6]).float(),

            'LR_img_target_gradient': torch.from_numpy(lr_gradient).float(),

            "reference_img": torch.from_numpy(t1_sample[0]).float(),
            'reference_Kspace': torch.from_numpy(t1_sample[2]).unsqueeze(0),
            'LR_img_reference': torch.from_numpy(t1_sample[3]).float(),
            'LR_Kspace_reference': torch.from_numpy(t1_sample[5]).unsqueeze(0), #
            'SR_zero_pad_image_reference': torch.from_numpy(t1_sample[6]).float(),

            # 'target_fname': os.path.basename(t1_sample[5]).split('.')[0] + '_'+ str(t1_sample[6]),
            # 'reference_fname': os.path.basename(t2_sample[5]).split('.')[0] + '_' + str(t2_sample[6]),
            "id": id
        }

    def _retrieve_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            num_slices = hf["image"].shape[0]
        return num_slices
    

@register('fastmri_brain_dataset_Reconstruction2')
class fastMRI_brain_dataset_Reconstruction2(Dataset):
    def __init__(self, 
                 list_dir='./fastMRI/',
                 data_dir= './fastMRI/',
                 challenge='singlecoil',
                 sample_rate=1,
                 mode='train',
                 MASKTYPE='random', 
                 CENTER_FRACTIONS=0.08, 
                 ACCELERATIONS=4, 
                 img_size=320,
                 use_seed=True):
        
        self.mode = mode
        self.img_size = img_size

        # # challenge
        # if challenge not in ("singlecoil", "multicoil"):
        #     raise ValueError('challenge should be either "singlecoil" or "multicoil"')
        # self.recons_key = (
        #     "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"      # 单线圈用reconstruction_esc， 多线圈用reconstruction_rss
        # )

        # if mode == "train": # # knee中 train中没有，官网测试集中有，而brain都没有，因此train 和val 需要mask.

        # transform
        self.transform = ReconstructionTransform(mode, challenge, MASKTYPE, CENTER_FRACTIONS, ACCELERATIONS, use_seed)
        
        # else:
        #     self.transform = ReconstructionTransform(challenge)

        self.examples = []

        self.cur_path = list_dir
        self.csv_file = os.path.join(self.cur_path, "t1_t2_paired_6875_" + self.mode + ".csv")

        # self.data_dir = os.path.join(data_dir, self.mode)

        # 读取CSV
        with open(self.csv_file, 'r') as f:
            reader = csv.reader(f)

            id = 0

            for row in reader:
                t1_num_slices = self._retrieve_metadata(os.path.join(data_dir, row[0]))

                t2_num_slices = self._retrieve_metadata(os.path.join(data_dir, row[1]))

                for slice_id in range(min(t1_num_slices, t2_num_slices)):
                    self.examples.append(
                        (os.path.join(data_dir, row[0]), os.path.join(data_dir, row[1])
                         , slice_id, id))
                id += 1

        if sample_rate < 1.0:
            random.shuffle(self.examples)
            num_examples = round(len(self.examples) * sample_rate)

            self.examples = self.examples[0:num_examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # 读取pd
        t1_fname, t2_fname, slice, id = self.examples[idx]

        with h5py.File(t1_fname, "r") as hf1:        # hf.keys(): <KeysViewHDF5 ['ismrmrd_header', 'kspace', 'reconstruction_esc', 'reconstruction_rss']> 单线圈用reconstruction_esc， 多线圈用reconstruction_rss
            target_img_t1 = hf1["image"][:][slice]

        with h5py.File(t2_fname, "r") as hf2:
            target_img_t2 = hf2["image"][:][slice]

        target_img_t1 = (target_img_t1-target_img_t1.min()) / (target_img_t1.max()-target_img_t1.min())
        # target_img_t1 = (255*target_img_t1).astype(int)

        target_img_t2 = (target_img_t2-target_img_t2.min()) / (target_img_t2.max()-target_img_t2.min())
        # target_img_t2 = (255*target_img_t2).astype(int)

        target_img_t1, target_img_t2 = map(lambda x: center_crop(x, [self.img_size]*2), (target_img_t1, target_img_t2))
        # else:
        #     t2_sample = self.transform(pdfs_kspace, pdfs_mask, pdfs_target, attrs, t2_fname, slice)

        # vis_data(t2_sample[0], pdfs_target[0], t1_fname, t2_fname, slice, 'vis_noise')

        t1_sample = self.transform(torch.from_numpy(target_img_t1).unsqueeze_(0))       # image, target, mean, std, image_cplx
        t2_sample = self.transform(torch.from_numpy(target_img_t2).unsqueeze_(0))   # image, target, mean, std, image_cplx

        # return {
        #     "target_sample": t1_sample,
        #     "reference_sample": t2_sample,
        #     "id": id
        # }
        target_fname = os.path.basename(os.path.dirname(t2_fname)) + f'_silce_{slice}'
        reference_fname = os.path.basename(os.path.dirname(t1_fname)) + f'_silce_{slice}'
        return {
            "target_img": t2_sample[1],
            'under_sample_img_target': t2_sample[0],
            'target_mean': t2_sample[2],
            'target_std': t2_sample[3],
            'target_fname': target_fname,
            # 'under_sample_img_target_clpx': t1_sample[6],
            'under_sample_img_target_clpx': t2_sample[4].squeeze(0),
            'under_sample_Kspace_target': t2_sample[5], 

            "reference_img": t1_sample[1],
            "under_sample_img_reference": t1_sample[0],
            "reference_mean": t1_sample[2],
            "reference_std": t1_sample[3],
            "reference_fname": reference_fname,
            "under_sample_img_reference_clpx": t1_sample[4].squeeze(0),
            'under_sample_Kspace_reference': t1_sample[5],
            
            'mask_target':  t2_sample[6].squeeze(0),
            'mask_reference': t1_sample[6].squeeze(0),

            "id": id
        }

    def _retrieve_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            num_slices = hf["image"].shape[0]
        return num_slices
    

@register('fastmri_brain_dataset_Reconstruction_VaildMask')
class fastMRI_brain_dataset_Reconstruction_VaildMask(Dataset):
    def __init__(self, 
                 list_dir='./fastMRI/',
                 data_dir= './fastMRI/singlecoil_train',
                 challenge='singlecoil',
                 sample_rate=1,
                 mode='train',
                #  MASKTYPE='random', 
                #  CENTER_FRACTIONS=0.08, 
                #  ACCELERATIONS=4, 
                #  use_seed=True,
                 img_size=240, 
                 mask_type='random_uniform', 
                 acceleration_rate=4):
        
        self.mode = mode

        # challenge
        if challenge not in ("singlecoil", "multicoil"):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')
        self.recons_key = (
            "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"      # 单线圈用reconstruction_esc， 多线圈用reconstruction_rss
        )

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

        # # if mode == "train":
        # mask_func = create_mask_for_mask_type(
        #         MASKTYPE, CENTER_FRACTIONS, ACCELERATIONS,
        #     )
        self.transform = ReconstructionTransform_Valid(challenge)
        
        # else:
        #     self.transform = ReconstructionTransform(challenge)

        self.examples = []

        self.cur_path = list_dir
        self.csv_file = os.path.join(self.cur_path, "singlecoil_" + self.mode + "_split_less.csv")

        # self.data_dir = os.path.join(data_dir, self.mode)

        # 读取CSV
        with open(self.csv_file, 'r') as f:
            reader = csv.reader(f)

            id = 0

            for row in reader:
                pd_metadata, pd_num_slices = self._retrieve_metadata(os.path.join(data_dir, row[0] + '.h5'))

                pdfs_metadata, pdfs_num_slices = self._retrieve_metadata(os.path.join(data_dir, row[1] + '.h5'))

                for slice_id in range(min(pd_num_slices, pdfs_num_slices)):
                    self.examples.append(
                        (os.path.join(data_dir, row[0] + '.h5'), os.path.join(data_dir, row[1] + '.h5')
                         , slice_id, pd_metadata, pdfs_metadata, id))
                id += 1

        if sample_rate < 1.0:
            random.shuffle(self.examples)
            num_examples = round(len(self.examples) * sample_rate)

            self.examples = self.examples[0:num_examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # 读取pd
        t1_fname, t2_fname, slice, pd_metadata, pdfs_metadata, id = self.examples[idx]
        fname = os.path.basename(t1_fname).split('.')[0] + '_slice_{}'.format(id)

        with h5py.File(t1_fname, "r") as hf:        # hf.keys(): <KeysViewHDF5 ['ismrmrd_header', 'kspace', 'reconstruction_esc', 'reconstruction_rss']> 单线圈用reconstruction_esc， 多线圈用reconstruction_rss
            pd_kspace = hf["kspace"][slice]

            pd_mask = np.asarray(hf["mask"]) if "mask" in hf else self.mask4hr      # train中没有，官网测试集中有

            pd_target = hf[self.recons_key][slice] if self.recons_key in hf else None

            attrs = dict(hf.attrs)

            attrs.update(pd_metadata)

        # if self.transform is None:
        t1_sample = (pd_kspace, pd_mask, pd_target, attrs, t1_fname, slice)
        # else:
        #     t1_sample = self.transform(pd_kspace, pd_mask, pd_target, attrs, t1_fname, slice)

        with h5py.File(t2_fname, "r") as hf:
            pdfs_kspace = hf["kspace"][slice]
            pdfs_mask = np.asarray(hf["mask"]) if "mask" in hf else self.mask4hr

            pdfs_target = hf[self.recons_key][slice] if self.recons_key in hf else None

            attrs = dict(hf.attrs)

            attrs.update(pdfs_metadata)

        # if self.transform is None:
        t2_sample = (pdfs_kspace, pdfs_mask, pdfs_target, attrs, t2_fname, slice)
        # else:
        #     t2_sample = self.transform(pdfs_kspace, pdfs_mask, pdfs_target, attrs, t2_fname, slice)

        # vis_data(t2_sample[0], pdfs_target[0], t1_fname, t2_fname, slice, 'vis_noise')

        t1_sample = self.transform(*t1_sample)       # image, target, mean, std, fname, slice_num, image_cplx, mask
        t2_sample = self.transform(*t2_sample)   # image, target, mean, std, fname, slice_num, image_cplx,  mask

        # return {
        #     "target_sample": t1_sample,
        #     "reference_sample": t2_sample,
        #     "id": id
        # }
        return {
            "target_img": t1_sample[1].float(),
            'under_sample_img_target': torch.from_numpy(t1_sample[0]).float(),
            'target_mean': t1_sample[2],
            'target_std': t1_sample[3],
            'target_fname': t1_sample[4],
            'target_slice_num': t1_sample[5],
            'under_sample_img_target_clpx': t1_sample[6],
            'under_sample_Kspace_target': t1_sample[7],

            "reference_img": t2_sample[1].float(),
            "under_sample_img_reference": torch.from_numpy(t2_sample[0]).float(),
            "reference_mean": t2_sample[2],
            "reference_std": t2_sample[3],
            "reference_fname": t2_sample[4],
            "reference_slice_num": t2_sample[5],
            "under_sample_img_reference_clpx": t2_sample[6],
            'under_sample_Kspace_reference': t1_sample[7],

            'mask_target':  t1_sample[8],
            'mask_reference': t2_sample[8],

            "id": id,
            "fname": fname
        }

    def _retrieve_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            et_root = etree.fromstring(hf["ismrmrd_header"][()])

            enc = ["encoding", "encodedSpace", "matrixSize"]
            enc_size = (
                int(et_query(et_root, enc + ["x"])),
                int(et_query(et_root, enc + ["y"])),
                int(et_query(et_root, enc + ["z"])),
            )
            rec = ["encoding", "reconSpace", "matrixSize"]
            recon_size = (
                int(et_query(et_root, rec + ["x"])),
                int(et_query(et_root, rec + ["y"])),
                int(et_query(et_root, rec + ["z"])),
            )

            lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
            enc_limits_center = int(et_query(et_root, lims + ["center"]))
            enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1

            padding_left = enc_size[1] // 2 - enc_limits_center
            padding_right = padding_left + enc_limits_max

            num_slices = hf["kspace"].shape[0]

        metadata = {
            "padding_left": padding_left,
            "padding_right": padding_right,
            "encoding_size": enc_size,
            "recon_size": recon_size,
        }

        return metadata, num_slices

if __name__ == "__main__":
    fastDataset = fastMRI_brain_dataset()
    print(fastDataset[0])

    print(fastDataset[0][0].shape)