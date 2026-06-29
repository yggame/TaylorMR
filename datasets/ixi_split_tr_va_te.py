import os
import numpy as np

from sklearn.model_selection import KFold
import SimpleITK as sitk
from tqdm import tqdm, trange

def chk_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

# 数据名获取并保存到text文件中
def data_name_save(data_dir, data_file_names, save_path, slice_id):
    """
    data_dir: 数据集目录
    save_path: 保存文件路径
    slice_id: 划分数据集的slice_id
    """

    with open(save_path, 'w') as f:
        for file_name in tqdm(data_file_names):
            full_file_path = os.path.join(data_dir, file_name + '-PD.nii.gz')
            pd_img = sitk.ReadImage(full_file_path)
            pd_img = sitk.GetArrayFromImage(pd_img)
            
            numOfSlice, h, w = pd_img.shape
            if numOfSlice < 120:
                continue
            for ii in slice_id:
                fname = file_name + '-{:03d}'.format(ii)
                f.write(fname + '\n')
                
    return

# 划分数据集
def split_dataset(data_dir, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, text_path=None):
    """
    data_dir: 数据集目录
    """
    # 计算划分的样本数量
    # fname = os.path.splitext(os.path.basename(fl))[0][:-7]
    data_file_name = sorted(os.listdir(data_dir))
    sorted_data_file_name = [os.path.splitext(os.path.basename(fl))[0][:-7] for fl in data_file_name]
    
    total_samples = len(sorted_data_file_name)
    train_size = int(train_ratio * total_samples)
    val_size = int(val_ratio * total_samples)
    test_size = total_samples - train_size - val_size
    
    # 随机打乱数据集
    np.random.seed(0)
    
    indices = list(np.arange(total_samples))
    np.random.shuffle(indices)
    
    # 划分数据集
    train_index = sorted(indices[:train_size])
    val_index = sorted(indices[train_size:train_size+val_size]) # indices[train_size:train_size+val_size]
    test_index = sorted(indices[train_size+val_size:])

        
    # 划分数据集
    train_data = [sorted_data_file_name[index] for index in train_index]
    val_data = [sorted_data_file_name[index] for index in val_index]
    test_data = [sorted_data_file_name[index] for index in test_index]
    
    slice_id = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 119]
    
    data_name_save(data_dir, train_data, os.path.join(text_path, 'IXI_train.txt'), slice_id=slice_id)
    data_name_save(data_dir, val_data, os.path.join(text_path, 'IXI_val.txt'), slice_id=slice_id)
    data_name_save(data_dir, test_data, os.path.join(text_path, 'IXI_test.txt'), slice_id=slice_id)
    data_name_save(data_dir, sorted_data_file_name, os.path.join(text_path, 'IXI_all.txt'), slice_id=slice_id)
    return
    
    
if __name__ == '__main__':
    # 根据PD的
    data_dir = '/hdd/yg_data/data_path/IXI_dataset/IXI-PD'
    
    text_path = '/hdd/yg_data/data_path/IXI_dataset/list_file_IXI_T2_PD'
    chk_path(text_path)
    # data_name_save(data_dir, save_path)
    
    split_dataset(data_dir, text_path=text_path)