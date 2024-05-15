import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import random

# A custom class for loading volumes and masks
class VolumeMaskDataset(torch.utils.data.Dataset):
    def __init__(self, volume_dir, data_type, n_valid=14):

        self.file_names = [os.path.join(volume_dir, f) for f in os.listdir(volume_dir)]
        All_idx = np.arange(0, len(self.file_names))

        np.random.seed(1377)
        Valid_idx = np.random.choice(All_idx, n_valid, replace=False)
        
        # split data
        if (data_type == 'valid'):
            self.file_names = [self.file_names[i] for i in Valid_idx]
        else:
            Train_idx = np.array([x for x in All_idx if x not in Valid_idx])
            self.file_names = [self.file_names[i] for i  in Train_idx]

        self.volume_path = []
        for name in file_names:
            for i in range(25):
                start = str(i * 32)

                for j in range(2):
                    TMP = name + '#' + start + ',' + str(j)
                    self.volume_path.append(TMP)
        
        # Log
        print(f'Number of patches for {data_type} set : {len(self.volume_path)}')

    def __len__(self):
        return len(self.volume_paths)
    
    def __getitem__(self, idx):
        volume_path = self.volume_paths[idx]
        mask_path   = self.mask_paths[idx]
        
        
        # Load Volume and mask
        volume = np.load(volume_path)
        mask   = np.load(mask_path)
        
        # Convert to Tensor
        volume = torch.from_numpy(volume).float().unsqueeze(0)
        mask   = torch.from_numpy(mask).float().unsqueeze(0)
        
        return volume, mask

def Test_Dataset_Class():
    volume_dir = '/kaggle/input/tdscabus-train-patches/TDSC_Patches/Volumes'
    mask_dir   = '/kaggle/input/tdscabus-train-patches/TDSC_Patches/Mask'
    
    # create dataset
    dataset = VolumeMaskDataset(volume_dir, mask_dir)
    
    print(dataset[10])

def DataLoaderCreator(volume_dir, mask_dir, batch_size, data_type, n_valid=50, shuffle = True):
    # Create dataset
    dataset = VolumeMaskDataset(volume_dir, mask_dir, data_type, n_valid)
    
    # create dataloader
    dataloader = DataLoader(dataset, batch_size, shuffle)
    
    return dataloader