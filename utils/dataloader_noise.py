import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import random

# A custom class for loading volumes and masks
class VolumeMaskDataset(torch.utils.data.Dataset):
    def __init__(self, volume_dir, mask_dir, data_type, n_valid=50):

        self.file_names = os.listdir(mask_dir)

        random.seed(1377)
        self.validation_names = random.choices(self.file_names, k=n_valid)

        if (data_type == 'valid'):
            self.volume_paths = [os.path.join(volume_dir, f) for f in self.validation_names]
            self.mask_paths   = [os.path.join(mask_dir, f) for f in self.validation_names]
        else:
            self.train_names = [x for x in self.file_names if x not in self.validation_names]
            self.volume_paths = [os.path.join(volume_dir, f) for f in self.train_names]
            self.mask_paths   = [os.path.join(mask_dir, f) for f in self.train_names]

        # Sort paths
        self.volume_paths.sort()
        self.mask_paths.sort()
        
        assert len(self.volume_paths) == len(self.mask_paths), "Unequal number of volumes and masks"
    
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