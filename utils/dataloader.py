import torch
from torch.utils.data import DataLoader
import numpy as np
import os

# A custom class for loading volumes and masks
class VolumeMaskDataset(torch.utils.data.Dataset):
    def __init__(self, volume_dir, mask_dir):
        self.volume_paths = [os.path.join(volume_dir, f) for f in os.listdir(volume_dir)]
        self.mask_paths   = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir)]
        
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
        
        # normalize data
        volume = (volume - volume.min()) / (volume.max() - volume.min())

        # mask Vec
        mask_vector = self.Mask2Vec(mask)
        
        # Convert to Tensor
        volume = torch.from_numpy(volume).float().unsqueeze(0)
        mask   = torch.from_numpy(mask_vector).float().unsqueeze(0)
        
        return volume, mask
    
    def Mask2Vec(self, mask):
        
        mask_vector = np.zeros((32))
        
        if (mask.max() > 0):
            for i in range(mask.shape[2]):
                if (mask[:,:,i].sum() >= 900):
                    mask_vector[i] = 1.0

        return mask_vector


def Test_Dataset_Class():
    volume_dir = '/kaggle/input/tdscabus-train-patches/TDSC_Patches/Volumes'
    mask_dir   = '/kaggle/input/tdscabus-train-patches/TDSC_Patches/Mask'
    
    # create dataset
    dataset = VolumeMaskDataset(volume_dir, mask_dir)
    
    print(dataset[10])

def DataLoaderCreator(volume_dir, mask_dir, batch_size, shuffle = True):
    # Create dataset
    dataset = VolumeMaskDataset(volume_dir, mask_dir)
    
    # create dataloader
    dataloader = DataLoader(dataset, batch_size, shuffle)
    
    return dataloader