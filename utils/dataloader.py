import torch
from torch.utils.data import DataLoader
import numpy as np
import os

# A custom class for loading volumes and masks
class VolumeMaskDataset(torch.utils.data.Dataset):
    def __init__(self, volume_dir, mask_dir, augmentation=False):

        self.augmentation = augmentation

        self.volume_dir = volume_dir
        self.mask_dir = mask_dir

        self.file_names = os.listdir(self.volume_dir)
        TMP = os.listdir(self.mask_dir)
        
        assert len(self.file_names) == len(TMP), "Unequal number of volumes and masks"
        del TMP

        # Log
        print(f'Number of files: {len(self.file_names)}')
    
    def __len__(self):
        return len(self.volume_paths)
    
    def __getitem__(self, idx):

        volume_path = os.path.join(self.volume_dir, self.file_names[idx])
        mask_path   = os.path.join(self.mask_dir, self.file_names[idx])
        
        # Load Volume and mask
        volume = np.load(volume_path)
        mask   = np.load(mask_path)
        
        # normalize data
        volume = (volume - volume.min()) / (volume.max() - volume.min())

        if self.augmentation == True:
            # apply augmentation
            P_vertical = np.random.rand() > 0.5
            P_horizontal = np.random.rand() > 0.5

            if P_vertical:
                volume = np.flip(volume, axis=0)
                mask = np.flip(mask, axis=0)
            
            if P_horizontal:
                volume = np.flip(volume, axis=1)
                mask = np.flip(mask, axis=1)

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

def DataLoaderCreator(volume_dir, mask_dir, batch_size, shuffle = True, augmentation=False):
    # Create dataset
    dataset = VolumeMaskDataset(volume_dir, mask_dir, augmentation)
    
    # create dataloader
    dataloader = DataLoader(dataset, batch_size, shuffle)
    
    return dataloader