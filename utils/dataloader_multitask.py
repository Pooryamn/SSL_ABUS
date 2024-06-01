import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import random

# A custom class for loading volumes and masks
class VolumeMaskDataset(torch.utils.data.Dataset):
    def __init__(self, volume_dir, mask_dir, data_type, n_valid=50, xy_range=[50, 80], z_range=[7, 10], num_range=[3,6]):
        
        self.xy_range = xy_range
        self.z_range = z_range
        self.num_range = num_range

        self.file_names = os.listdir(mask_dir)

        random.seed(1377)
        self.validation_names = random.choices(self.file_names, k=n_valid)
        with open("validation_names.txt", "w") as output:
            output.write(str(self.validation_names))

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
        mask2 = np.load(volume_path)
        mask1   = np.load(mask_path)
        
        volume = mask2.copy()

        number_of_masked_patches = np.random.randint(low=self.num_range[0], high=self.num_range[1])

        for i in range(number_of_masked_patches):
            x1, x2, y1, y2, z1, z2 = self.masking_coordinate_generator(mask2.shape, self.xy_range, self.z_range)
            volume[x1:x2, y1:y2, z1:z2] = 0

        # Convert to Tensor
        volume = torch.from_numpy(volume).float().unsqueeze(0)
        mask1  = torch.from_numpy(mask1).float().unsqueeze(0)
        mask2  = torch.from_numpy(mask2).float().unsqueeze(0)
        
        return volume, mask1, mask2

    def masking_coordinate_generator(self, img_shape,xy_range=[50, 80], z_range=[7, 10]):
        Width = Height = np.random.randint(low= xy_range[0], high=xy_range[1])
        Depth = np.random.randint(low= z_range[0], high=z_range[1])

        X_value = np.random.randint(low=xy_range[1] + 20, high=img_shape[0] - xy_range[1] - 20)
        Y_value = np.random.randint(low=xy_range[1] + 20, high=img_shape[1] - xy_range[1] - 20)
        Z_value = np.random.randint(low=z_range[1], high=img_shape[2] - z_range[1])

        return X_value, X_value + Width, Y_value, Y_value + Height, Z_value, Z_value + Depth


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