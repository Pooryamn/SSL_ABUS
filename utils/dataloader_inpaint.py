import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import random

# A custom class for loading volumes and masks
class VolumeMaskDataset(torch.utils.data.Dataset):
    def __init__(self, volume_dir, data_type, n_valid=14, xy_range=[50, 80], z_range=[7, 10]):

        self.xy_range = xy_range
        self.z_range = z_range

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

        self.mask_paths = []
        for name in self.file_names:
            for i in range(25):
                start = str(i * 32)

                for j in range(2):
                    TMP = name + '#' + start + ',' + str(j)
                    self.mask_paths.append(TMP)
        
        # Log
        print(f'Number of patches for {data_type} set : {len(self.mask_paths)}')

    def __len__(self):
        return len(self.mask_paths)
    
    def __getitem__(self, idx):
        mask_path = self.mask_paths[idx]
        [mask_path, metadata] = mask_path.split('#')
        
        # Load Volume
        mask = np.load(mask_path)['arr_0']

        start, patch_no = metadata.split(',')
        start = int(start)
        patch_no = int(patch_no)

        if patch_no == 0:
            # 0:250
            mask = mask[:250, :, start:start+32]
        else:
            #250:500
            mask = mask[250:, :, start:start+32]

        mask = (mask - mask.min()) / (mask.max() - mask.min())

        # transpose:
        mask = np.traspose(mask, (1,0,2))

        x1, x2, y1, y2, z1, z2 = self.masking_coordinate_generator(mask.shape, self.xy_range, self.z_range)

        # here volume is a patch of data with a randon black cube within
        volume = mask.copy()
        volume[x1:x2, y1:y2, z1:z2] = 0
        
        # Convert to Tensor
        volume = torch.from_numpy(volume).float().unsqueeze(0)
        mask   = torch.from_numpy(mask).float().unsqueeze(0)
        
        return volume, mask

    def masking_coordinate_generator(self, img_shape,xy_range=[50, 80], z_range=[7, 10]):
        Width = Height = np.random.randint(low= xy_range[0], high=xy_range[1])
        Depth = np.random.randint(low= z_range[0], high=z_range[1])

        X_value = np.random.randint(low=xy_range[1] + 20, high=img_shape[0] - xy_range[1] - 20)
        Y_value = np.random.randint(low=xy_range[1] + 20, high=img_shape[1] - xy_range[1] - 20)
        Z_value = np.random.randint(low=z_range[1], high=img_shape[2] - z_range[1])

        return X_value, X_value + Width, Y_value, Y_value + Height, Z_value, Z_value + Depth
        

def DataLoaderCreator(volume_dir, batch_size, data_type, n_valid=14, shuffle = True):
    # Create dataset
    dataset = VolumeMaskDataset(volume_dir, data_type, n_valid)
    
    # create dataloader
    dataloader = DataLoader(dataset, batch_size, shuffle)
    
    return dataloader