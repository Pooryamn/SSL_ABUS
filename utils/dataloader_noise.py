import torch
from torch.utils.data import DataLoader
import numpy as np
import os

# Function to generate noisy data
def generate_noisy_data(data, noise_prop):
    # generate noise
    noise = np.random.normal(0, 1, data.shape)

    noisy_data = data + (noise_prop * noise)

    return noisy_data

# function to return input (noisy data) and output(original data)
def Data_generator(data):

    noise_prop = np.random.randint(5,50) / 100
    noisy_data = generate_noisy_data(data, noise_prop)

    volume = torch.from_numpy(noisy_data).float().unsqueeze(0).unsqueeze(0)
    mask = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)

    return volume, mask