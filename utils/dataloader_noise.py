import torch
import numpy as np

# Function to generate noisy data
def generate_noisy_data(data, noise_std):
    # generate noise
    r = 0.5 
    noise = np.random.normal(0, noise_std, data.shape)

    noisy_data = data + (data ** r) * noise
    return noisy_data

# function to return input (noisy data) and output(original data)
def Data_generator(data):

    noise_std = np.random.randint(10,60) / 100
    noisy_data = generate_noisy_data(data, noise_std)

    volume = torch.from_numpy(noisy_data).float().unsqueeze(0).unsqueeze(0)
    mask = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)

    return volume, mask