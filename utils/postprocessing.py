import torch
import torch.nn.functional as F

def post_process(x, device):
    
    output = torch.empty_like(x, device=device)
    x = x.unsqueeze(0)

    kernel = torch.ones([1, 1, 3], dtype=torch.float32).to(device)

    for i in range(x.shape[1]):
        erosion = (F.conv1d(x[:,i,:], kernel, padding=1) == 3) * 1.0
        opening = (F.conv1d(erosion, kernel, padding=1) > 0) * 1.0
        dilation = (F.conv1d(opening, kernel, padding=1) > 0) * 1.0
        output[i,:] = (F.conv1d(dilation, kernel, padding=1) == 3) * 1.0

    return output