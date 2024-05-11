import sys
import gc

import torch
import torch.nn as nn
import numpy as np

from model.UNET import UNet
from model.ATT_UNET import Attention_Unet
from model.R2UNET import R2U_Net
from utils.dataloader_noise import Data_generator
from utils.metrics import PSNR
from skimage.metrics import structural_similarity as ssim

def TRAIN_Func(epochs, batch_size, model, train_volume_dir, validation_volume_dir, feature_maps):
    
    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Parameters
    learning_rate = 0.001

    train_data      = np.load(train_volume_dir)
    validation_data = np.load(validation_volume_dir)

    if model == "Unet":
        # Create Model 
        model = UNet(in_ch=1, out_ch=1, features=feature_maps).to(device)

    elif model == "Attention_Unet":
        model  = Attention_Unet(in_ch=1, out_ch=1, features=feature_maps).to(device)

    elif model == "R2Unet":
        model = R2U_Net(in_ch=1, out_ch=1, features=feature_maps, t=2).to(device)

    else:
        raise('Error in selecting the model')

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Loss Funcrion
    criterion = nn.MSELoss()
    
    # Trainin Loop
    for epoch in range(epochs):
    
        # Train model
        model.train()
    
        train_loss = 0

        (volumes, masks) = Data_generator(train_data)

        volumes = volumes.to(device)
        masks = masks.to(device)
        # Forward Pass
        outputs = model(volumes)
        
        # Memory related function
        del volumes
        
        # Calculate Loss
        loss = criterion(outputs, masks)
        train_loss += loss.item()

        # Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        masks = masks.squeeze(0).squeeze(0)
        outputs = outputs.squeeze(0).squeeze(0)

        masks = masks.cpu().detach().numpy()
        outputs = outputs.cpu().detach().numpy()

        # metrics
        Train_ssim_score = ssim(masks, outputs, full=True, data_range=1.0)
        Train_ssim_score = Train_ssim_score[0]

        Train_PSNR = PSNR(masks, outputs)
        
        # Memory related function
        del masks
        gc.collect()
        torch.cuda.empty_cache()

    
        # Evaluation 
        model.eval()
    
        with torch.no_grad():
        
            val_loss = 0

            (volumes, masks) = Data_generator(validation_data)
        
            volumes = volumes.to(device)
            masks   = masks.to(device)
            
            # Forward pass
            outputs = model(volumes)
            
            # Memory related function
            del volumes
            
            # Calculate Loss
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            
            # metrics
            masks = masks.squeeze(0).squeeze(0)
            outputs = outputs.squeeze(0).squeeze(0)

            masks = masks.cpu().detach().numpy()
            outputs = outputs.cpu().detach().numpy()

            Valid_ssim_score = ssim(masks, outputs, full=True, data_range=1.0)
            Valid_ssim_score = Valid_ssim_score[0]

            Valid_PSNR = PSNR(masks, outputs)
            
            # Memory related function
            del masks
            gc.collect()
            torch.cuda.empty_cache()
        
        # EPOCH LOG
        print(f"********** Epoch: {epoch+1}/{epochs},Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Train PSNR: {Train_PSNR:.4f}, Validation PSNR: {Valid_PSNR:.4f}, Train SSIM: {Train_ssim_score:.4f}, Validation SSIM: {Valid_ssim_score:.4f}")

        
    torch.save(model.state_dict(), 'model.pth')