import sys

import torch
import torch.nn as nn
# Memory management
import gc

from utils.dataloader_noise import DataLoaderCreator
from model.UNET import UNet
from model.ATT_UNET import Attention_Unet
from model.R2UNET import R2U_Net
from utils.metrics import PSNR
from skimage.metrics import structural_similarity as ssim


def TRAIN_Func(epochs, batch_size, model, volume_dir, mask_dir, feature_maps):
    
    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Parameters
    learning_rate = 0.001

    train_dataloader = DataLoaderCreator(volume_dir, mask_dir, batch_size, data_type='train',n_valid=40)
    test_dataloader  = DataLoaderCreator(volume_dir, mask_dir, batch_size, data_type='valid', n_valid=40)

    if model == "Unet":
        # Create Model 
        model = UNet(in_ch=1, out_ch=1, features=feature_maps).to(device)

    elif model == "Attention_Unet":
        model  = Attention_Unet(in_ch=1, out_ch=1, features=feature_maps).to(device)
    
    elif model == "R2Unet":
        model = R2U_Net(in_ch=1, out_ch=1, features=feature_maps, t=2).to(device)

    else:
        raise('Error in selecing the model')

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Loss Funcrion
    criterion = nn.MSELoss() # denoising Loss

    # Trainin Loop
    for epoch in range(epochs):
    
        # Train model
        model.train()
    
        Train_LOSS = 0
        Train_PSNR = 0
        Train_SSIM = 0
    
        for i, (volumes, masks) in enumerate(train_dataloader):
        
            volumes = volumes.to(device)
            masks = masks.to(device)
        
            # Forward Pass
            outputs = model(volumes)
        
            # Memory related function
            del volumes
        
            # Calculate Loss
            loss = criterion(outputs, masks)
            Train_LOSS += loss.item()

            # Calculate metrics
            # convert to numpy first
            masks   = np.array(masks.squeeze(0).squeeze(0))
            outputs = np.array(outputs.squeeze(0).squeeze(0))

            SSIM = ssim(masks, outputs, full=True, data_range=1.0)
            Train_SSIM += SSIM[0]

            PSNR_score = PSNR(masks, outputs)
            Train_PSNR += PSNR_score

            # Backward Pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            # Memory related function
            del masks
            gc.collect()
            torch.cuda.empty_cache()

            print(f'*** Batch {i+1} / {len(train_dataloader)} ***')
        
            
        # calculate averages
        AVG_train_loss = Train_LOSS / len(train_dataloader)
        AVG_train_psnr = Train_PSNR / len(train_dataloader)
        AVG_train_ssim = Train_SSIM / len(train_dataloader)
    
        # Evaluation 
        model.eval()
    
        with torch.no_grad():
               
            Val_LOSS = 0
            Val_PSNR = 0
            Val_SSIM = 0
            Max_PSNR = 0
        
            for j, (volumes, masks) in enumerate(test_dataloader):
            
                volumes = volumes.to(device)
                masks   = masks.to(device)
            
                # Forward pass
                outputs = model(volumes)
            
                # Memory related function
                del volumes
            
                # Calculate Loss
                loss = criterion(outputs, masks)
                Val_LOSS += loss.item()

                # Calculate metrics
                # convert to numpy first
                masks   = np.array(masks.squeeze(0).squeeze(0))
                outputs = np.array(outputs.squeeze(0).squeeze(0))

                SSIM = ssim(masks, outputs, full=True, data_range=1.0)
                Val_SSIM += SSIM[0]

                PSNR_score = PSNR(masks, outputs)
                Val_PSNR += PSNR_score
            
                # Memory related function
                del masks
                gc.collect()
                torch.cuda.empty_cache()
        
            # calculate averages
            AVG_valid_loss = Val_LOSS / len(test_dataloader)
            AVG_valid_psnr = Val_PSNR / len(test_dataloader)
            AVG_valid_ssim = Val_SSIM / len(test_dataloader)

        # Save Best
        if (AVG_valid_psnr > Max_PSNR):
            torch.save(model.state_dict(), 'model.pth')

        # EPOCH LOG
        print(f"******************** Epoch: {epoch+1}/{epochs}, Train Loss: {AVG_train_loss:.4f}, Validation Loss: {AVG_valid_loss:.4f}, Train SSIM: {AVG_train_ssim:.4f}, Validation SSIM: {AVG_valid_ssim:.4f}, Train PSNR: {AVG_train_psnr:.4f}, Validation PSNR: {AVG_valid_psnr:.4f}")




TRAIN_Func(
    epochs = 1,
    batch_size = 1,
    model = 'Attention_Unet',
    train_volume_dir = '/teamspace/studios/this_studio/TrainSet/TDSC_Patches/Volumes',
    train_mask_dir = '/teamspace/studios/this_studio/TrainSet/TDSC_Patches/Mask',
    test_volume_dir = '/teamspace/studios/this_studio/TestSet/TDSC_Patches/Volumes',
    test_mask_dir = '/teamspace/studios/this_studio/TestSet/TDSC_Patches/Mask',
    feature_maps = [16,32,64,128,256]
    )