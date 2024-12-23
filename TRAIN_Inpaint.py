import sys

import torch
import torch.nn as nn
import numpy as np
# Memory management
import gc

import pickle

from utils.dataloader_inpaint import DataLoaderCreator
from model.UNET import UNet
from model.ATT_UNET import Attention_Unet
from model.R2UNET import R2U_Net
from model.ATTR2_UNET import ATTR2U_Net
from model.DATTR2_UNET import DoubleATTR2U_Net
from utils.metrics import PSNR
from utils.early_stop import EarlyStopper
from utils.weight_init import WEIGHT_INITIALIZATION
from utils.losses import SSIMLoss


def TRAIN_Func(epochs, batch_size, model_name, volume_dir, feature_maps, learning_rate=0.001, weight_path = None, log_path = None, weight_init = None):
    
    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    train_dataloader = DataLoaderCreator(volume_dir, batch_size, data_type='train', n_valid=14)
    test_dataloader  = DataLoaderCreator(volume_dir, batch_size, data_type='valid', n_valid=14)

    if model_name == "Unet":
        # Create Model 
        model = UNet(in_ch=1, out_ch=1, features=feature_maps).to(device)

    elif model_name == "Attention_Unet":
        model  = Attention_Unet(in_ch=1, out_ch=1, features=feature_maps).to(device)
    
    elif model_name == "R2Unet":
        model = R2U_Net(in_ch=1, out_ch=1, features=feature_maps, t=2).to(device)

    elif model_name == "AttR2Unet":
        model = ATTR2U_Net(in_ch=1, out_ch=1, features=feature_maps, t=2).to(device)

    elif model_name == 'DAttR2Unet':
        model = DoubleATTR2U_Net(in_ch=1, out_ch=1, features=feature_maps, t=2).to(device)

    else:
        raise('Error in selecing the model')

    if (weight_path != None):
        model.load_state_dict(torch.load(weight_path, map_location=torch.device(device)))
    elif (weight_init != None):
        WEIGHT_INITIALIZATION(model, weight_init)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Loss Funcrion
    criterion =  SSIMLoss().to(device) # inpainting Loss

    # early stop
    early_stopper = EarlyStopper(patience=4, min_delta=0.3)

    # create a suitable name for saving the weights
    model_name = model_name + '.pth'

    # plot data
    if (log_path == None):
        plot_data = {
            'train_loss': [],
            'train_psnr': [],
            'train_ssim': [],
            'valid_loss': [],
            'valid_psnr': [],
            'valid_ssim': []
        }
    else:
        with open(log_path, 'rb') as f:
            plot_data = pickle.load(f)

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
        
            # Calculate Loss
            loss, SSIM = criterion(outputs, masks)
            Train_LOSS += loss.item()

            # Memory related function
            del volumes

            # metrics
            Train_SSIM += SSIM.item()

            # convert to numpy first
            masks   = np.array(masks.squeeze(0).squeeze(0).cpu().detach().numpy())
            outputs = np.array(outputs.squeeze(0).squeeze(0).cpu().detach().numpy())

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

            if (i % 30 == 0 and i!=0):
                print(f'*** Batch {i} / {len(train_dataloader)}, Train Loss: {(Train_LOSS / i):.4f}, Train SSIM: {(Train_SSIM / i):.4f}, Train PSNR: {(Train_PSNR / i):.4f} ***')
        
            
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
            Max_SSIM = 0
        
            for j, (volumes, masks) in enumerate(test_dataloader):
            
                volumes = volumes.to(device)
                masks   = masks.to(device)
            
                # Forward pass
                outputs = model(volumes)
            
                # Calculate Loss
                loss, SSIM = criterion(outputs, masks)
                Val_LOSS += loss.item()

                # Memory related function
                del volumes

                # metrics
                Val_SSIM += SSIM.item()

                # convert to numpy first
                masks   = np.array(masks.squeeze(0).squeeze(0).cpu().detach().numpy())
                outputs = np.array(outputs.squeeze(0).squeeze(0).cpu().detach().numpy())

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

        # save epoch information
        plot_data['train_loss'].append(AVG_train_loss)
        plot_data['train_psnr'].append(AVG_train_psnr)
        plot_data['train_ssim'].append(AVG_train_ssim)
        plot_data['valid_loss'].append(AVG_valid_loss)
        plot_data['valid_psnr'].append(AVG_valid_psnr)
        plot_data['valid_ssim'].append(AVG_valid_ssim)

        with open('Model_history.pkl', 'wb') as f:
            pickle.dump(plot_data, f)

        # Save Best
        if (AVG_valid_ssim > Max_SSIM):
            
            torch.save(model.state_dict(), model_name)
            Max_SSIM = AVG_valid_ssim       

        # EPOCH LOG
        print(f"******************** Epoch: {epoch+1}/{epochs}, Train Loss: {AVG_train_loss:.4f}, Validation Loss: {AVG_valid_loss:.4f}, Train SSIM: {AVG_train_ssim:.4f}, Validation SSIM: {AVG_valid_ssim:.4f}, Train PSNR: {AVG_train_psnr:.4f}, Validation PSNR: {AVG_valid_psnr:.4f}")

        # check for early stopping
        if (early_stopper.early_stop(AVG_valid_loss)):
            print(f'########## Eearly stop in epoch {epoch+1}')
            break  