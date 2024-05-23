import sys

import torch
import torch.nn as nn
# Memory management
import gc

from utils.dataloader import DataLoaderCreator
from model.UNET import UNet
from model.ATT_UNET import Attention_Unet
from model.R2UNET import R2U_Net
from model.ATTR2_UNET import ATTR2U_Net
from model.DATTR2_UNET import DoubleATTR2U_Net
from model.DETECTION import Detection_model
from utils.metrics import Sensitivity
from utils.early_stop import EarlyStopper
from utils.weight_init import WEIGHT_INITIALIZATION
from utils.losses import FocalLoss



def TRAIN_Func(epochs, batch_size, model, train_volume_dir, train_mask_dir, test_volume_dir, test_mask_dir, feature_maps, Detection_feature_maps=[16,32,64],Train_type='full' , learning_rate=0.001, weight_path = None, log_path = None, weight_init = None):
    
    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)


    train_dataloader = DataLoaderCreator(train_volume_dir, train_mask_dir, batch_size, augmentation=True)
    test_dataloader  = DataLoaderCreator(test_volume_dir, test_mask_dir, batch_size, augmentation=False)

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

    params = list(model.parameters())
    if (Train_type == 'Detection'):
        for param in params:
            param.requires_grad = False

    elif (Train_type == 'Partial'):
        num_of_freeze_params = len(params) // 2
        for param in params[:num_of_freeze_params]:
            param.requires_grad = False
    
    elif (Train_type == 'full'):
        pass
    
    else:
        raise('Invalid train type!')

    
    # Add two models toghether
    D_model = Detection_model(in_ch=1, out_ch=1, features=Detection_feature_maps, threshold=0.5).to(device)

    Combined_model = nn.Sequential(model, D_model).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Loss Funcrion
    criterion = FocalLoss().to(device) # Binary classification

    # early stop
    early_stopper = EarlyStopper(patience=4, min_delta=0.3)

    # create a suitable name for saving the weights
    model_name = model_name + '.pth'

    # plot data
    if (log_path == None):
        plot_data = {
            'train_loss': [],
            'train_sensitivity': [],
            'train_FP': [],
            'valid_loss': [],
            'valid_sensitivity': [],
            'valid_FP': []
        }
    else:
        with open(log_path, 'rb') as f:
            plot_data = pickle.load(f)

    # Trainin Loop
    for epoch in range(epochs):
    
        # Train model
        Combined_model.train()
    
        Train_LOSS = 0
        Train_SENSITIVITY = 0
        Train_FP = 0
    
        for i, (volumes, masks) in enumerate(train_dataloader):
        
            volumes = volumes.to(device)
            masks = masks.to(device)
        
            # Forward Pass
            outputs = Combined_model(volumes)
        
            # Memory related function
            del volumes
        
            # Calculate Loss
            loss = criterion(outputs, masks)
            Train_LOSS += loss.item()

            # Metrics
            Sen, fp = Sensitivity(outputs, masks)
            Train_SENSITIVITY += Sen.item()
            Train_FP += fp.item()

            # Backward Pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            # Memory related function
            del masks
            gc.collect()
            torch.cuda.empty_cache()

            if (i % 30 == 0 and i!=0):
                print(f'*** Batch {i} / {len(train_dataloader)}, Train Loss: {(Train_LOSS / i):.4f}, Train Sensitivity: {(Train_SENSITIVITY / i):.4f}, Train FP: {(Train_FP / i):.4f} ***')
      
        
            
        # calculate averages
        AVG_train_loss = Train_LOSS / len(train_dataloader)
        AVG_train_sensitivity = Train_SENSITIVITY / len(train_dataloader)
        AVG_train_FP = Train_FP / len(train_dataloader)
    
        # Evaluation 
        Combined_model.eval()
    
        with torch.no_grad():
        
            Val_LOSS = 0
            Val_SENSITIVITY = 0
            Val_FP = 0
            Max_SENSITIVITY = 0
        
            for j, (volumes, masks) in enumerate(test_dataloader):
            
                volumes = volumes.to(device)
                masks   = masks.to(device)
            
                # Forward pass
                outputs = Combined_model(volumes)
            
                # Memory related function
                del volumes
            
                # Calculate Loss
                loss = criterion(outputs, masks)
                Val_LOSS += loss.item()
            
                # Metrics
                Sen, fp = Sensitivity(outputs, masks)
                Val_SENSITIVITY += Sen.item()
                Val_FP += fp.item()
            
                # Memory related function
                del masks
                gc.collect()
                torch.cuda.empty_cache()
        
            # calculate averages
            AVG_valid_loss = Val_LOSS / len(test_dataloader)
            AVG_valid_sensitivity = Val_SENSITIVITY / len(test_dataloader)
            AVG_valid_FP = Val_FP / len(test_dataloader)
        
        # save epoch information
        plot_data['train_loss'].append(AVG_train_loss)
        plot_data['train_sensitivity'].append(AVG_train_sensitivity)
        plot_data['train_FP'].append(AVG_train_FP)
        plot_data['valid_loss'].append(AVG_valid_loss)
        plot_data['valid_sensitivity'].append(AVG_valid_sensitivity)
        plot_data['valid_FP'].append(AVG_valid_FP)
        
        with open('Model_history.pkl', 'wb') as f:
            pickle.dump(plot_data, f)

        # Save Best
        if (AVG_valid_sensitivity > Max_SENSITIVITY):
            torch.save(model.state_dict(), model_name)
            Max_SENSITIVITY =  AVG_valid_sensitivity    

        # EPOCH LOG
        print(f"******************** Epoch: {epoch+1}/{epochs}, Train Loss: {AVG_train_loss:.4f}, Validation Loss: {AVG_valid_loss:.4f}, Train Sensitivity: {AVG_train_sensitivity:.4f}, Validation Sensitivity: {AVG_valid_sensitivity:.4f}, Train FP: {AVG_train_FP:.4f}, Validation FP: {AVG_valid_FP:.4f}")

        # check for early stopping
        if (early_stopper.early_stop(AVG_valid_loss)):
            print(f'########## Eearly stop in epoch {epoch+1}')
            break 