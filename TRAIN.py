import sys

import torch
import torch.nn as nn
# Memory management
import gc

from utils.dataloader import DataLoaderCreator
from model.UNET import UNet
from utils.metrics import dice_score

def TRAIN_Func(epochs, batch_size, train_volume_dir, train_mask_dir, test_volume_dir, test_mask_dir, feature_maps):
    
    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Parameters
    learning_rate = 0.001

    train_dataloader = DataLoaderCreator(train_volume_dir, train_mask_dir, batch_size)
    test_dataloader  = DataLoaderCreator(test_volume_dir, test_mask_dir, batch_size)

    # Create Model
    model = UNet(in_ch=1, out_ch=1,features=feature_maps).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Loss Funcrion
    criterion = nn.BCELoss() # Binary Segmentation

    # Trainin Loop
    for epoch in range(epochs):
    
        # Train model
        model.train()
    
        train_loss = 0
    
        for i, (volumes, masks) in enumerate(train_dataloader):
        
            volumes = volumes.to(device)
            masks = masks.to(device)
        
            # Forward Pass
            outputs = model(volumes)
        
            # Memory related function
            del volumes
        
            # Calculate Loss
            loss = criterion(outputs, masks)
            train_loss += loss.item()

            # Log
            if i%50 == 0:     
                dice = dice_score(outputs, masks)
                print(f"Epoch: {epoch}/{epochs}, batch: {i+1}/{len(train_dataloader)}, Loss: {loss.item():.4f}, Dice: {dice.item():.4f}")
        

            # Backward Pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            # Memory related function
            del masks
            gc.collect()
            torch.cuda.empty_cache()
        
            
        # calculate average loss
        avg_train_loss = train_loss / len(train_dataloader)
    
        # Evaluation 
        model.eval()
    
        with torch.no_grad():
        
            val_loss = 0
            val_dice = 0
        
            for j, (volumes, masks) in enumerate(test_dataloader):
            
                volumes = volumes.to(device)
                masks   = masks.to(device)
            
                # Forward pass
                outputs = model(volumes)
            
                # Memory related function
                del volumes
            
                # Calculate Loss
                loss = criterion(outputs, masks)
                val_loss += loss.item()
            
            
                # Calculate Dice
                dice = dice_score(outputs, masks)
                val_dice += dice
            
                # Memory related function
                del masks
                gc.collect()
                torch.cuda.empty_cache()
        
            # Average loss and dice
            avg_val_loss = val_loss / len(test_dataloader)
            avg_val_dice = val_dice / len(test_dataloader)
        
        # EPOCH LOG
        print(f"********** Epoch: {epoch+1}/{epochs},Train Loss: {avg_train_loss:.4f}, Validation Loss:{avg_val_loss:.4f}, Validation Dice: {avg_val_dice:.4f}")

    torch.save(model.state_dict(), 'model.pth')


if __name__ == 'main':

    TRAIN_Func(
        epochs = int(sys.argv[1]),
        batch_size = int(sys.argv[2]),
        train_volume_dir = sys.argv[3],
        train_mask_dir = sys.argv[4],
        test_volume_dir = sys.argv[5],
        test_mask_dir = sys.argv[6],
        feature_maps = [16,32,64,128])