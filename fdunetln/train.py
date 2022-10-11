# Importing the necessary libraries:

#import os
import logging
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import Adam

from utils.funmatrix import createForwMat, applyInvMat
from utils.OATdataloader import gettraindata, get_trainloader
from utils.SLcheckpoint import load_ckp, save_ckp

from models.fdunetln import FDUNet

#import wandb

# ---------------------------------------------------------------------------
def expsetupparam():
    
    # Experimental setup parameters
    Ns=32      # number of detectors
    Nt=1024      # number of time samples
    dx=50e-6   # pixel size  in the x direction [m] 
    nx=128       # number of pixels in the x direction for a 2-D image region
    dsa=8.35e-3 # radius of the circunference where the detectors are placed [m]
    arco=360
    vs=1479;    # speed of sound [m/s]
    to=2e-6        # initial time [s]
    tf=15e-6    # final time [s] 

    return dx,nx,dsa,arco,vs,to,tf     

# ---------------------------------------------------------------------------
def train_fdunet(n_name,batch_size,epochs,continuetrain,plotresults,WandB,fecha):
    
    cache_dir = '../data/cache/'
    
    # Net Main Parameters
    lr = 1e-4 #1e-4
    beta1 = 0.9 #0.5
    #traindata = False
    ckp_last='fdunet' + fecha + '.pth' # name of the file of the saved weights of the trained net
    ckp_best='fdunet_best' + fecha + '.pth'
    # Experiment parameters
    Ns,Nt,dx,nx,dsa,arco,vs,to,tf = expsetupparam()
    
    # 1. Set device
    ngpu = 1 # number og GPUs available. Use 0 for CPU mode.
    device = ""
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device to be used: {device}")
    
    # 1. Create de network
    net = FDUNet(nx,nx).to(device=device)
    
    # Number of net parameters
    NoP = sum(p.numel() for p in net.parameters())
    print(f"The number of parameters of the network to be trained is: {NoP}")    
    
    # 2. Define loss function and optimizer and the the learning rate scheduler
    optimizer = Adam(net.parameters(), lr=lr, betas=(beta1, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.5,patience=2,threshold=0.005,eps=1e-6,verbose=True)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.5,verbose=True)
    LossFn = nn.MSELoss()
    
    # Handle multi-gpu if desired
    if (device.type == "cuda") and (ngpu > 1):
        net = nn.DataParallel(net, list(range(ngpu)))
    
    # 3. Create datase
    # Create Model-based Matrix
    Ao = createForwMat(Ns,Nt,dx,nx,dsa,arco,vs,to,tf)
    Ao = torch.as_tensor(Ao).type(torch.float32)
    Ao = Ao.to(device=device)
    
    # Get data
    X,Y = gettraindata(cache_dir, n_name)
    
    # 4. Create data loader
    val_percent = 0.1#0.2
    X = torch.as_tensor(X).type(torch.float32) 
    Y = torch.as_tensor(Y).type(torch.float32)
    train_loader, val_loader, n_train, n_val = get_trainloader(X, Y, val_percent, batch_size)
    
    # 5. Initialize logging and initialize weights or continue a previous training 
    logfilename='TrainingLog_FDUNet'+fecha+'.log'
    
    if continuetrain:
        net, optimizer, last_epoch, valid_loss_min = load_ckp(ckp_last, net, optimizer)
        print('Values loaded:')
        #print("model = ", net)
        print("optimizer = ", optimizer)
        print("last_epoch = ", last_epoch)
        print("valid_loss_min = ", valid_loss_min)
        print("valid_loss_min = {:.6f}".format(valid_loss_min))
        start_epoch = last_epoch + 1
        lr = optimizer.param_groups[0]['lr']
        logging.basicConfig(filename=logfilename,format='%(asctime)s - %(message)s', level=logging.INFO)
        logging.info(f'''Continuing training:
            Epochs:                {epochs}
            Batch size:            {batch_size}
            Initial learning rate: {lr}
            Training size:         {n_train}
            Validation size:       {n_val}
            Device:                {device.type}
            ''')
    else:
        # Apply the weights_init function to randomly initialize all weights
        net.apply(initialize_weights)
        start_epoch = 1
        valid_loss_min = 100
        logging.basicConfig(filename=logfilename, filemode='w',format='%(asctime)s - %(message)s', level=logging.INFO)
        logging.info(f'''Starting training:
            Epochs:                {epochs}
            Batch size:            {batch_size}
            Initial learning rate: {lr}
            Training size:         {n_train}
            Validation size:       {n_val}
            Device:                {device.type}
            ''')
    
    if WandB:
        experiment =wandb.init(project='fdunet_oa', entity='dl_oa_fiuba')
        experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=lr, val_percent=val_percent))
    # Print model
    # print(net)

    # 6. Begin training
    TLV=np.zeros((epochs,)) #vector to record the train loss per epoch 
    VLV=np.zeros((epochs,)) #vector to record the validation loss per epoch
    EV=np.zeros((epochs,)) # epoch vector to plot later
    global_step = 0
    
    #for epoch in range(epochs):
    for epoch in range(start_epoch, start_epoch+epochs):
        net.train() # Let pytorch know that we are in train-mode
        epoch_loss = 0.0
        epoch_val_loss = 0.0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs+start_epoch-1}', unit='sino') as pbar:
            for x,y in train_loader:
                # clear the gradients
                optimizer.zero_grad(set_to_none=True)
                # input and truth to device
                x = x.to(device=device) # (-1,Ns,Nt)
                x = torch.unsqueeze(x,1) # (-1,1,Ns,Nt)
                x = x.type(torch.float32)
                #
                dimS = x.shape # (-1,1,Ns,Nt)
                dimI = (dimS[0],dimS[1],nx,nx) # (-1,1,nx,nx)
                f0 = applyInvMat(x,Ao,dimS,dimI) # (-1,1,nx,nx)
                # compute the model output
                pred = net(f0) # (-1,1,nx,nx)
                pred = torch.squeeze(pred,1)
                # True image
                y = y.to(device=device) # (-1,nx,nx)
                y = y.type(torch.float32)
                # calculate image loss
                loss = LossFn(pred,y)
                # credit assignment
                loss.backward()
                # update model weights
                optimizer.step()
                
                pbar.update(x.shape[0])
                global_step += 1
                
                epoch_loss += loss.item()
                
                pbar.set_postfix(**{'loss (batch)': loss.item()})
            
            epoch_train_loss = epoch_loss / len(train_loader)
        # Scheduler Step
        #scheduler.step()
        # Evaluation round
        with torch.no_grad():
            for xv, yv in tqdm(val_loader, total=len(val_loader), desc='Validation round', position=0, leave=True):
                # input and truth to device
                xv = xv.to(device=device)
                xv = torch.unsqueeze(xv,1)
                xv = xv.type(torch.float32)
                #
                dimS = xv.shape # (-1,1,Ns,Nt)
                dimI = (dimS[0],dimS[1],nx,nx) # (-1,1,nx,nx)
                f0v = applyInvMat(xv,Ao,dimS,dimI) # (-1,1,nx,nx)
                #
                yv = yv.to(device=device) # (-1,nx,nx)
                yv = yv.type(torch.float32)
                # compute the model output
                predv = net(f0v) # (-1,1,nx,nx)
                predv = torch.squeeze(predv,1)
                # calculate image loss
                loss = LossFn(predv,yv)
                epoch_val_loss += loss.item()
        
        epoch_val_loss = epoch_val_loss / len(val_loader)
        # Scheduler ReduceLROnPlateau
        scheduler.step(epoch_val_loss)
        # Scheduler StepLR
        #scheduler.step()
        
        # logging validation score per epoch
        logging.info(f'''Epoch: {epoch} - Validation score: {np.round(epoch_val_loss,5)}''')
        
        # print training/validation statistics 
        #print('\n Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        #    epoch,
        print('\n Training Loss: {:.5f} \tValidation Loss: {:.5f}'.format(
            epoch_train_loss,
            epoch_val_loss
            ))
        
        # Loss vectors for plotting results
        TLV[epoch-start_epoch]=epoch_train_loss
        VLV[epoch-start_epoch]=epoch_val_loss
        EV[epoch-start_epoch]=epoch
        
        # create checkpoint variable and add important data
        checkpoint = {
            'epoch': epoch,
            'valid_loss_min': epoch_val_loss,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        
        # save checkpoint
        save_ckp(checkpoint, False, ckp_last, ckp_best)
        
        
        # save the model if validation loss has decreased
        if epoch_val_loss <= valid_loss_min:
            print('\n Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,epoch_val_loss),'\n')
            # save checkpoint as best model
            save_ckp(checkpoint, True, ckp_last, ckp_best)
            valid_loss_min = epoch_val_loss
            logging.info(f'Val loss deccreased on epoch {epoch}!')
        
        # Logging wandb
        if WandB:
            experiment.log({
                'train loss': epoch_train_loss,
                'val loss': epoch_val_loss,
                'epoch': epoch
            })
    
    if WandB:
        wandb.finish()
    
    del x,y,xv,yv,pred,predv,Ao
    
    if plotresults:
        plt.figure();
        plt.grid(True,linestyle='--')
        plt.xlabel('epoch'); plt.ylabel('Loss')
        plt.plot(EV,TLV,'--',label='Train Loss')
        plt.plot(EV,VLV,'-',label='Val Loss')
        plt.legend(loc='best',shadow=True, fontsize='x-large')
    
    #np.save('MSE'+fecha,EV,TLV,VLV)
    return EV,TLV,VLV
    #return
                
# --------------------------------------------------------------------------- 
def initialize_weights(m):
    if isinstance(m,(nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight.data,0.0,0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data,0)
    elif isinstance(m,(nn.BatchNorm2d,nn.LayerNorm)):
        nn.init.normal_(m.weight.data,1.0,0.02)
        nn.init.constant_(m.bias.data,0)

# ---------------------------------------------------------------------------
#if __name__=='__main__':
#    test()
