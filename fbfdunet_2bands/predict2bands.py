# Importing the necessary libraries:

import numpy as np
import time

import torch

from models.fbfdunetln import FDUNet
from fbfdunet_2bands.train2bands import expsetupparam 

from utils.funmatrix import createForwMat, applyInvMat
from utils.OATdataloader import gettestdata

from utils.comparison import comparisonSOTA, calcMOAPS 

# ---------------------------------------------------------------------------
def predict_out(net, inp, Ao, nx, device):
    x = torch.as_tensor(inp)
    x = x.to(device=device)
    x = torch.unsqueeze(x,1)
    x = x.type(torch.float32)
    Ao = torch.as_tensor(Ao).type(torch.float32)
    Ao = Ao.to(device=device)
    #
    dimS = x.shape # (-1,1,Ns,Nt)
    dimI = (dimS[0],dimS[1],nx,nx) # (-1,1,nx,nx)
    f0 = applyInvMat(x,Ao,dimS,dimI) # (-1,1,nx,nx)
    with torch.no_grad():
        pred = net(f0) # (-1,2,nx,nx)
        predL = pred[:,0,:,:] # (-1,1,nx,nx)
        predH = pred[:,1,:,:] # (-1,1,nx,nx)
        predL = predL.squeeze(1) # (-1,nx,nx)
        predH = predH.squeeze(1) # (-1,nx,nx)
    
    return predL.detach().to("cpu").numpy(), predH.detach().to("cpu").numpy()

# ---------------------------------------------------------------------------
def predict_fbfdunetln(n_name,ntest,ckp_best,MM=np.array(0)):
    
    device = "cpu"
    cache_dir = '../data/cache/'

    Ns,Nt,dx,nx,dsa,arco,vs,to,tf = expsetupparam()
    out_features=2 # two frequency bands
    dibujopaper=0
    
    # Loading test data
    X,Y,SNR,POSE = gettestdata(cache_dir,n_name,ntest)
    
    # Creating model-based matrix
    if not(MM.any()):
        Ao = createForwMat(Ns,Nt,dx,nx,dsa,arco,vs,to,tf)
    else:
        Ao = MM
           
    print('Loading network...')
    net = FDUNet(nx,nx,out_features)
    checkpoint = torch.load(ckp_best,map_location=torch.device(device))
    net.load_state_dict(checkpoint['state_dict'])
    print('done')
    
    print('Predicting...') 
    start = time.perf_counter()
    predL, predH = predict_out(net, X, Ao, nx, device)
    #pred = predL + predH
    end = time.perf_counter()
    TE = end - start
    print('done')
    
    # Calculate metrics and do a comparison with DAS and LBP
    SSIM,PC,RMSE,PSNR = comparisonSOTA(Ns,Nt,dx,nx,dsa,arco,vs,to,tf,Ao,X,Y,predL,predH,SNR,POSE,cache_dir,dibujopaper)
    
    f,sTf,sLf,sHf=calcMOAPS(predL[-1,:,:],predH[-1,:,:],Y[-1,:,:],Ao,tf-to,Ns,Nt)
    
    return 
