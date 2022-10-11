# Importing the necessary libraries:

import numpy as np
from scipy.sparse.linalg import lsqr
from fnnls import fnnls
from tqdm import tqdm # progress bar for loops
import matplotlib.pyplot as plt
import time
#from scipy.optimize import nnls # Non-negative least squares solver
from scipy.fft import fft
from scipy import sparse
from scipy.sparse import csc_matrix
#from scipy.sparse import hstack, vstack
from scipy.sparse import bmat

from utils.funmatrix import createForwMat
from utils.OATdataloader import gettestdata
from utils.filtros import CreateFilterMatrix2
from fbfdunet_2bands.train2bands import expsetupparam

import math
from scipy import stats
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio

from utils.comparison import calcMOAPS

import gc 

from scipy.io import savemat

# ---------------------------------------------------------------------------
def predictfbmb(n_name,ntest):
    
    Ns,Nt,dx,nx,dsa,arco,vs,to,tf = expsetupparam()
    
    L1 = 0.5
    L2 = 0.5
    eta = 1 # the square of the maximun value of the sinograms are approx. 1000
    L = 5 # mean(std(Sinograms_test)) = 1.35
    
    # Loading test data
    X,Y,SNR,POSE = gettestdata(cache_dir,n_name,ntest)
    
    # Creating model-based matrix
    Ao = createForwMat(Ns,Nt,dx,nx,dsa,arco,vs,to,tf)
    
    # Generate complementary filter matrix
    FL,FH = CreateFilterMatrix2(Nt,dx,nx,dsa,arco,vs,to,tf) # (Nt,Nt)
        
    Mm = bmat([[csc_matrix(Ao), csc_matrix(Ao)], [np.sqrt(L)*csc_matrix(np.eye(Ns*Nt,nx*nx)), np.sqrt(L)*csc_matrix(np.eye(Ns*Nt,nx*nx))],[np.sqrt(eta*L1)*sparse.kron(np.eye(Ns),FL)@csc_matrix(Ao),csc_matrix(np.zeros((Ns*Nt,nx*nx)))],[csc_matrix(np.zeros((Ns*Nt,nx*nx))),np.sqrt(eta*L2)*sparse.kron(np.eye(Ns),FH)@csc_matrix(Ao)]])
    
    del Ao, FL, FH
    gc.collect()
    
    print('Obtaining images from sinograms...')
    Mx2 = np.zeros((ntest,2 * nx * nx))
    TE = np.zeros((ntest,1)) # execution time
    for i1 in tqdm(range(0,ntest)):
        po = X[i1,:,:].ravel() # (Ns*Nt,)
        p = np.concatenate((po,np.zeros(3*Ns*Nt,)),axis=0) # (4*Ns*Nt,)
            
        start = time.perf_counter()
        
        x2, res = fnnls(Mm, p);
        
        gc.collect()    
        
        end = time.perf_counter()
        TE[i1,0] = end - start
                  
        Mx2[i1,:] = x2
    
    del Mm,p,x2
    gc.collect()
    
    xL = Mx2[:,0:(nx*nx)] # Low frequency images
    xH = Mx2[:,(nx*nx):] # High frequency images
    
    print('Calculating metrics...')
    SSIM=np.zeros((ntest,1))
    PC=np.zeros((ntest,1))
    RMSE=np.zeros((ntest,1))
    PSNR=np.zeros((ntest,1))
    
    xL = xL.astype(np.float32);
    xH = xH.astype(np.float32);
    pred = xL + xH
    pred = np.reshape(pred,(ntest,nx,nx))
    xL = np.reshape(xL,(ntest,nx,nx))
    xH = np.reshape(xH,(ntest,nx,nx))
    
    Best=0
    Worst=1
    for i1 in tqdm(range(0,ntest)):
        trueimage=Y[i1,:,:].astype(np.float32);
        predimage = pred[i1,:,:]; predL = xL[i1,:,:]; predH = xH[i1,:,:]; 
        valnorm = np.max(predimage.ravel())
        predimage = predimage/valnorm; predL = predL/valnorm; predH = predH/valnorm
        SSIM[i1,0]=structural_similarity(trueimage,predimage) 
        PC[i1,0]=stats.pearsonr(trueimage.ravel(),predimage.ravel())[0]  
        RMSE[i1,0]=math.sqrt(mean_squared_error(trueimage,predimage))
        PSNR[i1,0]=peak_signal_noise_ratio(trueimage,predimage)
        
        if SSIM[i1,0]>Best:
            Best = SSIM[i1,0]
            SNRb=SNR[i1]
            POSEb=POSE[i1]
            Ptb = trueimage
            Pmb=predimage; PmLb=predL;  PmHb=predH;
        if SSIM[i1,0]<Worst:
            Worst = SSIM[i1,0]
            SNRw=SNR[i1]
            POSEw=POSE[i1]
            Ptw = trueimage
            Pmw=predimage; PmLw=predL;  PmHw=predH; 
        
    print('\n')
    print('############################################################### \n')
    print('Metrics results Matrix Method: \n', 'SSIM: ',round(np.mean(SSIM[:,0]),3), ' PC: ', round(np.mean(PC[:,0]),3), ' RMSE: ', round(np.mean(RMSE[:,0]),3), ' PSNR: ', round(np.mean(PSNR[:,0]),3))
    print('\n')
    print('############################################################### \n')
    
    # Calculate and plot the Mean OA Power Spectra
    f,sTf,sLf,sHf=calcMOAPS(predL[-1,:,:],predH[-1,:,:],Y[-1,:,:],Ao,tf-to,Ns,Nt)
      
    tim = nx*dx
    colormap=plt.cm.gist_heat
    #colormap=plt.cm.gray
    plt.figure();
    plt.grid(False)
    plt.subplot(1,2,1);plt.xlabel('x (mm)'); plt.ylabel('y (mm)'); plt.title('True image',fontsize=8);
    plt.imshow(Ptb, aspect='equal', interpolation='none', extent=(-tim/2*1e3,tim/2*1e3,-tim/2*1e3,tim/2*1e3),cmap=colormap);
    plt.subplot(1,2,2);plt.xlabel('x (mm)'); plt.title('FBMB reconstruction',fontsize=8);
    plt.imshow(Pmb, aspect='equal', interpolation='none', extent=(-tim/2*1e3,tim/2*1e3,-tim/2*1e3,tim/2*1e3),cmap=colormap);   
    plt.suptitle('Best SSIM case - Input sinogram conditions: SNR= '+str(round(SNRb,1))+'   SPE= '+str(round(POSEb,5)),fontsize=10)
    
    colormap=plt.cm.gist_heat
    #colormap=plt.cm.gray
    plt.figure();
    plt.grid(False)
    plt.subplot(1,4,1);plt.xlabel('x (mm)'); plt.ylabel('y (mm)'); plt.title('True image',fontsize=8);
    plt.imshow(Ptb, aspect='equal', interpolation='none', extent=(-tim/2*1e3,tim/2*1e3,-tim/2*1e3,tim/2*1e3),cmap=colormap);
    plt.subplot(1,4,2);plt.xlabel('x (mm)'); plt.title('Low frequency',fontsize=8);
    plt.imshow(PmLb, aspect='equal', interpolation='none', extent=(-tim/2*1e3,tim/2*1e3,-tim/2*1e3,tim/2*1e3),cmap=colormap);  
    plt.subplot(1,4,3);plt.xlabel('x (mm)');  plt.title('High frequency',fontsize=8);
    plt.imshow(PmHb, aspect='equal', interpolation='none', extent=(-tim/2*1e3,tim/2*1e3,-tim/2*1e3,tim/2*1e3),cmap=colormap);  
    plt.subplot(1,4,4);plt.xlabel('x (mm)'); plt.title('Low + High freq.',fontsize=8);
    plt.imshow(Pmb, aspect='equal', interpolation='none', extent=(-tim/2*1e3,tim/2*1e3,-tim/2*1e3,tim/2*1e3),cmap=colormap);
    plt.suptitle('Best SSIM case - Input sinogram conditions: SNR= '+str(round(SNRb,1))+'   SPE= '+str(round(POSEb,5)),fontsize=10)
    
    colormap=plt.cm.gist_heat
    #colormap=plt.cm.gray
    plt.figure();
    plt.grid(False)
    plt.subplot(1,2,1);plt.xlabel('x (mm)'); plt.ylabel('y (mm)'); plt.title('True image',fontsize=8);
    plt.imshow(Ptw, aspect='equal', interpolation='none', extent=(-tim/2*1e3,tim/2*1e3,-tim/2*1e3,tim/2*1e3),cmap=colormap);
    plt.subplot(1,2,2);plt.xlabel('x (mm)'); plt.title('FBMB reconstruction',fontsize=8);
    plt.imshow(Pmw, aspect='equal', interpolation='none', extent=(-tim/2*1e3,tim/2*1e3,-tim/2*1e3,tim/2*1e3),cmap=colormap);      
    plt.suptitle('Worst SSIM case - Input sinogram conditions: SNR= '+str(round(SNRw,1))+'   SPE= '+str(round(POSEw,5)),fontsize=10)
    
    colormap=plt.cm.gist_heat
    #colormap=plt.cm.gray
    plt.figure();
    plt.grid(False)
    plt.subplot(1,4,1);plt.xlabel('x (mm)'); plt.ylabel('y (mm)'); plt.title('True image',fontsize=8);
    plt.imshow(Ptw, aspect='equal', interpolation='none', extent=(-tim/2*1e3,tim/2*1e3,-tim/2*1e3,tim/2*1e3),cmap=colormap);
    plt.subplot(1,4,2);plt.xlabel('x (mm)'); plt.title('Low frequency',fontsize=8);
    plt.imshow(PmLw, aspect='equal', interpolation='none', extent=(-tim/2*1e3,tim/2*1e3,-tim/2*1e3,tim/2*1e3),cmap=colormap);  
    plt.subplot(1,4,3);plt.xlabel('x (mm)');  plt.title('High frequency',fontsize=8);
    plt.imshow(PmHw, aspect='equal', interpolation='none', extent=(-tim/2*1e3,tim/2*1e3,-tim/2*1e3,tim/2*1e3),cmap=colormap);  
    plt.subplot(1,4,4);plt.xlabel('x (mm)'); plt.title('Low + High freq.',fontsize=8);
    plt.imshow(Pmw, aspect='equal', interpolation='none', extent=(-tim/2*1e3,tim/2*1e3,-tim/2*1e3,tim/2*1e3),cmap=colormap);
    plt.suptitle('Worst SSIM case - Input sinogram conditions: SNR= '+str(round(SNRw,1))+'   SPE= '+str(round(POSEw,5)),fontsize=10)
    
    print('saving results...')
    mdic = {"xL": xL, "xH": xH, "SSIM": SSIM, "PC": PC, "RMSE": RMSE, "PSNR": PSNR, "TE": TE}
    savemat("predict"+fecha+'.mat',mdic)
    
    return xL, xH, SSIM, PC, RMSE, PSNR, TE
