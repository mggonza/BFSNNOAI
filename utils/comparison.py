import numpy as np
from tqdm import tqdm # progress bar for loops
import matplotlib.pyplot as plt

from skimage.color import rgb2gray#, rgba2rgb
from skimage import io 
from scipy.fft import fft

import math
from scipy import stats
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio 

from utils.model_based_matrix import build_matrix, SensorMaskCartCircleArc
from utils.dasandubp import DAS

# ---------------------------------------------------------------------------
def applyDAS(Ns,Nt,dx,nx,dsa,arco,vs,to,tf,p): #    
    t = np.linspace(to, tf, Nt) # time grid
    posSens = SensorMaskCartCircleArc(dsa,arco,Ns) # position of the center of the detectors (3,Ns) [m]
    Pdas = DAS(nx,dx,dsa,posSens,vs,t,p)
    
    return Pdas

# ---------------------------------------------------------------------------
def comparisonSOTA(Ns,Nt,dx,nx,dsa,arco,vs,to,tf,Ao,X,Y,predL,predH,SNR,POSE,cache_dir,dibujopaper):
        
    SSIM=np.zeros((X.shape[0],3))
    PC=np.zeros((X.shape[0],3))
    RMSE=np.zeros((X.shape[0],3))
    PSNR=np.zeros((X.shape[0],3))
        
    Best=0
    Worst=1
    
    print('Calculating metrics and doing comparison with DAS and LBP...')
    pred = predL + predH
    for i1 in tqdm(range(0,X.shape[0])):
        trueimage=Y[i1,:,:].astype(np.float32);
        
        predimage=pred[i1,:,:].astype(np.float32); valnorm = np.max(np.abs(predimage.ravel())); predimage=predimage/valnorm;
        predimageL=predL[i1,:,:].astype(np.float32); predimageL=predimageL/np.max(np.abs(predimageL));
        predimageH=predH[i1,:,:].astype(np.float32); predimageH=predimageH/np.max(np.abs(predimageH));
        SSIM[i1,0]=structural_similarity(trueimage,predimage) 
        PC[i1,0]=stats.pearsonr(trueimage.ravel(),predimage.ravel())[0]  
        RMSE[i1,0]=math.sqrt(mean_squared_error(trueimage,predimage))
        PSNR[i1,0]=peak_signal_noise_ratio(trueimage,predimage)
        # LBP
        Plbp = Ao.T@X[i1,:,:].ravel(); 
        Plbp=Plbp/np.max(np.abs(Plbp)); Plbp=np.reshape(Plbp,(nx,nx)); Plbp=Plbp.astype(np.float32)
        SSIM[i1,1]=structural_similarity(trueimage,Plbp) 
        PC[i1,1]=stats.pearsonr(trueimage.ravel(),Plbp.ravel())[0]  
        RMSE[i1,1]=math.sqrt(mean_squared_error(trueimage,Plbp))
        PSNR[i1,1]=peak_signal_noise_ratio(trueimage,Plbp)
        # DAS
        Pdas = applyDAS(Ns,Nt,dx,nx,dsa,arco,vs,to,tf,X[i1,:,:]); 
        Pdas=Pdas/np.max(np.abs(Pdas)); Pdas=np.reshape(Pdas,(nx,nx)); Pdas=Pdas.astype(np.float32)
        SSIM[i1,2]=structural_similarity(trueimage,Pdas) 
        PC[i1,2]=stats.pearsonr(trueimage.ravel(),Pdas.ravel())[0]  
        RMSE[i1,2]=math.sqrt(mean_squared_error(trueimage,Pdas))
        PSNR[i1,2]=peak_signal_noise_ratio(trueimage,Pdas)
        if SSIM[i1,0]>Best:
            Best = SSIM[i1,0]
            SNRb=SNR[i1]
            POSEb=POSE[i1]
            Ptb = trueimage
            Pnetb=predimage; PnetLb=predimageL;  PnetHb=predimageH;
            Pdasb=Pdas
            Plbpb=Plbp
        if SSIM[i1,0]<Worst:
            Worst = SSIM[i1,0]
            SNRw=SNR[i1]
            POSEw=POSE[i1]
            Ptw = trueimage
            Pnetw=predimage; PnetLw=predimageL;  PnetHw=predimageH; 
            Pdasw=Pdas
            Plbpw=Plbp
    
    print('\n')
    print('############################################################### \n')
    print('Metrics results NET: \n', 'SSIM: ',round(np.mean(SSIM[:,0]),3), ' PC: ', round(np.mean(PC[:,0]),3), ' RMSE: ', round(np.mean(RMSE[:,0]),3), ' PSNR: ', round(np.mean(PSNR[:,0]),3))
    print('Metrics results LBP: \n', 'SSIM: ',round(np.mean(SSIM[:,1]),3), ' PC: ', round(np.mean(PC[:,1]),3), ' RMSE: ', round(np.mean(RMSE[:,1]),3), ' PSNR: ', round(np.mean(PSNR[:,1]),3))
    print('Metrics results DAS: \n', 'SSIM: ',round(np.mean(SSIM[:,2]),3), ' PC: ', round(np.mean(PC[:,2]),3), ' RMSE: ', round(np.mean(RMSE[:,2]),3), ' PSNR: ', round(np.mean(PSNR[:,2]),3))
    print('\n')
    print('############################################################### \n')
    
    tim = nx*dx
    
    colormap=plt.cm.gist_heat
    #colormap=plt.cm.gray
    plt.figure();
    plt.grid(False)
    plt.subplot(1,4,1);plt.xlabel('x (mm)'); plt.ylabel('y (mm)'); plt.title('True image',fontsize=8);
    plt.imshow(Ptb, aspect='equal', interpolation='none', extent=(-tim/2*1e3,tim/2*1e3,-tim/2*1e3,tim/2*1e3),cmap=colormap);
    plt.subplot(1,4,2);plt.xlabel('x (mm)'); plt.title('DAS reconstruction',fontsize=8);
    plt.imshow(Pdasb, aspect='equal', interpolation='none', extent=(-tim/2*1e3,tim/2*1e3,-tim/2*1e3,tim/2*1e3),cmap=colormap);  
    plt.subplot(1,4,3);plt.xlabel('x (mm)');  plt.title('LBP reconstruction',fontsize=8);
    plt.imshow(Plbpb, aspect='equal', interpolation='none', extent=(-tim/2*1e3,tim/2*1e3,-tim/2*1e3,tim/2*1e3),cmap=colormap);  
    plt.subplot(1,4,4);plt.xlabel('x (mm)'); plt.title('predicted image',fontsize=8);
    plt.imshow(Pnetb, aspect='equal', interpolation='none', extent=(-tim/2*1e3,tim/2*1e3,-tim/2*1e3,tim/2*1e3),cmap=colormap);    
    plt.suptitle('Best SSIM case - Input sinogram conditions: SNR= '+str(round(SNRb,1))+'   SPE= '+str(round(POSEb,5)),fontsize=10)
    
    colormap=plt.cm.gist_heat
    #colormap=plt.cm.gray
    plt.figure();
    plt.grid(False)
    plt.subplot(1,4,1);plt.xlabel('x (mm)'); plt.ylabel('y (mm)'); plt.title('True image',fontsize=8);
    plt.imshow(Ptb, aspect='equal', interpolation='none', extent=(-tim/2*1e3,tim/2*1e3,-tim/2*1e3,tim/2*1e3),cmap=colormap);
    plt.subplot(1,4,2);plt.xlabel('x (mm)'); plt.title('Low frequency',fontsize=8);
    plt.imshow(PnetLb, aspect='equal', interpolation='none', extent=(-tim/2*1e3,tim/2*1e3,-tim/2*1e3,tim/2*1e3),cmap=colormap);  
    plt.subplot(1,4,3);plt.xlabel('x (mm)');  plt.title('High frequency',fontsize=8);
    plt.imshow(PnetHb, aspect='equal', interpolation='none', extent=(-tim/2*1e3,tim/2*1e3,-tim/2*1e3,tim/2*1e3),cmap=colormap);  
    plt.subplot(1,4,4);plt.xlabel('x (mm)'); plt.title('Low + High freq.',fontsize=8);
    plt.imshow(Pnetb, aspect='equal', interpolation='none', extent=(-tim/2*1e3,tim/2*1e3,-tim/2*1e3,tim/2*1e3),cmap=colormap);
    plt.suptitle('Best SSIM case - Input sinogram conditions: SNR= '+str(round(SNRb,1))+'   SPE= '+str(round(POSEb,5)),fontsize=10)
    
    colormap=plt.cm.gist_heat
    #colormap=plt.cm.gray
    plt.figure();
    plt.grid(False)
    plt.subplot(1,4,1);plt.xlabel('x (mm)'); plt.ylabel('y (mm)'); plt.title('True image',fontsize=8);
    plt.imshow(Ptw, aspect='equal', interpolation='none', extent=(-tim/2*1e3,tim/2*1e3,-tim/2*1e3,tim/2*1e3),cmap=colormap);
    plt.subplot(1,4,2);plt.xlabel('x (mm)'); plt.title('DAS reconstruction',fontsize=8);
    plt.imshow(Pdasw, aspect='equal', interpolation='none', extent=(-tim/2*1e3,tim/2*1e3,-tim/2*1e3,tim/2*1e3),cmap=colormap);  
    plt.subplot(1,4,3);plt.xlabel('x (mm)');  plt.title('LBP reconstruction',fontsize=8);
    plt.imshow(Plbpw, aspect='equal', interpolation='none', extent=(-tim/2*1e3,tim/2*1e3,-tim/2*1e3,tim/2*1e3),cmap=colormap);  
    plt.subplot(1,4,4);plt.xlabel('x (mm)'); plt.title('predicted image',fontsize=8);
    plt.imshow(Pnetw, aspect='equal', interpolation='none', extent=(-tim/2*1e3,tim/2*1e3,-tim/2*1e3,tim/2*1e3),cmap=colormap);    
    plt.suptitle('Worst SSIM case - Input sinogram conditions: SNR= '+str(round(SNRw,1))+'   SPE= '+str(round(POSEw,5)),fontsize=10)
    
    colormap=plt.cm.gist_heat
    #colormap=plt.cm.gray
    plt.figure();
    plt.grid(False)
    plt.subplot(1,4,1);plt.xlabel('x (mm)'); plt.ylabel('y (mm)'); plt.title('True image',fontsize=8);
    plt.imshow(Ptw, aspect='equal', interpolation='none', extent=(-tim/2*1e3,tim/2*1e3,-tim/2*1e3,tim/2*1e3),cmap=colormap);
    plt.subplot(1,4,2);plt.xlabel('x (mm)'); plt.title('Low frequency',fontsize=8);
    plt.imshow(PnetLw, aspect='equal', interpolation='none', extent=(-tim/2*1e3,tim/2*1e3,-tim/2*1e3,tim/2*1e3),cmap=colormap);  
    plt.subplot(1,4,3);plt.xlabel('x (mm)');  plt.title('High frequency',fontsize=8);
    plt.imshow(PnetHw, aspect='equal', interpolation='none', extent=(-tim/2*1e3,tim/2*1e3,-tim/2*1e3,tim/2*1e3),cmap=colormap);  
    plt.subplot(1,4,4);plt.xlabel('x (mm)'); plt.title('Low + High freq.',fontsize=8);
    plt.imshow(Pnetw, aspect='equal', interpolation='none', extent=(-tim/2*1e3,tim/2*1e3,-tim/2*1e3,tim/2*1e3),cmap=colormap);
    plt.suptitle('Worst SSIM case - Input sinogram conditions: SNR= '+str(round(SNRw,1))+'   SPE= '+str(round(POSEw,5)),fontsize=10)
    
    
    if dibujopaper:
        colormap=plt.cm.gist_heat
        #colormap=plt.cm.gray
        plt.figure();
        plt.grid(False)
        plt.subplot(1,4,1);plt.xlabel('x (mm)'); plt.ylabel('y (mm)'); plt.title('True image',fontsize=8);
        plt.imshow(trueimage, aspect='equal', interpolation='none', extent=(-tim/2*1e3,tim/2*1e3,-tim/2*1e3,tim/2*1e3),cmap=colormap);
        plt.subplot(1,4,2);plt.xlabel('x (mm)'); plt.title('DAS reconstruction',fontsize=8);
        plt.imshow(Pdas, aspect='equal', interpolation='none', extent=(-tim/2*1e3,tim/2*1e3,-tim/2*1e3,tim/2*1e3),cmap=colormap);  
        plt.subplot(1,4,3);plt.xlabel('x (mm)');  plt.title('LBP reconstruction',fontsize=8);
        plt.imshow(Plbp, aspect='equal', interpolation='none', extent=(-tim/2*1e3,tim/2*1e3,-tim/2*1e3,tim/2*1e3),cmap=colormap);  
        plt.subplot(1,4,4);plt.xlabel('x (mm)'); plt.title('predicted image',fontsize=8);
        plt.imshow(predimage, aspect='equal', interpolation='none', extent=(-tim/2*1e3,tim/2*1e3,-tim/2*1e3,tim/2*1e3),cmap=colormap);    
        plt.suptitle('Paper image - Input sinogram conditions: SNR= '+str(round(SNR[-1],1))+'   SPE= '+str(round(POSE[-1],5)),fontsize=10)
    
        colormap=plt.cm.gist_heat
        #colormap=plt.cm.gray
        plt.figure();
        plt.grid(False)
        plt.subplot(1,4,1);plt.xlabel('x (mm)'); plt.ylabel('y (mm)'); plt.title('True image',fontsize=8);
        plt.imshow(Ptw, aspect='equal', interpolation='none', extent=(-tim/2*1e3,tim/2*1e3,-tim/2*1e3,tim/2*1e3),cmap=colormap);
        plt.subplot(1,4,2);plt.xlabel('x (mm)'); plt.title('Low frequency',fontsize=8);
        plt.imshow(predimageL, aspect='equal', interpolation='none', extent=(-tim/2*1e3,tim/2*1e3,-tim/2*1e3,tim/2*1e3),cmap=colormap);  
        plt.subplot(1,4,3);plt.xlabel('x (mm)');  plt.title('High frequency',fontsize=8);
        plt.imshow(predimageH, aspect='equal', interpolation='none', extent=(-tim/2*1e3,tim/2*1e3,-tim/2*1e3,tim/2*1e3),cmap=colormap);  
        plt.subplot(1,4,4);plt.xlabel('x (mm)'); plt.title('Low + High freq.',fontsize=8);
        plt.imshow(predimage, aspect='equal', interpolation='none', extent=(-tim/2*1e3,tim/2*1e3,-tim/2*1e3,tim/2*1e3),cmap=colormap);
        plt.suptitle('Paper image - Input sinogram conditions: SNR= '+str(round(SNR[-1],1))+'   SPE= '+str(round(POSE[-1],5)),fontsize=10)
    
    
    return SSIM,PC,RMSE,PSNR

# ---------------------------------------------------------------------------
def calcMOAPS(imgL,imgH,imgT,Ao,T,Ns,Nt):
    # Calculate and plot the Mean OA Power Spectra
    Fs = Nt/T # [Hz]
    f=np.arange(0,Nt)/Nt*Fs # [Hz]
    sT = Ao@imgT.ravel()
    sL = Ao@imgL.ravel()
    sH = Ao@imgH.ravel()
    sT = np.reshape(sT,(Ns,Nt))
    sL = np.reshape(sL,(Ns,Nt))
    sH = np.reshape(sH,(Ns,Nt))
    sTf = np.mean(np.abs(fft(sT,axis=1))**2,axis=0)
    sLf = np.mean(np.abs(fft(sL,axis=1))**2,axis=0)
    sHf = np.mean(np.abs(fft(sH,axis=1))**2,axis=0)
    nv = np.max(sTf.ravel())
    nv2 = np.max((sLf+sHf).ravel())
    sTf = sTf/nv; sLf = sLf/nv2; sHf = sHf/nv2; 
    
    plt.figure();
    plt.grid(True,linestyle='--')
    plt.xlabel('f (MHz)'); plt.ylabel('Mean OA power spectra')
    plt.plot(f/1e6,sTf,'-',linewidth=1.5,label='Full BW'),
    plt.plot(f/1e6,sLf,'-',linewidth=1.5,label='Band 1'),
    plt.plot(f/1e6,sHf,'-',linewidth=1.5,label='Band 2')
    #plt.xlim([-0.5,Fs/2/1e6])
    plt.xlim([-0.5,15])
    plt.ylim([-0.1,1.1])
    plt.legend(loc='best',shadow=True, fontsize='x-large')
    
    return f,sTf,sLf,sHf
