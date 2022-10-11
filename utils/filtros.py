import numpy as np
from scipy import signal
from scipy.linalg import convolution_matrix
import torch

# ---------------------------------------------------------------------------
def CreateFilterMatrix(Nt,dx,nx,dsa,arco,vs,to,tf):
    
    t = np.linspace(to, tf, Nt) # (Nt,)
    
    # FILTER DESIGN
    Fs = 1/(t[1]-t[0]) # Sampling frequency of the original time grid
    filOrd = 4 # filter order
    
    # Low-pass Filter 
    flc = 1.23e6 # cutoff frequency [Hz]
    bl, al = signal.butter(filOrd, flc, 'low', fs = Fs)
    
    # Band-pass Filter design BW% = 1.7 and we are assuming a cutoff freq of 0.1 MHz
    fbc1 = 1.23e6#1.25e6 # cutoff frequency [Hz]
    fbc2 = 15.2e6 # cutoff frequency [Hz]
    bb, ab = signal.butter(filOrd, (fbc1, fbc2), 'bandpass', fs = Fs)
    
    # FILTER IMPULSE RESPONSE
    po=np.zeros(Nt,)
    po[int(Nt/2)]=1
    impL = signal.filtfilt(bl,al,po)
    impH = signal.filtfilt(bb,ab,po)
    
    # GENERATE FILTER MATRIX
    FL=convolution_matrix(impL,Nt,'same')
    FH=convolution_matrix(impH,Nt,'same')
        
    # Complementary
    I=np.eye(Nt)
    FL = I - FL
    FH = I - FH
    
    return FL, FH

# ---------------------------------------------------------------------------
def CreateFilterMatrix2(Nt,dx,nx,dsa,arco,vs,to,tf):
    
    t = np.linspace(to, tf, Nt) # (Nt,)
    
    # FILTER DESIGN
    Fs = 1/(t[1]-t[0]) # Sampling frequency of the original time grid
    filOrd = 8#4 # filter order
    
    # Low-pass Filter 
    flc = 1.23e6 # cutoff frequency [Hz]
    bl, al = signal.butter(filOrd, flc, 'low', fs = Fs)
    
    # Band-pass Filter design BW% = 1.7 and we are assuming a cutoff freq of 0.1 MHz
    fbc1 = 1.23e6#1.25e6 # cutoff frequency [Hz]
    fbc2 = 15.2e6 # cutoff frequency [Hz]
    bb, ab = signal.butter(filOrd, (fbc1, fbc2), 'bandpass', fs = Fs)
    
    # FILTER IMPULSE RESPONSE
    po=np.zeros(Nt,)
    po[int(Nt/2)]=1
    impL = signal.lfilter(bl,al,po) # ---> acá está lo distinto!
    impH = signal.lfilter(bb,ab,po) # ---> acá está lo distinto!
    
    # GENERATE FILTER MATRIX
    FL=convolution_matrix(impL,Nt,'same')
    FH=convolution_matrix(impH,Nt,'same')
        
    # Complementary
    I=np.eye(Nt)
    FL = I - FL
    FH = I - FH
    
    return FL, FH

# ---------------------------------------------------------------------------
def applyFilter(s, F, device):
    s = torch.squeeze(s,1) # (-1,Ns,Nt)
    dimS = s.shape # (-1,Ns,Nt)
    sf = torch.zeros(dimS) # (-1,Ns,Nt)
    for i1 in range(0,dimS[1]):
        sf[:,i1,:] = torch.matmul(F,s[:,i1,:].T).T # # (-1,Ns,Nt)
    sf = torch.unsqueeze(sf,1) # (-1,1,Ns,Nt)
    sf = sf.to(device=device) 
    
    return sf