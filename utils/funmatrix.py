import numpy as np
import torch
from utils.model_based_matrix import build_matrix, SensorMaskCartCircleArc   

# ---------------------------------------------------------------------------
def createForwMat(Ns,Nt,dx,nx,dsa,arco,vs,to,tf): # 
    print('Creating Forward Model-based Matrix')
    
    t = np.linspace(to, tf, Nt) # time grid
    posSens = SensorMaskCartCircleArc(dsa,arco,Ns) # position of the center of the detectors (3,Ns) [m]
    Ao = build_matrix(nx,dx,Ns,posSens,1,1,vs,t,True,True,True,True,tlp=2*dx/vs)
    Ao=Ao.astype(np.float32)
    print('done')
    
    return Ao

# ---------------------------------------------------------------------------
def applyInvMat(x, Ao, dimS, dimI): # [Ao] = (Ns*Nt,nx*nx)
    x = torch.squeeze(x,1) # (-1,Ns,Nt)
    x = torch.reshape(x,(dimS[0],int(dimS[2]*dimS[3]))) # (-1,Ns*Nt)
    y = torch.matmul(Ao.T,x.T).T # ((nx*nx,Ns*Nt) @ (Ns*Nt,-1)).T = (-1,nx*nx)
    y = torch.reshape(y,(dimI[0],dimI[2],dimI[3])) # (-1,nx,nx)
    y = torch.unsqueeze(y,1) # (-1,1,nx,nx)
    
    return y

# ---------------------------------------------------------------------------
def applyForwMat(y, Ao, dimS, dimI):
    y = torch.squeeze(y,1) # (-1,nx,nx)
    y = torch.reshape(y,(dimI[0],int(dimI[2]*dimI[3]))) # (-1,nx*nx)
    x = torch.matmul(Ao,y.T).T # ((Ns*Nt,nx*nx) @ (nx*nx,-1)).T = (-1,Ns*Nt)
    x = torch.reshape(x,(dimS[0],dimS[2],dimS[3])) # (-1,Ns,Nt)
    x = torch.unsqueeze(x,1) # (-1,1,Ns,Nt)
        
    return x