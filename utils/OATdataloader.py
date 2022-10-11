import numpy as np
import os
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split

# ---------------------------------------------------------------------------
def gettraindata(cache_dir, n_name):
    
    print('Obtaining data for training...')
    
    #cache_dir = '../data/cache'

    X = np.load(os.path.join(cache_dir, n_name+'X.npy')) # Noisy sinogram

    Y = np.load(os.path.join(cache_dir, n_name+'Y.npy')) # True image

    #Z = np.load(os.path.join(cache_dir, n_name+'Z.npy')) # True sinogram
        
    X=X.astype(np.float32)
    Y=Y.astype(np.float32)
    #Z=Z.astype(np.float32)
    print('done')
    
    return X,Y #,Z

# ---------------------------------------------------------------------------
class OAImageDataset(Dataset):
    def __init__(self, X, Y):
        super(OAImageDataset, self).__init__()
        self.X = X
        self.Y = Y

    def __getitem__(self, item):
        return self.X[item, :, :], self.Y[item, :, :]

    def __len__(self):
        return self.X.shape[0]

# ---------------------------------------------------------------------------
def get_trainloader(X, Y, val_percent, batch_size): 
        
    dataset_train = OAImageDataset(X, Y)
    
    # Split into train / validation partitions
    n_val = int(len(dataset_train) * val_percent)
    n_train = len(dataset_train) - n_val
    train_set, val_set = random_split(dataset_train, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    
    # Create data loaders
    #loader_args = dict(batch_size=batch_size, num_workers=8, pin_memory=True) # for local uncomment this
    loader_args = dict(batch_size=batch_size, num_workers=2, pin_memory=True) # for google_colab uncomment this
    train_loader = DataLoader(train_set, shuffle=True,  drop_last=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args) #drop_last=True, drop the last batch if the dataset size is not divisible by the batch size.
    
    return train_loader, val_loader, n_train, n_val

# ---------------------------------------------------------------------------
def gettestdata(cache_dir,n_name,ntest):
    
    print('Obtaining data for testing...')

    X = np.load(os.path.join(cache_dir, n_name+'Xtest.npy')) # Noisy sinogram
    Y = np.load(os.path.join(cache_dir, n_name+'Ytest.npy')) # True image
    #Z = np.load(os.path.join(cache_dir, n_name+'Ztest.npy')) # True sinogram
    SNR = np.load(os.path.join(cache_dir, n_name+'SNRtest.npy')) # Sinogram SNR
    POSE = np.load(os.path.join(cache_dir, n_name+'POSEtest.npy')) # error position
      
    X = X[0:ntest,:,:]
    Y = Y[0:ntest,:,:]
    #Z = Z[0:ntest,:,:]
    SNR = SNR[0:ntest]
    POSE = POSE[0:ntest]
    
    X=X.astype(np.float32)
    Y=Y.astype(np.float32)
    #Z=Z.astype(np.float32)
    print('done')
    
    return X,Y,SNR,POSE #,Z

# ---------------------------------------------------------------------------
def gettraindata2(cache_dir, n_name):
    
    print('Obtaining data for training...')
    
    #cache_dir = '../data/cache'

    X = np.load(os.path.join(cache_dir, n_name+'X.npy')) # Noisy Image

    Y = np.load(os.path.join(cache_dir, n_name+'Y.npy')) # True image

    #Z = np.load(os.path.join(cache_dir, n_name+'Z.npy')) # Noisy sinogram
        
    X=X.astype(np.float32)
    Y=Y.astype(np.float32)
    #Z=Z.astype(np.float32)
    print('done')
    
    return X,Y #,Z

# ---------------------------------------------------------------------------
def gettestdata2(cache_dir,n_name,ntest):
    
    print('Obtaining data for testing...')

    X = np.load(os.path.join(cache_dir, n_name+'Xtest.npy')) # Noisy Image
    Y = np.load(os.path.join(cache_dir, n_name+'Ytest.npy')) # True image
    Z = np.load(os.path.join(cache_dir, n_name+'Ztest.npy')) # Noisy sinogram
    SNR = np.load(os.path.join(cache_dir, n_name+'SNRtest.npy')) # Sinogram SNR
      
    X = X[0:ntest,:,:]
    Y = Y[0:ntest,:,:]
    Z = Z[0:ntest,:,:]
    SNR = SNR[0:ntest]
    
    X=X.astype(np.float32)
    Y=Y.astype(np.float32)
    Z=Z.astype(np.float32)
    print('done')
    
    return X,Y,Z,SNR
