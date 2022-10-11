import os
import numpy as np

tfbfdunet =  False # Train FB-FDUNet?
tfdunet =  False # Train FDUNet?
pfbfdunet = False # predict with FB-FDUNet?
pfdunet = False # predict with FDUNet?
pfbmb = False # predict with FB-MB
t_name = 'dataset' # filename dataset for training and validation
p_name = 'dataset_test' # filename dataset for testing

if tfbfdunet:
	from fbfdunet_2bands.train2bands import train_fbfdunetln
	os.chdir('./fbfdunet_2bands')

	batch_size = 2
	epochs = 100
	continuetrain = False
	plotresults = True
	WandB = False
	fecha = '51022' 

	epoch,trainloss,valloss = train_fbfdunetln(t_name,batch_size,epochs,continuetrain,plotresults,WandB,fecha)
	os.chdir('..')

if tfbfdunet:
	from fdunetln.train import train_fdunet
	os.chdir('./fdunetln')

	batch_size = 2
	epochs = 100
	continuetrain = False
	plotresults = True
	WandB = False
	fecha = '51022' 

	epoch,trainloss,valloss = train_fdunet(t_name,batch_size,epochs,continuetrain,plotresults,WandB,fecha)
	os.chdir('..')

if pfbfdunet:
	from fbfdunet_2bands.predict2bands import preditc_fbfdunetln
	os.chdir('./fbfdunet_2bands')

	ckp_best='fbfdunetln_best' + fecha + '.pth'  # name of the file of the saved weights of the trained net 

	predict_fbfdunetln(n_name,ntest,ckp_best)
	os.chdir('..')

if pfdunet:
	from fdunetln.predict import predict_fdunet
	os.chdir('./fdunetln')

	ckp_best='fdunetln_best' + fecha + '.pth'  # name of the file of the saved weights of the trained net 

	predict_fdunet(n_name,ntest,ckp_best)
	os.chdir('..')

if pfbmb:
	from fbmb.predict_fbmb import predictfbmb
	os.chdir('./fbmb')

	predictfbmb(n_name,ntest)
	os.chdir('..')
