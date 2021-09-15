from kidney_dataloader_1_c_aug import KidneyDataset
from kidney_loss_factory import SoftDiceLoss
from torch.autograd import Variable
import evaluate_tools
# ******************
import os
import shutil
import time
import logging

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

# from model_2 import Modified3DUNet
from model import Modified3DUNet
# from Unet_3D import unet_3D
import paths
# import pdb
# ********************

csv_train_path = '../train_aug.csv'
csv_val_path ='../val_weighted_dice_2.csv'
kidney_dataset_train = KidneyDataset(csv_train_path)
kidney_dataset_val = KidneyDataset(csv_val_path)

train_loader  = DataLoader(kidney_dataset_train, batch_size = 4, shuffle=True, num_workers=4)
val_loader = DataLoader(kidney_dataset_val, batch_size = 4, shuffle = True, num_workers = 2)

def datestr():
	now = time.localtime()
	return '{:04}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)

curtime = datestr()

# logging train and val loss
train_loss_l = []
val_loss_l = []
log_save_dir = '/data/ccusr/xinyug/annotation/Modified-3D-UNet-Pytorch/results/softdice_1c_aug'
if not os.path.exists(log_save_dir):
    os.mkdir(log_save_dir)

# 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Loading the model
in_channels = 1
n_classes = 1
base_n_filter = 16

# train a new model
model = Modified3DUNet(in_channels, n_classes, base_n_filter).to(device)

# load existing model
# modelpath = '/data/ccusr/xinyug/annotation/Modified-3D-UNet-Pytorch/checkpoints/bce/checkpoint_8_20191216_1551.pth.tar' # lr 0.001
# model = Modified3DUNet(in_channels, n_classes, base_n_filter).to(device)
# model = evaluate_tools.load_checkpoint_with_date(model, modelpath)

# criterion = nn.CrossEntropyLoss(weight=weight)
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# criterion = nn.BCEWithLogitsLoss()
criterion = SoftDiceLoss(0.000001)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
check_points_save = '/data/ccusr/xinyug/annotation/Modified-3D-UNet-Pytorch/checkpoints/softdice_1c_aug'
if not os.path.exists(check_points_save):
    os.mkdir(check_points_save)

# training 
running_loss = 0
val_loss_best = 65535
model.train()
for i in range(100000):
    print('**** epoch ',i,' ****')
    running_loss = 0
    for train_batch in tqdm(train_loader):
        train_data = train_batch['data'].to(device) 
        train_data = torch.unsqueeze(train_data,1)
        train_label = train_batch['mask'].to(device)
        train_label = train_label.view(1,-1).squeeze(0)
        optimizer.zero_grad()
        output = model(train_data)
        output = output[0]
        loss = criterion(output, train_label)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
    print('training loss : %.3f'%(running_loss/len(train_loader)))
    train_loss_l.append(running_loss/len(train_loader))
    
    if i % 1 == 0 : # generate val loss and save model 
        model.eval() # model in evaluation mode
        run_val_loss = 0
        for val_batch in tqdm(val_loader):               
            val_data = val_batch['data'].to(device) 
            val_data = torch.unsqueeze(val_data,1)
            val_label = val_batch['mask'].view(1,-1).squeeze(0).to(device)
            # train_label = train_label.view(1,-1).squeeze(0)
            output_val = model(val_data)
            logits_val = output_val[0]
            val_loss = criterion(logits_val, val_label)
            run_val_loss+=val_loss.item()
        val_loss_final = run_val_loss/len(val_loader)    
        val_loss_l.append(val_loss_final)

        # save model
        if val_loss_final < val_loss_best:
            val_loss_best = val_loss_final
            print('**** best val loss is : %.3f ******'%(val_loss_best))
            evaluate_tools.save_checkpoint_with_date(model, loss, i, curtime, check_points_save)
            print ('********** model saved !************')
        
        evaluate_tools.save_loss_csv(train_loss_l, val_loss_l, log_save_dir)
        model.train()