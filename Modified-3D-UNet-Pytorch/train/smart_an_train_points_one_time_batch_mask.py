# from kidney_dataloader import KidneyDataset
from kidney_dataloader_1_c import KidneyDataset_1_c
from kidney_loss_factory import SoftDiceLoss
from torch.autograd import Variable
import evaluate_tools
import sim_clicks_batch_mask
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
from model import Modified3DUNet
import paths
import pdb
import numpy as np
# ********************

csv_train_path = '../train_0122_20.csv'
csv_val_path = '../val_0122_20.csv'
# csv_train_path = '../train_new.csv'
# csv_val_path = '../val_new.csv'

# one channel dataloader for simulate clicks
kidney_data =  KidneyDataset_1_c(csv_train_path)
val_data =  KidneyDataset_1_c(csv_val_path)
train_loader_init = DataLoader(kidney_data, batch_size = 8, shuffle=False, num_workers=4)
val_loader_init = DataLoader(val_data, batch_size = 8, shuffle=False, num_workers=4)

# GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# one channel pretrained model to simulate a initial click
in_channels_init = 1
n_classes_init = 1
base_n_filter_init = 16
modelpath_init = '/data/ccusr/xinyug/annotation/Modified-3D-UNet-Pytorch/checkpoints/softdice_1c/checkpoint_71_20191225_1636.pth.tar'
model_1_c = Modified3DUNet(in_channels_init, n_classes_init, base_n_filter_init).to(device)
model_init = evaluate_tools.load_checkpoint_with_date(model_1_c, modelpath_init)

# simulate initial clicks based on a trained 3-channel input mdoel
print('simulate initial clicks ...')
sigma = 10
clicks_info_init = sim_clicks_batch_mask.sim_initial_click_dataset_batch(train_loader_init, model_init, sigma, device)
clicks_info_val_init = sim_clicks_batch_mask.sim_initial_click_dataset_batch(val_loader_init, model_init, sigma, device)

# train
train_loader = DataLoader(kidney_data, batch_size = 4, shuffle=True, num_workers=4)
val_loader = DataLoader(val_data, batch_size = 2, shuffle = True, num_workers = 4)

def datestr():
	now = time.localtime()
	return '{:04}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)

curtime = datestr()

# logging train and val loss
train_loss_l = []
val_loss_l = []
log_save_dir = '/data/ccusr/xinyug/annotation/Modified-3D-UNet-Pytorch/results/softdice_sim_points_one_time_batch_mask'
if not os.path.exists(log_save_dir):
    os.mkdir(log_save_dir)

# model save path
check_points_save = '/data/ccusr/xinyug/annotation/Modified-3D-UNet-Pytorch/checkpoints/softdice_sim_points_one_time_batch_mask'
if not os.path.exists(check_points_save):
    os.mkdir(check_points_save)

# model  image + foreground + background
in_channels = 4
n_classes = 1
base_n_filter = 16
model = Modified3DUNet(in_channels, n_classes, base_n_filter).to(device)
criterion = SoftDiceLoss(0.000001)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# training
running_loss = 0
val_loss_best = 65535
num_points = 3
sim_epochs = 1

print('simulate clicks ...')
clicks_info_train = sim_clicks_batch_mask.sim_clicks_dataset_batch(train_loader,clicks_info_init,model,num_points,sigma,device)
clicks_info_val_epoch = sim_clicks_batch_mask.sim_clicks_dataset_batch(val_loader,clicks_info_val_init,model,num_points,sigma,device)

print('train model...')
for i in range(100000):
    print('**** epoch ',i,' ****')
    running_loss = 0
    for train_batch in tqdm(train_loader): 
        train_data_1_c = train_batch['data'].to(device)
        train_data_1_c = torch.unsqueeze(train_data_1_c,1)
        data_keys = train_batch['kidney_id'] 
        t_arr_shape = train_batch['mask'].shape[1:4]
        train_maps = torch.tensor(sim_clicks_batch_mask.get_prob_maps(data_keys,clicks_info_train,t_arr_shape,sigma)).to(device)
        train_data_1_c_mask = train_batch['pred_mask'].to(device)
        train_data_1_c_mask = torch.unsqueeze(train_data_1_c_mask,1)
        train_data = torch.cat((train_data_1_c, train_maps, train_data_1_c_mask),1)
        # print(train_data.shape)
        # pdb.set_trace()
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

    print('validate model ...')
    if i % 1 == 0:
        model.eval() # model in evaluation mode
        run_val_loss = 0
        for val_batch in tqdm(val_loader):
            val_data_1_c = val_batch['data'].to(device)
            val_data_1_c = torch.unsqueeze(val_data_1_c,1)
            data_keys_val = val_batch['kidney_id']
            val_maps = torch.tensor(sim_clicks_batch_mask.get_prob_maps(data_keys_val, clicks_info_val_epoch,\
                                    val_batch['mask'].shape[1:4],sigma)).to(device)
            val_data_1_c_mask = val_batch['pred_mask'].to(device)
            val_data_1_c_mask = torch.unsqueeze(val_data_1_c_mask,1)  

            val_data = torch.cat((val_data_1_c, val_maps, val_data_1_c_mask),1)
            val_label = val_batch['mask'].view(1,-1).squeeze(0).to(device)
            output_val = model(val_data)
            logits_val = output_val[0]
            val_loss = criterion(logits_val, val_label)
            run_val_loss+=val_loss.item()
        val_loss_final = run_val_loss/len(val_loader) 
        val_loss_l.append(val_loss_final)
        print('**** val loss is : %.3f ******'%(val_loss_final))
        
        # save model
        if val_loss_final < val_loss_best:
            val_loss_best = val_loss_final
            print('**** best val loss is : %.3f ******'%(val_loss_best))
            evaluate_tools.save_checkpoint_with_date(model, val_loss_best, i, curtime, check_points_save)
            print ('********** model saved !************')
        evaluate_tools.save_loss_csv(train_loss_l, val_loss_l, log_save_dir)
        model.train()