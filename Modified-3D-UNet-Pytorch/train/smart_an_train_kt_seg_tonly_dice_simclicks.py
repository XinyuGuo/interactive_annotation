from kidney_dataloader_1_c_aug_2_classes import KidneyDataset_standardization
from kidney_loss_factory import SoftDiceLoss_MultiClasses
import evaluate_tools
import sim_clicks_batch_tumor
from model import Modified3DUNet
# ******************
import os
import shutil
import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import paths
import numpy as np
import pdb
# ********************

# csv_train_path = '../train_multiC_aug.csv'
# csv_val_path ='../val_weighted_dice_2.csv'
csv_train_path = '../train_tumor_aug.csv'
csv_val_path = '../val_tumor.csv'
# csv_train_path = '../train_multiC_aug_test.csv'
# csv_val_path = '../val_multiC_aug_test.csv'

# one channel dataloader for simulate clicks
kidney_data = KidneyDataset_standardization(csv_train_path)
val_data =  KidneyDataset_standardization(csv_val_path)
train_loader_init = DataLoader(kidney_data, batch_size = 6, shuffle=False, num_workers=4)
val_loader_init = DataLoader(val_data, batch_size = 6, shuffle=False, num_workers=4)

# GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# one channel pretrained model to simulate a initial click
in_channels_init = 1
n_classes_init = 3
base_n_filter_init = 16
modelpath_init = '/data/ccusr/xinyug/annotation/Modified-3D-UNet-Pytorch/checkpoints/models/checkpoint_dice_aug_stand_8505.pth.tar'
model_1_c = Modified3DUNet(in_channels_init, n_classes_init, base_n_filter_init).to(device)
model_init = evaluate_tools.load_checkpoint_with_date(model_1_c, modelpath_init)

# simulate initial clicks based on a trained 3-channel input mdoel
# print('simulate initial clicks ...')
sigma = 10
clicks_info_init = sim_clicks_batch_tumor.sim_initial_click_dataset_batch(dataloader=train_loader_init, model=model_init,\
                                                                           sigma=sigma, device=device)
# print(clicks_info_init)
clicks_info_val_init = sim_clicks_batch_tumor.sim_initial_click_dataset_batch(dataloader=val_loader_init, model=model_init,\
                                                                           sigma=sigma, device=device)

# pdb.set_trace()


# train
train_loader = DataLoader(kidney_data, batch_size = 6, shuffle=True, num_workers=4)
val_loader = DataLoader(val_data, batch_size = 6, shuffle = True, num_workers = 4)

def datestr():
	now = time.localtime()
	return '{:04}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)

curtime = datestr()

# logging train and val loss
log_save_dir = '/data/ccusr/xinyug/annotation/Modified-3D-UNet-Pytorch/results/softdice_clicks_finetune_withtumor'
if not os.path.exists(log_save_dir):
    os.mkdir(log_save_dir)

train_loss_file_name = 'train_loss.csv'
val_loss_file_name = 'val_loss.csv'
train_csv_writer = evaluate_tools.get_csv_writer(log_save_dir,train_loss_file_name)
val_csv_writer = evaluate_tools.get_csv_writer(log_save_dir,val_loss_file_name)

# model save path
check_points_save = '/data/ccusr/xinyug/annotation/Modified-3D-UNet-Pytorch/checkpoints/softdice_clicks_finetune_withtumor'
if not os.path.exists(check_points_save):
    os.mkdir(check_points_save)

# model  image + foreground + background
in_channels = 3
n_classes = 3
base_n_filter = 16
model = Modified3DUNet(in_channels, n_classes, base_n_filter).to(device)
model_1c_2_3c = evaluate_tools.load_checkpoint_1c_2_3c(model, modelpath_init, device)
criterion = SoftDiceLoss_MultiClasses(0.000001)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay = 0.001)

# training
running_loss = 0
val_loss_best = 65535
num_points = 3
sim_epochs = 1

print('simulate clicks ...')
clicks_info_train = sim_clicks_batch_tumor.sim_clicks_dataset_batch(dataloader=train_loader,clicks_info=clicks_info_init,\
                                                                    model=model_1c_2_3c,num_points=num_points,sigma=sigma,device=device)
# print(clicks_info_train)
# clicks_info_val_epoch = sim_clicks_batch_tumor.sim_clicks_dataset_batch(dataloader=val_loader,clicks_info=clicks_info_val_init,\
#                                                                     model=model_1c_2_3c,num_points=num_points,sigma=sigma,device=device)
# fine-tune the model
print('train model...')
for i in range(100000):
    print('**** epoch ',i,' ****')
    running_loss = 0
    for train_batch in tqdm(train_loader): 
        train_data_1_c = train_batch['data'].to(device)
        train_data_1_c = torch.unsqueeze(train_data_1_c,1)
        data_keys = train_batch['kidney_id'] 
        t_arr_shape = train_batch['mask'].shape[1:4]
        train_maps = torch.tensor(sim_clicks_batch_tumor.get_prob_maps(data_keys,clicks_info_train,t_arr_shape,sigma)).to(device)
        train_data = torch.cat((train_data_1_c, train_maps),1)
        train_label = train_batch['mask'].to(device)
        optimizer.zero_grad()
        output = model_1c_2_3c(train_data)
        output = output[1]
        loss = criterion(output, train_label)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
    print('training loss : %.3f'%(running_loss/len(train_loader)))
    evaluate_tools.write_row_csv(train_csv_writer,running_loss/len(train_loader))

    print('validate model ...')
    if i % 1 == 0:
        model_1c_2_3c.eval() # model in evaluation mode
        run_val_loss = 0
        with torch.no_grad():
            clicks_info_val_epoch = sim_clicks_batch_tumor.sim_clicks_dataset_batch(dataloader=val_loader,clicks_info=clicks_info_val_init,\
                                                                    model=model_1c_2_3c,num_points=num_points,sigma=sigma,device=device)
            for val_batch in tqdm(val_loader):
                val_data_1_c = val_batch['data'].to(device)
                val_data_1_c = torch.unsqueeze(val_data_1_c,1)
                data_keys_val = val_batch['kidney_id']
                val_maps = torch.tensor(sim_clicks_batch_tumor.get_prob_maps(data_keys_val, clicks_info_val_epoch,\
                                        val_batch['mask'].shape[1:4],sigma)).to(device)
                val_data = torch.cat((val_data_1_c, val_maps),1)
                val_label = val_batch['mask'].to(device)
                output_val = model_1c_2_3c(val_data)
                logits_val = output_val[1]
                val_loss = criterion(logits_val, val_label)
                run_val_loss+=val_loss.item()
            val_loss_final = run_val_loss/len(val_loader)
            print('val loss : %.3f'%(val_loss_final))
            evaluate_tools.write_row_csv(val_csv_writer,val_loss_final)

            # save model
            if val_loss_final < val_loss_best:
                val_loss_best = val_loss_final
                print('**** best val loss is : %.3f ******'%(val_loss_best))
                evaluate_tools.save_checkpoint_with_date(model_1c_2_3c, val_loss_best, i, curtime, check_points_save)
                print ('********** model saved !************')
        model_1c_2_3c.train()