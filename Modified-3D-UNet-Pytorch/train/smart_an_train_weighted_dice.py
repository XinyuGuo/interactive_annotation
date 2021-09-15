from kidney_dataloader_weighted_dice import KidneyDataset_Weighted_Dice_1, KidneyDataset_Weighted_Dice_2,KidneyDataset_Weighted_Dice_3
from kidney_loss_factory import WeightedDiceLoss
from torch.autograd import Variable
import evaluate_tools
import sim_clicks_weighted_dice
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
import pickle
import pdb
import numpy as np
import pandas as pd
import sys
# ********************
# train & val data
csv_train_path = '../train_aug.csv'
csv_val_path = '../val_new.csv'
# csv_train_path = '../train_aug_test.csv'
# csv_val_path = '../val_aug_test.csv'

# GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

###############################
####simulate initial clicks####
###############################
def save_init_clicks_info(clicks_info,path):
    '''
    save the clicks dictionary to pickle
    '''
    with open(path, 'wb') as handle:
        pickle.dump(clicks_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_init_clicks_info(path):
    '''
    load the pickle file
    '''
    with open(path, 'rb') as handle:
        info = pickle.load(handle)
    return info

def add_column(src_csv,save_root,npy_name,col_name,csv_save_path):
    '''
    add one column to the dataframe
    '''
    src_df = pd.read_csv(src_csv)
    save_names = [case_id + npy_name + '.npy' for case_id  in src_df['case_img_id'].tolist()]
    save_paths = [os.path.join(save_root,save_name) for save_name in save_names]
    src_df[col_name] = save_paths
    src_df.to_csv(csv_save_path,index=False)

simulate = True
sigma = 10
init_save_root = '../kidney/init_res_simulations_aug/'
if not os.path.exists(init_save_root):
        os.mkdir(init_save_root)

if simulate :
    # one channel dataloader for simulate clicks
    kidney_data =  KidneyDataset_Weighted_Dice_1(csv_train_path)
    val_data =  KidneyDataset_Weighted_Dice_1(csv_val_path)
    train_loader_init = DataLoader(kidney_data, batch_size = 8, shuffle=False, num_workers=4)
    val_loader_init = DataLoader(val_data, batch_size = 8, shuffle=False, num_workers=4)
    # load 1c model
    in_channels_init = 1
    n_classes_init = 1
    base_n_filter_init = 16
    print('load the 1-chanel model...')
    # modelpath_init = './checkpoints/softdice_1c/checkpoint_71_20191225_1636.pth.tar'
    modelpath_init = './checkpoints/softdice_1c_aug/checkpoint_89_20200302_2005.pth.tar'
    model_1_c = Modified3DUNet(in_channels_init, n_classes_init, base_n_filter_init).to(device)
    model_init = evaluate_tools.load_checkpoint_with_date(model_1_c, modelpath_init)
    # simulate clicks and get the initial weighted maps
    print('simulate initial clicks ...')
    clicks_info_init =sim_clicks_weighted_dice.sim_initial_click_dataset_batch(train_loader_init, model_init, sigma, init_save_root, device)
    train_save_path = os.path.join(init_save_root,'train_clicks_info_init.pickle')
    save_init_clicks_info(clicks_info_init,train_save_path)
    clicks_info_val_init = sim_clicks_weighted_dice.sim_initial_click_dataset_batch(val_loader_init, model_init, sigma,init_save_root,device)
    val_save_path = os.path.join(init_save_root,'val_clicks_info_init.pickle')
    save_init_clicks_info(clicks_info_val_init,val_save_path)
    add_column(csv_train_path,init_save_root,'_init_map','init_weighted_path','../train_aug_2.csv')
    add_column(csv_val_path,init_save_root,'_init_map','init_weighted_path','../val_aug_2.csv')
else:
    print('load initial clicks ...')
    train_save_path = os.path.join(init_save_root,'train_clicks_info_init.pickle')
    val_save_path = os.path.join(init_save_root,'val_clicks_info_init.pickle')
    clicks_info_init = load_init_clicks_info(train_save_path)
    clicks_info_val_init = load_init_clicks_info(val_save_path)

# pdb.set_trace()

###############################
#### simulate train clicks ####
###############################
try:
    csv_train_path2 = '../train_aug_2.csv'
    print('load file ' + csv_train_path2)
    pd.read_csv(csv_train_path2)
except:
    print('no such file!')
    sys.exit(0)

try:
    csv_val_path2 = '../val_aug_2.csv'
    print('load file ' + csv_val_path2)
    pd.read_csv(csv_val_path2)
except:
    print('no such file!')
    sys.exit(0)

kidney_data_2 =  KidneyDataset_Weighted_Dice_2(csv_train_path2)
val_data_2 =  KidneyDataset_Weighted_Dice_2(csv_val_path2)
train_loader_2 = DataLoader(kidney_data_2, batch_size = 4, shuffle=True, num_workers=4)
val_loader_2 = DataLoader(val_data_2, batch_size = 2, shuffle = True, num_workers = 4)

def datestr():
	now = time.localtime()
	return '{:04}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)

curtime = datestr()

# logging train and val loss
train_loss_l = []
val_loss_l = []
log_save_dir = './results/weighted_softdice'
if not os.path.exists(log_save_dir):
    os.mkdir(log_save_dir)

# model save path
check_points_save = './checkpoints/weighted_soft'
if not os.path.exists(check_points_save):
    os.mkdir(check_points_save)

# model  image + foreground + background
print('construct the 3-chanel fine-tune model')
in_channels = 3
n_classes = 1
base_n_filter = 16
num_points = 3
model = Modified3DUNet(in_channels, n_classes, base_n_filter).to(device)
# modelpath_1_c = './checkpoints/softdice_1c/checkpoint_71_20191225_1636.pth.tar'
modelpath_1_c = './checkpoints/softdice_1c_aug/checkpoint_89_20200302_2005.pth.tar'
model_1c_2_3c = evaluate_tools.load_checkpoint_1c_2_3c(model, modelpath_1_c, device)
# criterion = SoftDiceLoss(0.000001)
criterion = WeightedDiceLoss(0.000001,2,device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

print('simulate clicks ...')
res_map_root = init_save_root 
# res_map_root = '../kidney/initial_simulations'    
clicks_info_train = sim_clicks_weighted_dice.sim_clicks_dataset_batch(train_loader_2,clicks_info_init,model_1c_2_3c,num_points,sigma,res_map_root,device)
clicks_info_val_epoch = sim_clicks_weighted_dice.sim_clicks_dataset_batch(val_loader_2,clicks_info_val_init,model_1c_2_3c,num_points,sigma,res_map_root,device)

add_column(csv_train_path2,init_save_root,'_res_map','res_weighted_path','../train_aug_3.csv')
add_column(csv_val_path2,init_save_root,'_res_map','res_weighted_path','../val_aug_3.csv')
# pdb.set_trace()

###############################
####      train model      ####
###############################
try:
    csv_train_path3 = '../train_aug_3.csv'
    print('load file ' + csv_train_path3)
    pd.read_csv(csv_train_path3)
except:
    print('no such file!')
    sys.exit(0)

try:
    csv_val_path3 = '../val_aug_3.csv'
    print('load file ' + csv_val_path3)
    pd.read_csv(csv_val_path3)
except:
    print('no such file!')
    sys.exit(0)

kidney_data_3 =  KidneyDataset_Weighted_Dice_3(csv_train_path3)
val_data_3 =  KidneyDataset_Weighted_Dice_3(csv_val_path3)
train_loader_3 = DataLoader(kidney_data_3, batch_size = 4, shuffle=True, num_workers=4)
val_loader_3 = DataLoader(val_data_3, batch_size = 2, shuffle = True, num_workers = 4)

running_loss = 0
val_loss_best = 65535
sim_epochs = 1
print('train model...')
for i in range(100000):
    print('**** epoch ',i,' ****')
    running_loss = 0
    for train_batch in tqdm(train_loader_3): 
        train_data_1_c = train_batch['data'].to(device)
        train_data_1_c = torch.unsqueeze(train_data_1_c,1)
        data_keys = train_batch['kidney_id'] 
        t_arr_shape = train_batch['mask'].shape[1:4]
        train_maps = torch.tensor(sim_clicks_weighted_dice.get_prob_maps(data_keys,clicks_info_train,t_arr_shape,sigma)).to(device)
        train_data = torch.cat((train_data_1_c, train_maps),1)
        train_label = train_batch['mask'].to(device)
        optimizer.zero_grad()
        output = model_1c_2_3c(train_data)
        output = output[1]
        output = torch.squeeze(output,1)
        pred = torch.sigmoid(output)
        weighted_maps = train_batch['res_weighted_map'].to(device)
        loss = criterion(pred, train_label, weighted_maps)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
    print('training loss : %.3f'%(running_loss/len(train_loader_3)))
    train_loss_l.append(running_loss/len(train_loader_3))

    print('validate model ...')
    if i % 1 == 0:
        model_1c_2_3c.eval() # model in evaluation mode
        run_val_loss = 0
        for val_batch in tqdm(val_loader_3):
            val_data_1_c = val_batch['data'].to(device)
            val_data_1_c = torch.unsqueeze(val_data_1_c,1)
            data_keys_val = val_batch['kidney_id']
            val_maps = torch.tensor(sim_clicks_weighted_dice.get_prob_maps(data_keys_val, clicks_info_val_epoch,\
                                    val_batch['mask'].shape[1:4],sigma)).to(device)
            val_data = torch.cat((val_data_1_c, val_maps),1)
            val_label = val_batch['mask'].view(1,-1).squeeze(0).to(device)
            output_val = model_1c_2_3c(val_data)
            logits_val = output_val[1]
            output = torch.squeeze(logits_val,1)
            pred = torch.sigmoid(output)
            weighted_maps = val_batch['res_weighted_map'].to(device)
            val_loss = criterion(pred, val_label, weighted_maps)
            run_val_loss+=val_loss.item()
        val_loss_final = run_val_loss/len(val_loader_3) 
        val_loss_l.append(val_loss_final)
        print('**** val loss is : %.3f ******'%(val_loss_final))
        
        # save model
        if val_loss_final < val_loss_best:
            val_loss_best = val_loss_final
            print('**** best val loss is : %.3f ******'%(val_loss_best))
            evaluate_tools.save_checkpoint_with_date(model_1c_2_3c, val_loss_best, i, curtime, check_points_save)
            print ('********** model saved !************')
        evaluate_tools.save_loss_csv(train_loss_l, val_loss_l, log_save_dir)
        model_1c_2_3c.train()