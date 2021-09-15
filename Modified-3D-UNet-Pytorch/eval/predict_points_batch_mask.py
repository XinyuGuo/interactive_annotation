# system
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import paths
import pdb
import numpy as np
import pandas as pd
# self-defined 
from kidney_dataloader_1_c import KidneyDataset_1_c
from kidney_loss_factory import SoftDiceLoss
import evaluate_tools
import sim_clicks_batch_mask
from model import Modified3DUNet

# get device 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# one channel dataloader for simulate clicks
# csv_val_path = '../val_points_small_1.csv'
# csv_val_path = '../train.csv'
csv_val_path = '../val_0122_20.csv'
# csv_val_path = '../train_0122_20_small.csv'
val_data =  KidneyDataset_1_c(csv_val_path)
val_loader_init = DataLoader(val_data, batch_size = 1, shuffle=False, num_workers=1)

# one channel pretrained model
in_channels_init = 1
n_classes_init = 1
base_n_filter_init = 16
modelpath_init = '/data/ccusr/xinyug/annotation/Modified-3D-UNet-Pytorch/checkpoints/softdice_1c/checkpoint_71_20191225_1636.pth.tar'
model_1_c = Modified3DUNet(in_channels_init, n_classes_init, base_n_filter_init).to(device)
model_init = evaluate_tools.load_checkpoint_with_date(model_1_c, modelpath_init)

# simulate initial clicks based on a trained 3-channel input mdoel
print('simulate initial clicks ...')
sigma = 10
clicks_info_val_init = sim_clicks_batch_mask.sim_initial_click_dataset_batch(val_loader_init, model_init, sigma, device)
# pdb.set_trace()

# load 4-channel trained model
in_channels = 4
n_classes = 1
base_n_filter = 16
model = Modified3DUNet(in_channels, n_classes, base_n_filter).to(device)
# modelpath = '/data/ccusr/xinyug/annotation/Modified-3D-UNet-Pytorch/checkpoints/softdice_3_sim/checkpoint.pth.tar' 
# checkpoint_5_20200109_1743.pth.tar
# modelpath = '/data/ccusr/xinyug/annotation/Modified-3D-UNet-Pytorch/checkpoints/softdice_3_sim/checkpoint.pth.tar'
modelpath = '/data/ccusr/xinyug/annotation/Modified-3D-UNet-Pytorch/checkpoints/softdice_sim_points_one_time_batch_mask/checkpoint_195_20200129_1649.pth.tar'
model = evaluate_tools.load_checkpoint_with_date(model, modelpath)

# simulate clicks
print('simulate clicks ...')
num_points = 3
clicks_info_val = sim_clicks_batch_mask.sim_clicks_dataset_batch(val_loader_init,clicks_info_val_init,model,num_points, sigma,device)
# print(clicks_info_val)
# pdb.set_trace()

# predict
pred_ious = []
case_ids = []
case_clicks = []
sum_iou = torch.tensor(0.0).to(device)
for val_case in tqdm(val_loader_init): 
    val_data_1_c = val_case['data'].to(device)
    val_data_1_c = torch.unsqueeze(val_data_1_c,1)
    data_key = val_case['kidney_id'] 
    val_map = torch.tensor(sim_clicks_batch_mask.get_prob_maps(data_key,clicks_info_val,val_case['mask'].shape[1:4],sigma)).to(device)
    val_data_1_c_mask = val_case['pred_mask'].to(device)
    val_data_1_c_mask = torch.unsqueeze(val_data_1_c_mask,1)
    val_data = torch.cat((val_data_1_c, val_map, val_data_1_c_mask),1)
    # val_data = torch.cat((val_data_1_c, val_map),1)
    out = model(val_data)
    mask_tensor = torch.sigmoid(out[0])
    mask_tensor[mask_tensor>0.5] = 1
    mask_tensor[mask_tensor<=0.5] = 0
    iou = evaluate_tools.getIOU_two_classes(mask_tensor, val_case['mask'].view(1,-1).squeeze(0).to(device))
    pred_ious.append(iou.cpu().numpy())
    case_ids.append(data_key[0])
    clicks = clicks_info_val[data_key[0]]
    all_clicks = tuple(list(map(tuple,np.transpose(click))) for click in clicks)
    # all_clicks = tuple(list(map(tuple,np.transpose(clicks))))
    case_clicks.append(all_clicks)
    sum_iou+=iou

print(sum_iou/len(val_loader_init))
df_res = pd.DataFrame({'case_id':case_ids, 'case_iou': pred_ious, 'case_clicks': case_clicks})
df_res.to_csv('results_without_largest_cc_3_channel_mask.csv',index=False)


    # val_data_1_c = val_batch['data'].to(device)
    # val_data_1_c = torch.unsqueeze(val_data_1_c,1)
    # data_keys_val = val_batch['kidney_id']
    # val_maps = torch.tensor(sim_clicks.get_prob_maps(data_keys_val, clicks_info_val,\
    #                         val_batch['mask'].shape[1:4],sigma)).to(device)
    # val_data = torch.cat((val_data_1_c, val_maps),1)
