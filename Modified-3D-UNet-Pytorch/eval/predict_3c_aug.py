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
from kidney_dataloader_weighted_dice import KidneyDataset_Weighted_Dice_1
# import sim_clicks_weighted_dice
from kidney_loss_factory import SoftDiceLoss
import evaluate_tools
import sim_clicks_batch_v2
from model import Modified3DUNet

# get device 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# one channel dataloader for simulate clicks
csv_val_path = '../val_new.csv'
val_data =  KidneyDataset_Weighted_Dice_1(csv_val_path)
val_loader_init = DataLoader(val_data, batch_size = 1, shuffle=False, num_workers=1)

# one channel pretrained model
in_channels_init = 1
n_classes_init = 1
base_n_filter_init = 16
modelpath_init = './checkpoints/softdice_1c_aug/checkpoint_89_20200302_2005.pth.tar'
model_1_c = Modified3DUNet(in_channels_init, n_classes_init, base_n_filter_init).to(device)
model_init = evaluate_tools.load_checkpoint_with_date(model_1_c, modelpath_init)

# simulate initial clicks based on a trained 3-channel input mdoel
print('simulate initial clicks ...')
sigma = 10
init_save_root = '../kidney/pred_val/'
if not os.path.exists(init_save_root):
    os.mkdir(init_save_root)
clicks_info_val_init = sim_clicks_batch_v2.sim_initial_click_dataset_batch(val_loader_init, model_init, sigma, device)

# clicks_info_val_init = sim_clicks_batch_v2.sim_initial_click_dataset_batch(val_loader_init, model_init, sigma, device)
# (val_loader_init,clicks_info_val_init,model,num_points, sigma,device)

# load 3-channel trained model
in_channels = 3
n_classes = 1
base_n_filter = 16
model_3c = Modified3DUNet(in_channels, n_classes, base_n_filter).to(device)
# modelpath = './checkpoints/weighted_soft/checkpoint_20_20200319_1957.pth.tar'
modelpath = './checkpoints/softdice_clicks_finetune/checkpoint_64_20200319_2308.pth.tar'
model_3c = evaluate_tools.load_checkpoint_with_date(model_3c, modelpath)

# simulate clicks
print('simulate clicks ...')
num_points = 15
# clicks_info_val = sim_clicks_weighted_dice.sim_clicks_dataset_batch_pred(val_loader_init,clicks_info_val_init,model_3c,num_points,sigma,device)
clicks_info_val = sim_clicks_batch_v2.sim_clicks_dataset_batch(val_loader_init,clicks_info_val_init,model_3c,num_points, sigma,device)
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
    val_map = torch.tensor(sim_clicks_batch_v2.get_prob_maps(data_key,clicks_info_val,val_case['mask'].shape[1:4],sigma)).to(device)
    val_data = torch.cat((val_data_1_c, val_map),1)
    out = model_3c(val_data)
    mask_tensor = torch.sigmoid(out[0])
    mask_tensor[mask_tensor>0.5] = 1
    mask_tensor[mask_tensor<=0.5] = 0
    iou = evaluate_tools.getIOU_two_classes(mask_tensor, val_case['mask'].view(1,-1).squeeze(0).to(device))
    pred_ious.append(iou.cpu().numpy())
    case_ids.append(data_key[0])
    clicks = clicks_info_val[data_key[0]]
    all_clicks = tuple(list(map(tuple,np.transpose(click))) for click in clicks)
    case_clicks.append(all_clicks)
    sum_iou+=iou

print(sum_iou/len(val_loader_init))
df_res = pd.DataFrame({'case_id':case_ids, 'case_iou': pred_ious, 'case_clicks': case_clicks})
df_res.to_csv('weighted_dice_res.csv',index=False)
