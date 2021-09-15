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
import sim_clicks_for_check_batch
from model import Modified3DUNet

# get device 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# csv_val_path = '../check_all_cases.csv'
csv_val_path = '../check_case.csv'
# create check cases save dir
case_check_save = 'case_batch_scipy_k_means_small'
if not os.path.exists(case_check_save):
    os.mkdir(case_check_save)

val_data =  KidneyDataset_1_c(csv_val_path)
val_loader_init = DataLoader(val_data, batch_size = 4, shuffle=False, num_workers=1)

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
clicks_info_val_init, all_ious = sim_clicks_for_check_batch.sim_initial_click_dataset_batch(val_loader_init, model_init, sigma, case_check_save, device)
# pdb.set_trace()
# clicks_info_val_init, iou_info = sim_clicks_for_check_batch.sim_initial_click_dataset(val_loader_init, model_init, sigma, case_check_save, device)

# load 4-channel trained model
in_channels = 3 
n_classes = 1
base_n_filter = 16
model = Modified3DUNet(in_channels, n_classes, base_n_filter).to(device)
# modelpath = '/data/ccusr/xinyug/annotation/Modified-3D-UNet-Pytorch/checkpoints/softdice_3_sim/checkpoint.pth.tar' 
# checkpoint_5_20200109_1743.pth.tar
modelpath = '/data/ccusr/xinyug/annotation/Modified-3D-UNet-Pytorch/checkpoints/softdice_sim_3_channel_v1/checkpoint_25_20200117_1404.pth.tar'
model = evaluate_tools.load_checkpoint_with_date(model, modelpath)

# simulate clicks
print('simulate clicks ...')
num_points = 3
clicks_info_val, all_ious = sim_clicks_for_check_batch.sim_clicks_dataset_batch(val_loader_init,clicks_info_val_init,all_ious,model,num_points, sigma, case_check_save,device)

# print('save clicks prob maps ...')
# sim_clicks_for_check_batch.save_click_prob_maps(val_loader_init,clicks_info_val,sigma,case_check_save)
sim_clicks_for_check_batch.save_multiple_clicks_prob_maps(val_loader_init,clicks_info_val,sigma,case_check_save)
ious_seq = []
all_clicks = []
case_ids = []
for key in clicks_info_val.keys():
    ious = np.stack(all_ious[key])
    ious = np.round(ious,decimals=4).tolist()
    ious_seq.append(ious)
    clicks = clicks_info_val[key]
    clicks_list = tuple(list(map(tuple,np.transpose(click))) for click in clicks) 
    all_clicks.append(clicks_list)
    case_ids.append(key)

df_res = pd.DataFrame({'case_id':case_ids, 'case_iou': ious_seq, 'case_clicks': all_clicks})
df_res.to_csv('results_csv/case_batch_scipy_k_means_small.csv',index=False)

# # predict
# pred_ious = []
# case_ids = []
# case_clicks = []
# sum_iou = torch.tensor(0.0).to(device)
# for val_case in tqdm(val_loader_init): 
#     train_data_1_c = val_case['data'].to(device)
#     data_key = val_case['kidney_id'] 
#     val_map = torch.tensor(sim_clicks_for_check.get_prob_maps(data_key,clicks_info_val,val_case['mask'].shape[1:4],sigma)).to(device)
#     val_data = torch.stack((train_data_1_c, val_map),1)
#     out = model(val_data)
#     mask_tensor = torch.sigmoid(out[0])
#     mask_tensor[mask_tensor>0.5] = 1
#     mask_tensor[mask_tensor<=0.5] = 0
#     iou = evaluate_tools.getIOU_two_classes(mask_tensor, val_case['mask'].view(1,-1).squeeze(0).to(device))
#     pred_ious.append(iou.cpu().numpy())
#     case_ids.append(data_key[0])
#     clicks = clicks_info_val[data_key[0]]
#     # print(clicks)
#     all_clicks = tuple(list(map(tuple,np.transpose(click))) for click in clicks)
#     case_clicks.append(all_clicks)
#     sum_iou+=iou

# print(sum_iou/len(val_loader_init))
# df_res = pd.DataFrame({'case_id':case_ids, 'case_iou': pred_ious, 'case_clicks': case_clicks})
# df_res.to_csv('checkpoint_val.csv',index=False)