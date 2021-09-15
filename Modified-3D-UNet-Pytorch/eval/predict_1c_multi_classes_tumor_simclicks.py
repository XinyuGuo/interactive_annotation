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
import torch.nn.functional as F
# self-defined 
from kidney_dataloader_1_c_aug_2_classes import KidneyDataset_standardization,KidneyDataset_standardization_2
from kidney_loss_factory import SoftDiceLoss_MultiClasses
import evaluate_tools
import sim_clicks_batch_tumor
from model import Modified3DUNet

# get device 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
csv_val_path = '../val_tumor.csv'
val_data =  KidneyDataset_standardization(csv_val_path)
val_loader_init = DataLoader(val_data, batch_size = 1, shuffle=False, num_workers=1)

# one channel pretrained model
in_channels_init = 1
n_classes_init = 3
base_n_filter_init = 16
modelpath_init = './checkpoints/models/checkpoint_dice_aug_stand_8505.pth.tar'
model_1_c = Modified3DUNet(in_channels_init, n_classes_init, base_n_filter_init).to(device)
model_init = evaluate_tools.load_checkpoint_with_date(model_1_c, modelpath_init)

# simulate initial clicks based on a trained 3-channel input mdoel
print('simulate initial clicks ...')
sigma = 10

clicks_info_val_init = sim_clicks_batch_tumor.sim_initial_click_dataset_batch(dataloader=val_loader_init, model=model_init,\
                                                                           sigma=sigma, device=device)

# load 4-channel trained model
in_channels = 3 
n_classes = 3
base_n_filter = 16
model = Modified3DUNet(in_channels, n_classes, base_n_filter).to(device)
modelpath = './checkpoints/softdice_clicks_finetune_withtumor/checkpoint_137_20200406_1618.pth.tar'
model = evaluate_tools.load_checkpoint_with_date(model, modelpath)

# simulate clicks
print('simulate clicks ...')
num_points = 3
clicks_info_val = sim_clicks_batch_tumor.sim_clicks_dataset_batch(dataloader=val_loader_init,clicks_info=clicks_info_val_init,\
                                                                  model=model, num_points = num_points, sigma=sigma, device=device)

# predict
case_ious = []
case_ids = []
case_clicks = []
sum_iou = torch.tensor(0.0).to(device)
model.eval()
for val_case in tqdm(val_loader_init): 
    val_data_1_c = val_case['data'].to(device)
    val_data_1_c = torch.unsqueeze(val_data_1_c,1)
    data_key = val_case['kidney_id'] 
    val_map = torch.tensor(sim_clicks_batch_tumor.get_prob_maps(data_key,clicks_info_val,val_case['mask'].shape[1:4],sigma)).to(device)
    val_data = torch.cat((val_data_1_c, val_map),1)
    out = model(val_data)
    out = out[1]
    out = F.softmax(out,dim=1)
    # construct pred onehot 
    _, pred_labels = torch.max(out,1)
    pred_labels[pred_labels==1] = 0
    pred_labels[pred_labels==2] = 1
    tumor_labels = pred_labels
    t_labels = tumor_labels.view(1,-1).squeeze(0).to(device)
    # mask_gt_shape = torch.squeeze(pred_batch['mask'],0).shape
    pred_batch_M = val_case['mask']
    pred_batch_M[pred_batch_M==1]=0
    pred_batch_M[pred_batch_M==2]=1
    iou = evaluate_tools.getIOU_two_classes(t_labels, pred_batch_M.view(1,-1).squeeze(0).to(device))
    case_ious.append(iou.cpu().detach().numpy())
    case_ids.append(data_key[0])
    clicks = clicks_info_val[data_key[0]]
    all_clicks = tuple(list(map(tuple,np.transpose(click))) for click in clicks)
    # all_clicks = tuple(list(map(tuple,np.transpose(clicks))))
    case_clicks.append(all_clicks)
    sum_iou+=iou

print(sum_iou/len(val_loader_init))
df_res = pd.DataFrame({'case_id':case_ids, 'case_iou': case_ious, 'case_clicks': case_clicks})
df_res.to_csv('one_chanel_aug_tumor_simclicks.csv.csv',index=False)


    # val_data_1_c = val_batch['data'].to(device)
    # val_data_1_c = torch.unsqueeze(val_data_1_c,1)
    # data_keys_val = val_batch['kidney_id']
    # val_maps = torch.tensor(sim_clicks.get_prob_maps(data_keys_val, clicks_info_val,\
    #                         val_batch['mask'].shape[1:4],sigma)).to(device)
    # val_data = torch.cat((val_data_1_c, val_maps),1)
