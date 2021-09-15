import evaluate_tools
from model import Modified3DUNet
from model_attn_2 import Modified3DUNet_ATTN
from torch.utils.data import DataLoader
# from kidney_dataloader_pred import KidneyDatasetPred
from kidney_dataloader_1_c_aug_2_classes import KidneyDataset, KidneyDataset_standardization,KidneyDataset_standardization_2
import torch.nn.functional as F
import torch.nn
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
import pdb

# dataset
csv_pred_path = '../val_weighted_dice_2.csv'
kidney_dataset_pred = KidneyDataset_standardization_2(csv_pred_path)
pred_loader  = DataLoader(kidney_dataset_pred, batch_size = 1, shuffle=False, num_workers=1)

# gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model
in_channels = 1
n_classes = 3
base_n_filter = 16

modelpath = './checkpoints/models/unet_annt_2_dice_balance_8719.pth.tar' # best balance
model = Modified3DUNet_ATTN(in_channels, n_classes, base_n_filter).to(device)
model_pred = evaluate_tools.load_checkpoint_with_date(model, modelpath) 

sum_iou = torch.tensor(0.0).to(device)
case_ious = []
case_ids = []

# pred kidned
for pred_batch in tqdm(pred_loader):
    _, out = model_pred(torch.unsqueeze(pred_batch['data'],1).to(device))
    out = F.softmax(out,dim=1)
    # construct pred onehot 
    _, max_index = torch.max(out,1)
    pred_labels = out.data.clone().zero_()
    pred_labels.scatter_(1, max_index.unsqueeze(1),1)
    kidney_labels = pred_labels[:,1,:,:,:]
    k_labels = kidney_labels.view(1,-1).squeeze(0).to(device)
    data_key = pred_batch['kidney_id']
    mask_meta = pred_batch['mask_meta']
    mask_gt_shape = torch.squeeze(pred_batch['mask'],0).shape
    pred_batch_M = pred_batch['mask']
    pred_batch_M[pred_batch_M==2]=0
    iou = evaluate_tools.getIOU_two_classes(k_labels, pred_batch_M.view(1,-1).squeeze(0).to(device))
    case_ious.append(iou.cpu().detach().numpy())
    case_ids.append(data_key[0])
    sum_iou+=iou

print(sum_iou/len(pred_loader))
df_res = pd.DataFrame({'case_id':case_ids, 'case_iou': case_ious})