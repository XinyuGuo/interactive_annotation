import evaluate_tools
from model import Modified3DUNet
from torch.utils.data import DataLoader
# from kidney_dataloader_pred import KidneyDatasetPred
from kidney_dataloader_1_c_aug import KidneyDataset
import torch.nn
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
import pdb

def save_mask_to_nii(case_id, mask, mask_meta, mask_gt_shape,save_root,is_mask):
    '''
    save fp_map and fn_map
    '''
    # mask size
    mask = mask.view(mask_gt_shape)
    # numpy array to medical image
    mask_arr = mask.cpu().detach().numpy() 
    mask_img = sitk.GetImageFromArray(mask_arr)
    # set origin, direction, spacing
    mask_img.SetSpacing([x.numpy()[0] for x in mask_meta['spacing']])
    mask_img.SetOrigin([x.numpy()[0] for x in mask_meta['origin']])
    mask_img.SetDirection([x.numpy()[0] for x in mask_meta['direction']])
    # save mdical image
    dir_ele = case_id.split('_')
    dir_name = dir_ele[0] + '_' + dir_ele[1]
    kidney_id = dir_ele[2]
    
    if is_mask:
        file_name = 'pred_mask_' + kidney_id + '.nii.gz'
        case_path = os.path.join(os.path.join(save_root, dir_name), file_name)
    else:
        file_name = 'pred_prob_' + kidney_id + '.nii.gz'
        case_path = os.path.join(os.path.join(save_root, dir_name), file_name)
    # print(case_path)
    sitk.WriteImage(mask_img, case_path)
    
# dataset
csv_pred_path = '../val_weighted_dice_2.csv'
kidney_dataset_pred = KidneyDataset(csv_pred_path)
pred_loader  = DataLoader(kidney_dataset_pred, batch_size = 1, shuffle=False, num_workers=1)

# gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model
in_channels = 1
n_classes = 1
base_n_filter = 16

# modelpath = '/data/ccusr/xinyug/annotation/Modified-3D-UNet-Pytorch/checkpoints/softdice/checkpoint_84_20191217_1208.pth.tar'
# modelpath = '/data/ccusr/xinyug/annotation/Modified-3D-UNet-Pytorch/checkpoints/softdice_1c/checkpoint_71_20191225_1636.pth.tar'
modelpath = '/data/ccusr/xinyug/annotation/Modified-3D-UNet-Pytorch/checkpoints/softdice_1c_aug/checkpoint_89_20200302_2005.pth.tar'
# modelpath = '/data/ccusr/xinyug/annotation/Modified-3D-UNet-Pytorch/checkpoints/weighted_soft/checkpoint_1_20200319_1957.pth.tar'
model = Modified3DUNet(in_channels, n_classes, base_n_filter).to(device)
model_pred = evaluate_tools.load_checkpoint_with_date(model, modelpath)

# save_root = '/data/ccusr/xinyug/annotation/kidney/ori/'
sum_iou = torch.tensor(0.0).to(device)
case_ious = []
case_ids = []

for pred_batch in tqdm(pred_loader):
    out = model_pred(torch.unsqueeze(pred_batch['data'],1).to(device))
    data_key = pred_batch['kidney_id']
    # print(data_key[0]) 
    mask_meta = pred_batch['mask_meta']
    mask_gt_shape = torch.squeeze(pred_batch['mask'],0).shape
    mask_tensor = torch.sigmoid(out[0])

    # save mask prob
    # save_mask_to_nii(data_key[0], mask_tensor, mask_meta, mask_gt_shape,save_root,False)

    mask_tensor[mask_tensor>0.5] = 1
    mask_tensor[mask_tensor<=0.5] = 0
    
    # save predicted mask
    # save_mask_to_nii(data_key[0], mask_tensor, mask_meta, mask_gt_shape,save_root,True)

    iou = evaluate_tools.getIOU_two_classes(mask_tensor, pred_batch['mask'].view(1,-1).squeeze(0).to(device))
    case_ious.append(iou.cpu().detach().numpy())
    case_ids.append(data_key[0])
    sum_iou+=iou

print(sum_iou/len(pred_loader))
df_res = pd.DataFrame({'case_id':case_ids, 'case_iou': case_ious})
df_res.to_csv('one_chanel_model_with_aug.csv',index=False)