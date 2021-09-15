import evaluate_tools
from model_attn_2 import Modified3DUNet_ATTN
from torch.utils.data import DataLoader
from kidney_dataloader_1_c_aug_2_classes import KidneyDataset, KidneyDataset_standardization,KidneyDataset_standardization_2
import torch.nn.functional as F
import torch.nn
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
import pdb

def save_pred_labels(mask, mask_meta, mask_path):
    '''
    save fp_map and fn_map
    '''
    # mask size
    # mask = mask.view(mask_gt_shape)
    # numpy array to medical image
    mask = torch.squeeze(mask,0)
    # print(mask.shape)
    # pdb.set_trace()
    mask_arr = mask.cpu().detach().numpy()
    mask_arr = mask_arr.astype(np.uint8) 
    # print(mask_arr.dtype)
    # print(np.unique(mask_arr))
    
    mask_img = sitk.GetImageFromArray(mask_arr)
    # set origin, direction, spacing
    # print([x.numpy()[0] for x in mask_meta['spacing']])
    mask_img.SetSpacing([x.numpy()[0] for x in mask_meta['spacing']])
    mask_img.SetOrigin([x.numpy()[0] for x in mask_meta['origin']])
    mask_img.SetDirection([x.numpy()[0] for x in mask_meta['direction']])
    # pdb.set_trace()
    # save medical image
    case_name = os.path.split(mask_path)[-1]
    case_new_name = case_name.replace('mask','pred_tumor')
    case_path = os.path.join(os.path.split(mask_path)[0],case_new_name)
    # print(case_path)
    # pdb.set_trace()
    sitk.WriteImage(mask_img, case_path)

    
# dataset
# csv_pred_path = '../val_weighted_dice_2.csv'
csv_pred_path = '../val_tumor.csv'
# kidney_dataset_pred = KidneyDataset(csv_pred_path)
kidney_dataset_pred = KidneyDataset_standardization_2(csv_pred_path)
pred_loader  = DataLoader(kidney_dataset_pred, batch_size = 1, shuffle=False, num_workers=1)

# gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model
in_channels = 1
n_classes = 3
base_n_filter = 16

# modelpath = './checkpoints/models/unet_annt_2_dice_tbest_4926.pth.tar' # best so far
model = Modified3DUNet_ATTN(in_channels, n_classes, base_n_filter).to(device)
model_pred = evaluate_tools.load_checkpoint_with_date(model, modelpath)

# save_root = '/data/ccusr/xinyug/annotation/kidney/ori/'
sum_iou = torch.tensor(0.0).to(device)
case_ious = []
case_ids = []

# pred tumor
save_pred = False
for pred_batch in tqdm(pred_loader):
    _, out = model_pred(torch.unsqueeze(pred_batch['data'],1).to(device))
    out = F.softmax(out,dim=1)
    # construct pred onehot 
    _, pred_labels = torch.max(out,1)
    data_key = pred_batch['kidney_id']
    if save_pred:
        m_path = pred_batch['mask_path'][0]
        m_meta = pred_batch['mask_meta']
        save_pred_labels(pred_labels,m_meta,m_path)

    pred_labels[pred_labels==1] = 0
    pred_labels[pred_labels==2] = 1
    tumor_labels = pred_labels
    t_labels = tumor_labels.view(1,-1).squeeze(0).to(device)
    mask_gt_shape = torch.squeeze(pred_batch['mask'],0).shape
    pred_batch_M = pred_batch['mask']
    pred_batch_M[pred_batch_M==1]=0
    pred_batch_M[pred_batch_M==2]=1
    iou = evaluate_tools.getIOU_two_classes(t_labels, pred_batch_M.view(1,-1).squeeze(0).to(device))
    case_ious.append(iou.cpu().detach().numpy())
    case_ids.append(data_key[0])
    sum_iou+=iou

print(sum_iou/len(pred_loader))
# print(sum_iou/21)
df_res = pd.DataFrame({'case_id':case_ids, 'case_iou': case_ious})
df_res.to_csv('one_chanel_aug_tumor_dice_8505.csv',index=False)