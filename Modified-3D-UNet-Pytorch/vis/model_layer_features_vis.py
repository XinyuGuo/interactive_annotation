import evaluate_tools
from model_attn import Modified3DUNet_ATTN
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

# csv_train_path = '../train_multiC_aug.csv'
# csv_val_path ='../val_weighted_dice_2.csv'
# csv_train_path = '../train_multiC_aug_test.csv'
# csv_val_path = '../val_multiC_aug_test.csv'
data_path = '../val_weighted_dice_2.csv'

kidney_dataset = KidneyDataset_standardization_2(data_path)
# kidney_dataset_val = KidneyDataset_standardization_2(csv_val_path)

data_loader  = DataLoader(kidney_dataset, batch_size = 1, shuffle=True, num_workers=4)

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output[1].detach()
    return hook

# model.fc0.conv2.register_forward_hook(get_activation('fc0.conv2'))
# model.fc1.conv2.register_forward_hook(get_activation('fc1.conv2'))

# output = model(x)
# print(activation['fc0.conv2'])
# print(activation['fc0.conv1'])


# gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model
in_channels = 1
n_classes = 3
base_n_filter = 16
# modelpath = './checkpoints/unet_attn_softdice/checkpoint_64_20200408_1414.pth.tar'
modelpath =  './checkpoints/unet_attn_softdice_2/checkpoint_22_20200408_2044.pth.tar'
model = Modified3DUNet_ATTN(in_channels, n_classes, base_n_filter).to(device)
model_pred = evaluate_tools.load_checkpoint_with_date(model, modelpath) 
model_pred.eval()
# print(model_pred._modules.get('attention_4'))
target_layer = model_pred._modules.get('attention_4')
target_layer.register_forward_hook(get_activation('attention_4'))

attn_maps_path = './attn_maps'
if not os.path.exists(attn_maps_path):
    os.mkdir(attn_maps_path)

for data_batch in tqdm(data_loader):
    img_id = data_batch['kidney_id'][0]
    # print(img_id[0])
    data = data_batch['data'].to(device) 
    data = torch.unsqueeze(data,1)
    output = model(data)
    output = output[1]
    mask_meta = data_batch['mask_meta']
    att_out = activation['attention_4']
    att_arr =  att_out[0,0,:,:,:].cpu().detach().numpy() 
    # print(att_arr.shape)
    # pdb.set_trace()
    att_img = sitk.GetImageFromArray(att_arr[1:])
    # set origin, direction, spacing
    att_img.SetSpacing([x.numpy()[0] for x in mask_meta['spacing']])
    att_img.SetOrigin([x.numpy()[0] for x in mask_meta['origin']])
    att_img.SetDirection([x.numpy()[0] for x in mask_meta['direction']])
    img_name =  img_id+'.nii.gz'
    img_path = os.path.join(attn_maps_path,img_name)
    sitk.WriteImage(att_img,img_path)
    # pdb.set_trace()
    # print(activation['attention_4'].shape)