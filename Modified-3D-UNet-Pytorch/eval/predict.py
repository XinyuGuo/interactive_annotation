import evaluate_tools
from model import Modified3DUNet
from torch.utils.data import DataLoader
from kidney_dataloader_pred import KidneyDatasetPred
import torch.nn
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
import pdb

# dataset
# csv_pred_path = '../pred_small.csv'
# csv_pred_path  ='../val_sigma_10.csv'
csv_pred_path = '../train_sigma_10.csv'
kidney_dataset_pred = KidneyDatasetPred(csv_pred_path)
pred_loader  = DataLoader(kidney_dataset_pred, batch_size = 1, shuffle=False, num_workers=1)

# gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model
in_channels = 2
n_classes = 1
base_n_filter = 16
# modelpath = '/data/ccusr/xinyug/annotation/Modified-3D-UNet-Pytorch/checkpoints/softdice/checkpoint_84_20191217_1208.pth.tar'
# modelpath = '/data/ccusr/xinyug/annotation/Modified-3D-UNet-Pytorch/checkpoints/softdice_aug/checkpoint_149_20191219_2240.pth.tar'

modelpath = '/data/ccusr/xinyug/annotation/Modified-3D-UNet-Pytorch/checkpoints/softdice_sigma10/checkpoint_65_20191227_1426.pth.tar'
model = Modified3DUNet(in_channels, n_classes, base_n_filter).to(device)
model_pred = evaluate_tools.load_checkpoint_with_date(model, modelpath)

sum_iou = torch.tensor(0.0).to(device)
for pred_batch in tqdm(pred_loader):
    out = model_pred(pred_batch['data'].to(device))
    mask_meta = pred_batch['mask_meta']
    mask_tensor = torch.sigmoid(out[0])
    mask_tensor[mask_tensor>0.5] = 1
    mask_tensor[mask_tensor<=0.5] = 0
    iou = evaluate_tools.getIOU_two_classes(mask_tensor, pred_batch['mask'].view(1,-1).squeeze(0).to(device))
    # print(iou)
    # pdb.set_trace()
    sum_iou+=iou
print(sum_iou/len(pred_loader))

    # mask_arr = mask_tensor.view(128,128,64).detach().cpu().clone().numpy().astype(np.uint16)
    # pred_mask = sitk.GetImageFromArray(mask_arr)
    # # ori_mask =sitk.GetImageFromArray(pred_batch['mask_img'])   
    # # print(mask_meta['spacing'])
    # # print ([x.numpy()[0] for x in mask_meta['spacing']])
    # # pdb.set_trace() 
    # pred_mask.SetSpacing([x.numpy()[0] for x in mask_meta['spacing']])
    # pred_mask.SetDirection([x.numpy()[0] for x in mask_meta['direction']])
    # pred_mask.SetOrigin([x.numpy()[0] for x in mask_meta['origin']])
    # sitk.WriteImage(pred_mask, 'pred_00206_1.nii.gz')


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# modelpath = '/data/ccusr/xinyug/annotation/Modified-3D-UNet-Pytorch/checkpoints/bce/checkpoint_8_20191216_1551.pth.tar'

# model = Modified3DUNet(1, 2, 16).to(device)
# model_eval = evaluate_tools.load_checkpoint_with_date(model, modelpath)

# data = '/data/ccusr/xinyug/annotation/kidney/ori/case_00071/kidney_1_resampled.nii.gz'
# mask_data = '/data/ccusr/xinyug/annotation/kidney/ori/case_00071/mask_1_resampled.nii.gz'
# pro_data = '/data/ccusr/xinyug/annotation/kidney/ori/case_00071/prob_map_1_resmapled.nii.gz'
# data_arr = sitk.GetArrayFromImage(sitk.ReadImage(data))
# prob_arr = sitk.GetArrayFromImage(sitk.ReadImage(data))

# data_tensor = torch.FloatTensor(np.array([[data_arr]])).to(device)

# print(data_tensor.shape)

# mask = model_eval(data_tensor)

# print (mask[0].shape)
# print (mask[0].argmax(dim=1))


# img = sitk.ReadImage(data)
# print(img.GetSpacing())
# print(img.GetOrigin())
# print(img.GetDirection())


# # pdb.set_trace()

# pred_cpu_arr = mask[0].argmax(dim=1).view(128,128,64).detach().cpu().clone().numpy().astype(np.uint16)
# print (pred_cpu_arr)

# # print (pred_cpu.shape)
# # pdb.set_trace()
# pred_mask =sitk.GetImageFromArray(pred_cpu_arr)
# pred_mask.SetSpacing(img.GetSpacing())
# pred_mask.SetDirection(img.GetDirection())
# pred_mask.SetOrigin(img.GetOrigin())

# # pred_mask = pred_mask.CopyInformation(sitk.ReadImage(mask_data))
# sitk.WriteImage( pred_mask, 'pred_00071_1.nii.gz')
