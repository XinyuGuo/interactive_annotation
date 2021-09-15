from kidney_dataloader_with_aug import KidneyDatasetAug
from batchgenerators.dataloading.data_loader import DataLoader
import pandas as pd
import numpy as np
from batchgenerators.dataloading import MultiThreadedAugmenter
from batchgenerators.transforms import Compose
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2, MirrorTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform
import SimpleITK as sitk
from tqdm import tqdm
import os
import math
import pdb

def save_data(img_np, mask_np, case_id, meta, save_root, case_aug_times):
    s = meta['spacing']
    o = meta['origin']
    d = meta['direction'] 
    img = sitk.GetImageFromArray(img_np)
    img.SetDirection(d)
    img.SetOrigin(o)
    img.SetSpacing(s)
    img_name = case_id + '_' + 'imgaug'+'_'+str(case_aug_times[case_id])+'.nii.gz'
    sitk.WriteImage(img, os.path.join(save_root,img_name))
    mask = sitk.GetImageFromArray(mask_np)
    mask.SetDirection(d)
    mask.SetOrigin(o)
    mask.SetSpacing(s)
    mask_name = case_id + '_' + 'maskaug'+'_'+str(case_aug_times[case_id])+'.nii.gz'
    sitk.WriteImage(mask, os.path.join(save_root,mask_name))
    case_aug_times[case_id]+=1
      
def save_aug(aug_batch, save_dir, case_aug_times):
    data_arr = aug_batch['data']
    data_arr = np.squeeze(data_arr, axis=1)
    mask_arr = aug_batch['seg']
    mask_arr = np.squeeze(mask_arr, axis=1)
    meta = aug_batch['case_meta']
    case_id = aug_batch['case_id']
    b_n = data_arr.shape[0]
    for i in range(b_n):
        # print(i)
        # print(case_id[i])
        case_path = os.path.join(save_root, case_id[i])
        if not os.path.exists(case_path):
            os.mkdir(case_path)
        save_data(data_arr[i], mask_arr[i],  case_id[i], meta[i], case_path, case_aug_times)

# def get_train_transform():
#     '''
#     define transforms
#     '''
#     transforms = []
#     transforms.append(BrightnessMultiplicativeTransform((0.7, 1.5), per_channel=True, p_per_sample=1))
#     return transforms


def get_train_transform():
    '''
    define transforms
    '''
    transforms = []
    # mirror
    # transforms.append(MirrorTransform(axes=(0, 1, 2)))

    # scale & rotation
    transforms.append(
        SpatialTransform_2(
            patch_size=(128,128,64),
            do_elastic_deform=True, deformation_scale=(0, 0.25),
            do_rotation=True,
            angle_x=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            angle_y=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            angle_z=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            do_scale=True, scale=(0.75, 1.25),
            border_mode_data='constant', border_cval_data=0,
            border_mode_seg='constant', border_cval_seg=0,
            order_seg=0, order_data=3,
            random_crop=False,
            p_rot_per_sample=0.1, p_scale_per_sample=0.1
        )
    )    
    # brightness
    transforms.append(BrightnessMultiplicativeTransform((0.7, 1.5), per_channel=True, p_per_sample=0.15))

    # gamma correction 
    transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=False, per_channel=True, p_per_sample=0.15))

    # gaussian noise
    transforms.append(GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_sample=0.15)) 

    return transforms

# prepare the data augmentation list
csv_path = '../one_chanel_model_pred_on_train.csv'
pred_res_df = pd.read_csv(csv_path)
hard_cases_df = pred_res_df[pred_res_df['case_iou']<0.7]
case_aug_list = hard_cases_df['case_id'].tolist()
train_csv = '../train_weighted_dice.csv'
train_df = pd.read_csv(train_csv)
indices = []
case_aug_times = {}
for index, row in train_df.iterrows():
    if  row['case_img_id'] in case_aug_list:
        indices.append(index)
        print(row['case_img_id'])
        case_aug_times.update({row['case_img_id']:0})
aug_data_df = train_df.iloc[indices]


all_transforms = Compose(get_train_transform())
batch_size = 4
kidney_aug_loader = KidneyDatasetAug(aug_data_df, batch_size = batch_size)
train_gen = MultiThreadedAugmenter(kidney_aug_loader, all_transforms, num_processes=4,\
                                   num_cached_per_queue=2,\
                                   seeds=None, pin_memory=False)

# generate and save augmented cases
save_root = '../kidney/aug_multiC'
if not os.path.exists(save_root):
    os.mkdir(save_root)

batch_num = math.ceil(len(aug_data_df)/batch_size) 
for i in range(20):
    print(i)
    for j in tqdm(range(batch_num)):
        aug_batch = train_gen.next()
        save_aug(aug_batch, save_root, case_aug_times)