from torch.utils.data import Dataset, DataLoader
import pandas as pd
import SimpleITK as sitk
import numpy as np
import pdb
import os
from PIL import Image

class KidneyDataset_1_c(Dataset):
    '''kidey dataset for smart annotation'''
    def __init__(self, kidney_csv):
        self.kidney_df = pd.read_csv(kidney_csv)
    
    def __len__(self):
        return len(self.kidney_df)
    
    def __getitem__(self, idx):
        row = self.kidney_df.loc[idx]
        k_r_path = row['kidney_resampled_path']
        m_r_path = row['mask_resampled_path']
        case_img_id = row['case_img_id']
        pred_mask_path = row['pred_mask_paths']
        
        case_id = m_r_path.split('/')[-2]
        img_id = os.path.split(m_r_path)[-1].split('_')[-2]
        kidney_id = case_id + '_' + img_id
        img = sitk.ReadImage(k_r_path)
        mask = sitk.ReadImage(m_r_path) 
        pred_mask = sitk.ReadImage(pred_mask_path)
        img_arr = sitk.GetArrayFromImage(img)
        img_arr = self.normalize_img_arr(img_arr)
        mask_arr = sitk.GetArrayFromImage(mask)
        pred_mask_arr = sitk.GetArrayFromImage(pred_mask)
        mask_arr[mask_arr==2]=0
        sample = {'data': img_arr.astype(np.float32), 'mask': mask_arr.astype(np.float32),\
                  'mask_meta': {'origin': mask.GetOrigin(), 'direction': mask.GetDirection(),
                  'spacing': mask.GetSpacing()}, 'kidney_id': kidney_id, 'pred_mask': pred_mask_arr}
        return sample
    
    def normalize_img_arr(self, img_arr):
        '''
        normalize 
        '''
        MIN_BOUND = -200
        MAX_BOUND = 400
        img_arr = (img_arr-MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        img_arr[img_arr>1]=1
        img_arr[img_arr<0]=0
        return img_arr