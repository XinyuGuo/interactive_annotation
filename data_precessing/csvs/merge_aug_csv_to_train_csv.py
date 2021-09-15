import pandas as pd  
import os
import pdb

aug_csv_path = 'aug_data_multiC.csv'
csv_path = 'train_weighted_dice_2.csv'

aug_df = pd.read_csv(aug_csv_path)
data_df =  pd.read_csv(csv_path)

kidney_aug_paths = aug_df['aug_kidney_path'].tolist()
kidney_paths = data_df['kidney_resampled_path'].tolist()
k_ps = kidney_aug_paths + kidney_paths 

kidney_aug_mask_paths = aug_df['aug_mask_path'].tolist()
kidney_mask_paths = data_df['mask_resampled_path'].tolist()
m_ps = kidney_aug_mask_paths + kidney_mask_paths 

aug_case_ids = aug_df['case_img_id'].tolist()
case_ids = data_df['case_img_id'].tolist()
c_ids = aug_case_ids + case_ids

aug_df = pd.DataFrame({'case_img_id':c_ids, 'kidney_resampled_path': k_ps, 'mask_resampled_path': m_ps})
aug_df.to_csv('train_aug.csv',index=False)
