import pandas as pd
import numpy as np
import os

# save image path, mask, kidney image path, mask path to csv
root_dir = '/data/ccusr/xinyug/annotation/kidney/ori'
dirs = os.listdir(root_dir)
dir_paths = [ os.path.join(root_dir, d) for d in dirs if not os.path.isfile(os.path.join(root_dir, d))]
dir_ids = [d for d in dirs if not os.path.isfile(os.path.join(root_dir, d))]
img_name = 'imaging.nii.gz'
mask_name = 'segmentation.nii.gz'
img_paths = [os.path.join(dir_path, img_name) for dir_path in dir_paths]
mask_paths = [os.path.join(dir_path, mask_name) for dir_path in dir_paths]

k_0_path = []
k_1_path = []
m_0_path = []
m_1_path = []

m_0_prob_path = []
m_1_prob_path = []

k_0_r_path = []
k_1_r_path = []
m_0_r_path = []
m_1_r_path = []

m_0_prob_r_path = []
m_1_prob_r_path = []

# sigma = 10, probability map
m_0_prob_r_10_path = []
m_1_prob_r_10_path = []

for dir_path in dir_paths:
    dir_files = os.listdir(dir_path)
    if 'kidney_0.nii.gz' in dir_files:
        k_0_path.append(os.path.join(dir_path, 'kidney_0.nii.gz'))
    else:
        k_0_path.append('no')

    if 'kidney_1.nii.gz' in dir_files:
        k_1_path.append(os.path.join(dir_path, 'kidney_1.nii.gz'))
    else:
        k_1_path.append('no')

    if 'mask_0.nii.gz' in dir_files:
        m_0_path.append(os.path.join(dir_path, 'mask_0.nii.gz'))
    else:
        m_0_path.append('no')
       
    if 'mask_1.nii.gz' in dir_files:
        m_1_path.append(os.path.join(dir_path, 'mask_1.nii.gz'))
    else:
        m_1_path.append('no')

    if 'prob_map_0.nii.gz' in dir_files:
        m_0_prob_path.append(os.path.join(dir_path, 'prob_map_0.nii.gz'))
    else:
        m_0_prob_path.append('no')

    if 'prob_map_1.nii.gz' in dir_files:
        m_1_prob_path.append(os.path.join(dir_path, 'prob_map_1.nii.gz'))
    else:
        m_1_prob_path.append('no')

    # 
    if 'kidney_0_resampled.nii.gz' in dir_files:
        k_0_r_path.append(os.path.join(dir_path, 'kidney_0_resampled.nii.gz'))
    else:
        k_0_r_path.append('no')

    if 'kidney_1_resampled.nii.gz' in dir_files:
        k_1_r_path.append(os.path.join(dir_path, 'kidney_1_resampled.nii.gz'))
    else:
        k_1_r_path.append('no')

    if 'mask_0_resampled.nii.gz' in dir_files:
        m_0_r_path.append(os.path.join(dir_path, 'mask_0_resampled.nii.gz'))
    else:
        m_0_r_path.append('no')
       
    if 'mask_1_resampled.nii.gz' in dir_files:
        m_1_r_path.append(os.path.join(dir_path, 'mask_1_resampled.nii.gz'))
    else:
        m_1_r_path.append('no')

    if 'prob_map_0_resampled.nii.gz' in dir_files:
        m_0_prob_r_path.append(os.path.join(dir_path, 'prob_map_0_resampled.nii.gz'))
    else:
        m_0_prob_r_path.append('no')

    if 'prob_map_1_resampled.nii.gz' in dir_files:
        m_1_prob_r_path.append(os.path.join(dir_path, 'prob_map_1_resampled.nii.gz'))
    else:
        m_1_prob_r_path.append('no')

    # sigma = 10 
    # m_0_prob_r_10_path = []
    # m_1_prob_r_10_path = []
    if 'prob_map_0_resampled_10.nii.gz' in dir_files:
        m_0_prob_r_10_path.append(os.path.join(dir_path, 'prob_map_0_resampled_10.nii.gz'))
    else:
        m_0_prob_r_10_path.append('no')

    if 'prob_map_1_resampled_10.nii.gz' in dir_files:
        m_1_prob_r_10_path.append(os.path.join(dir_path, 'prob_map_1_resampled_10.nii.gz'))
    else:
        m_1_prob_r_10_path.append('no')
    
csv_dict = {'case_id':dir_ids, 'image_path':img_paths, 'mask_path':mask_paths, 'kidney_0':k_0_path, 'kidney_1':k_1_path, \
            'mask_0':m_0_path, 'mask_1':m_1_path, 'prob_0': m_0_prob_path, 'prob_1': m_1_prob_path, \
            'kidney_0_resampled': k_0_r_path, 'kidney_1_resampled':k_1_r_path, 'mask_0_resampled': m_0_r_path, \
            'mask_1_resampled' : m_1_r_path , 'prob_resampled_1':m_1_prob_r_path, 'prob_resampled_0':m_0_prob_r_path, \
            'prob_resmapled_1_10':m_1_prob_r_10_path, 'prob_resmapled_0_10':m_0_prob_r_10_path}

csv_df = pd.DataFrame(csv_dict)
csv_df.to_csv('kidney_data.csv',index=False)