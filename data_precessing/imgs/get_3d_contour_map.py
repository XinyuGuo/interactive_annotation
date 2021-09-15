import numpy as np
import SimpleITK as sitk
import os
import pandas as pd
import pdb

def get_prob_map(points, shape, sigma):
    '''
    generate the 3d probability map 
    '''
    z = np.arange(0, shape[2], 1, float)
    y = np.arange(0, shape[1], 1, float)
    y = y[:,np.newaxis]
    x = np.arange(0, shape[0], 1, float)
    x = x[:,np.newaxis,np.newaxis]
    # M = x + y + z
    M = np.zeros(shape)
    # points = np.stack(points)

    for i in range(points.shape[1]):
        # print(i)
        center = points[:,i]
        x0 = center[0]
        y0 = center[1]
        z0 = center[2]
        temp_probs = np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2 + (z-z0)**2) / sigma ** 2)#.astype(d_type)   
        M = np.maximum(temp_probs, M)
    return M

# def normalize_img()
def get_3d_contour_prob_map(mask, sigma_value, save_root, index):
    '''
    save prob map for each case 
    '''
    contour_filter = sitk.LabelContourImageFilter()
    contour_filter.SetBackgroundValue(0)
    new_mask = contour_filter.Execute(mask)
    c_arr = sitk.GetArrayFromImage(new_mask)
    indices = np.where(c_arr==1)
    indices_np = np.stack(indices)
    chosen_indices = range(0,indices[0].shape[0],100) # 10
    chosen_points = indices_np[:,chosen_indices]
    
    print(chosen_points.shape)
    pdb.set_trace()

    sigma = sigma_value
    prob_map = get_prob_map(chosen_points, c_arr.shape, sigma)
    prob_img = sitk.GetImageFromArray(prob_map)
    prob_img.CopyInformation(mask)
    # filename = 'prob_map_' + str(index) + '.nii.gz'
    # filename = 'prob_map_' + str(index) + '_resampled.nii.gz'
    filename = 'prob_map_' + str(index) + '_resampled_10.nii.gz'
    sitk.WriteImage(prob_img, os.path.join(save_root, filename)) 

# script
df = pd.read_csv('kidney_data.csv')
sigma_value = 10
for index, row in df.iterrows():
    print(row['case_id'])
    if row['mask_0_resampled'] != 'no':
        mask_path_0 = row['mask_0_resampled']        
        mask_0 = sitk.ReadImage(mask_path_0)
        save_root = os.path.split(mask_path_0)[0]
        get_3d_contour_prob_map(mask_0, sigma_value, save_root, 0)
    else:
        continue
    if row['mask_1_resampled'] != 'no':
        mask_path_1 = row['mask_1_resampled']        
        mask_1 = sitk.ReadImage(mask_path_1)
        save_root = os.path.split(mask_path_1)[0]
        get_3d_contour_prob_map(mask_1, sigma_value, save_root, 1)
    else:
        continue

# root_dir = '/data/ccusr/xinyug/annotation/kidney/ori'
# case_dir = 'case_00071'
# case_path = os.path.join(root_dir, case_dir)
# mask_name = 'mask_0.nii.gz'
# mask_path = os.path.join(case_path, mask_name)
# mask = sitk.ReadImage(mask_path)
# # print(mask_path)
# get_3d_contour_prob_map(mask)