import SimpleITK as sitk
import numpy as np
import os
import pandas as pd

def normalize_d(img_path, size):
    '''
    normalize image size
    '''
    img = sitk.ReadImage(img_path)
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputDirection(img.GetDirection())
    # get new spacing
    img_spacing = np.array(img.GetSpacing())
    img_size = np.array(img.GetSize())
    scalars = img_size/size
    new_spacing = scalars*img_spacing
    resampler.SetOutputSpacing(tuple(new_spacing))
    resampler.SetSize(size)
    resampled_img = resampler.Execute(img)
    return resampled_img
    # sitk.WriteImage(resampled_img, save_path)

def normalize_dimention(kidney, mask, size):
    '''
    kidney : kidney file path
    mask: mask file path
    image dimention noralization
    resize nii to a pre-defined size.
    '''
    # build a new dir
    i = normalize_d(kidney,size)
    m = normalize_d(mask,size)
    root = os.path.split(kidney)[0]
    k_name = os.path.split(kidney)[-1]
    if k_name == 'kidney_0.nii.gz':
        new_k_name = 'kidney_0_resampled.nii.gz'
    if k_name == 'kidney_1.nii.gz':
        new_k_name = 'kidney_1_resampled.nii.gz'
    m_name = os.path.split(mask)[-1]
    if m_name == 'mask_0.nii.gz':
        new_m_name = 'mask_0_resampled.nii.gz'
    if m_name == 'mask_1.nii.gz':
        new_m_name = 'mask_1_resampled.nii.gz'
    new_k_path = os.path.join(root, new_k_name)
    new_m_path = os.path.join(root, new_m_name)   
    sitk.WriteImage(i,new_k_path)
    sitk.WriteImage(m,new_m_path)

# root_dir = '/data/ccusr/xinyug/annotation/kidney/ori'
# case_dir = 'case_00071'
# case_path = os.path.join(root_dir, case_dir)
# size = (64,128,128)

# files = os.listdir(case_path)
# if 'kidney_0.nii.gz' in files:
#     kidney_path = os.path.join(case_path,'kidney_0.nii.gz')
#     mask_path = os.path.join(case_path,'mask_0.nii.gz')
#     normalize_dimention(kidney_path, mask_path, size)
# if 'kidney_1.nii.gz' in files:
#     kidney_path = os.path.join(case_path,'kidney_1.nii.gz')
#     mask_path = os.path.join(case_path,'mask_1.nii.gz')
#     normalize_dimention(kidney_path, mask_path, size)    

# script
df = pd.read_csv('kidney_data.csv')
size = (64,128,128)
for index, row in df.iterrows():
    print(row['case_id'])    
    if row['kidney_0'] != 'no':
        kidney_path = row['kidney_0']
        mask_path = row['mask_0']
        normalize_dimention(kidney_path, mask_path, size)
    if row['kidney_1'] != 'no':
        kidney_path = row['kidney_1']
        mask_path = row['mask_1']
        normalize_dimention(kidney_path, mask_path, size)