import pandas as pd
import SimpleITK as sitk
import numpy as np
import pdb

# script    
csv_train_path = '../train_multiC_aug.csv'
# csv_train_path = '../val_new.csv'
df = pd.read_csv(csv_train_path)

mask_paths = df['mask_resampled_path'].tolist()
# print(mask_paths)
b_vn = 0 #background voxels number
k_vn = 0 #kidney voxels number
t_vn = 0 #tumor voxels number
t_0 = 0

tumor_indices = []
kidney_indices = []
for index, row in df.iterrows():
    # print(row['c1'], row['c2'])
    mask_path = row['mask_resampled_path']
# for mask_path in mask_paths:
    print(mask_path)
    m_img = sitk.ReadImage(mask_path)
    m_arr = sitk.GetArrayFromImage(m_img)
    b = np.sum(m_arr==0)
    k = np.sum(m_arr==1)
    t = np.sum(m_arr==2)
    b_vn+=b
    k_vn+=k
    t_vn+=t
    if t != 0:
        t_0+=1
        tumor_indices.append(index)
    else:
        kidney_indices.append(index)
    print('bnum: ' + str(b) + ' knum: ' + str(k) + ' tnum: ' + str(t))
    
print('background: ' + str(b_vn) + ' kidney: ' + str(k_vn) + ' tumor: ' + str(t_vn))
print('tumor: ' + str(t_0) )
print('whole patients '+ str(df.shape[0]))

tumorcases_df = df.iloc[tumor_indices]
kidney_df = df.iloc[kidney_indices]

tumorcases_df.to_csv('train_tumor_aug.csv', index=False)
kidney_df.to_csv('train_kidney_aug.csv', index=False)
