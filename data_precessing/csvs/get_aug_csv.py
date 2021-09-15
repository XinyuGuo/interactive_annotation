import pandas as pd  
import os
import pdb 

# data_path = './kidney/aug'
proj_root = os.path.abspath(os.getcwd())
# data_path = os.path.join(proj_root,'kidney/aug')
data_path = os.path.join(proj_root,'kidney/aug_multiC')
print(data_path)

dirs = os.listdir(data_path)
dir_paths = [os.path.join(data_path,dir) for dir in dirs]
dirs_paths = zip(dirs, dir_paths)
imgs_paths = []
masks_paths = []
image_ids = []
for dir_path in dirs_paths:
    print(dir_path)
    dir = dir_path[0]
    dir_path = dir_path[1]
    filenum = len(os.listdir(dir_path))
    aug_num = filenum // 2
    for i in range(aug_num):
        # print(i)
        img_name = dir + '_imgaug_' + str(i) + '.nii.gz'
        mask_name = dir + '_maskaug_' + str(i) + '.nii.gz'        
        imgs_paths.append(os.path.join(dir_path,img_name))
        masks_paths.append(os.path.join(dir_path,mask_name))
        image_ids.append(dir + '_' + str(i))
    # pdb.set_trace()
df = pd.DataFrame({'case_img_id':image_ids, 'aug_kidney_path': imgs_paths, 'aug_mask_path':masks_paths})
df.to_csv('aug_data_multiC.csv',index=False)