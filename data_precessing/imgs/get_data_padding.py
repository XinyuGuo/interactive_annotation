'''
chop kidney data with more background around the bounding box - multiprocess 
'''
import pandas as pd
import numpy as np
import SimpleITK  as sitk
from multiprocessing import Pool 
import os
import csv

def callback_write_to_csv(paras):
    runtime = paras[0]
    img_id = paras[1]
    if runtime:
        csv_runtime.writerow([img_id])

def realbox(bbox_3d,threshold):
    '''
    judge if the bbox is a real bbox containing the kidney
    '''
    if bbox_3d[3] <= threshold or bbox_3d[4] <= threshold or bbox_3d[5]<=threshold:
        return False
    else:
        return True

def get_kidney_bbox(img, mask):
    '''
    get kidney's bboxes based on the mask
    '''
    threshold =5 # bbox threshold
    mask_binary = sitk.BinaryThreshold(mask,1,2,1)
    connected_filter = sitk.ConnectedComponentImageFilter()
    mask_comp = connected_filter.Execute(mask_binary)
    obj_nums = connected_filter.GetObjectCount()
    obj_labels = range(1,obj_nums+1,1)

    # get 3d bbox for each label
    shape_filter = sitk.LabelShapeStatisticsImageFilter()
    shape_filter.Execute(mask_comp)
    
    bbox_data = []
    for label in obj_labels :
        bbox_3d = np.array(shape_filter.GetBoundingBox(label))
        if not realbox(bbox_3d,threshold):
            continue
        else:
            # print(bbox_3d)
            bbox_data.append(bbox_3d)                
    return bbox_data

def chop_img_with_pading(paras):
    '''
    chop 3d image patch from image based on bbox and pading some voxels 
    '''
    Runtime = False
    
    try:
        img_id = paras[0]
        print(img_id)
        img_path = paras[1]
        mask_path = paras[2]
        p_voxels = 15
        img = sitk.ReadImage(img_path)
        mask = sitk.ReadImage(mask_path)
        bboxes = get_kidney_bbox(img, mask)
        save_root = os.path.split(mask_path)[0]
        img_shape = img.GetSize()
        
        x_size = img_shape[0]
        y_size = img_shape[1]
        z_size = img_shape[2]
        cnt = 0 
        for bbox in bboxes:
            # get the bbox element
            x = bbox[0].item()
            y = bbox[1].item()
            z = bbox[2].item()
            xlen = bbox[3].item()
            ylen = bbox[4].item()
            zlen = bbox[5].item()
            x_start = max(x-p_voxels,0) 
            x_end =  min(x+xlen+p_voxels,x_size)
            y_start = max(y-p_voxels,0) 
            y_end =  min(y+ylen+p_voxels,y_size)
            z_start = max(z-p_voxels,0) 
            z_end =  min(z+zlen+p_voxels,z_size)

            kidney_img = img[x_start:x_end, y_start:y_end, z_start:z_end]
            kidney_mask = mask[x_start:x_end, y_start:y_end, z_start:z_end]
            # kidney_img = img[x:x+xlen, y:y+ylen, z:z+zlen]
            # kidney_mask = mask[x:x+xlen, y:y+ylen, z:z+zlen]
            img_name = 'kidneyp_' + str(cnt) + '.nii.gz'
            mask_name ='maskp_' + str(cnt) + '.nii.gz'
            img_path = os.path.join(save_root, img_name)
            mask_path = os.path.join(save_root, mask_name)
            cnt+=1        

            sitk.WriteImage(kidney_img,img_path)
            sitk.WriteImage(kidney_mask, mask_path)
    except RuntimeError:
        Runtime=True
    return [Runtime,img_id]

# script
# df = pd.read_csv('kidney_data.csv')
df = pd.read_csv('data_chop_test.csv')
ids = df['case_id'].tolist()
image_paths = df['image_path'].tolist()
mask_paths = df['mask_path'].tolist()
img_info = zip(ids,image_paths,mask_paths)
error_file = open('get_data_padding_runtime.csv', 'w')
csv_runtime = csv.writer(error_file)

p = Pool(16)
for i in range(len(img_info)):
    p.apply_async(chop_img_with_pading, (img_info[i],), callback=callback_write_to_csv)
p.close()
p.join()
# chop_img_with_pading(bboxes, img, mask, save_root, pad_voxels)