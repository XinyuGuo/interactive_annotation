import SimpleITK as sitk
import numpy as np
import pandas as pd
import SimpleITK as sitk
import os
import pdb

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
    chop kidney from image
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

def chop_img(bboxes, img, mask,save_root):
    '''
    chop 3d image patch from image 
    '''
    cnt = 0 
    for bbox in bboxes:
        # get the bbox element
        x = bbox[0].item()
        y = bbox[1].item()
        z = bbox[2].item()
        xlen = bbox[3].item()
        ylen = bbox[4].item()
        zlen = bbox[5].item()
        kidney_img = img[x:x+xlen, y:y+ylen, z:z+zlen]
        kidney_mask = mask[x:x+xlen, y:y+ylen, z:z+zlen]
        
        img_name = 'kidney_' + str(cnt) + '.nii.gz'
        mask_name ='mask_' + str(cnt) + '.nii.gz'
        img_path = os.path.join(save_root, img_name)
        mask_path = os.path.join(save_root, mask_name)
        cnt+=1        

        sitk.WriteImage(kidney_img,img_path)
        sitk.WriteImage(kidney_mask, mask_path)

def chop_img_with_pading(bboxes, img, mask,save_root, p_voxels):
    '''
    chop 3d image patch from image based on bbox and pading some voxels 
    '''
    cnt = 0 
    for bbox in bboxes:
        # get the bbox element
        x = bbox[0].item()
        y = bbox[1].item()
        z = bbox[2].item()
        xlen = bbox[3].item()
        ylen = bbox[4].item()
        zlen = bbox[5].item()
        
        kidney_img = img[x:x+xlen, y:y+ylen, z:z+zlen]
        kidney_mask = mask[x:x+xlen, y:y+ylen, z:z+zlen]
        
        img_name = 'kidney_' + str(cnt) + '.nii.gz'
        mask_name ='mask_' + str(cnt) + '.nii.gz'
        img_path = os.path.join(save_root, img_name)
        mask_path = os.path.join(save_root, mask_name)
        cnt+=1        

        sitk.WriteImage(kidney_img,img_path)
        sitk.WriteImage(kidney_mask, mask_path)


# script
df = pd.read_csv('kidney_data.csv')

for index, row in df.iterrows():
    print(row['case_id'])
    imgpath = row['image_path']
    maskpath = row['mask_path']
    img = sitk.ReadImage(imgpath)
    mask = sitk.ReadImage(maskpath)
    bboxes = get_kidney_bbox(img, mask)
    save_root = os.path.split(maskpath)[0]
    chop_img(bboxes, img, mask, save_root)

# root_dir = '/data/ccusr/xinyug/annotation/kidney/ori'
# case_dir = 'case_00071'
# case_path = os.path.join(root_dir, case_dir)
# mask_name = 'segmentation.nii.gz'
# mask_path = os.path.join(case_path, mask_name)
# mask = sitk.ReadImage(mask_path)
# img = sitk.ReadImage(os.path.join(root_dir, 'case_00071/imaging.nii.gz'))

# bboxes = get_kidney_bbox(img, mask)
# # print(bboxes)
# chop_img(bboxes, img, mask, os.path.join(root_dir, case_dir))