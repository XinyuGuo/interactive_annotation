import torch
from model import Modified3DUNet
from torch.utils.data import DataLoader
from kidney_dataloader_1_c import KidneyDataset_1_c
import evaluate_tools
from tqdm import tqdm
import SimpleITK as sitk
import numpy as np
import random
from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.morphology import distance_transform_cdt
import SimpleITK as sitk
import os
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
    M = np.zeros(shape)

    for i in range(points.shape[1]):
        center = points[:,i]
        x0 = center[0]
        y0 = center[1]
        z0 = center[2]
        temp_probs = np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2 + (z-z0)**2) / sigma ** 2)#.astype(d_type)   
        M = np.maximum(temp_probs, M)
    return M

def get_prob_maps(keys, clicks_info, shape, sigma):
    '''
    get probility map during the trainng
    '''
    prob_maps_fp = []
    prob_maps_fn = []
    for key in keys:
        clicks = clicks_info[key]
        clicks = np.concatenate(clicks,axis=1)
        fp_clicks = clicks[:,::2]
        fn_clicks = clicks[:,1::2] 
        M_fp = get_prob_map(fp_clicks, shape, sigma)
        M_fn = get_prob_map(fn_clicks, shape, sigma)
        prob_maps_fn.append(M_fn)
        prob_maps_fp.append(M_fp)
    prob_maps_fp_arr = np.stack(prob_maps_fp).astype(np.float32)
    prob_maps_fn_arr = np.stack(prob_maps_fn).astype(np.float32)
    probs_M = np.stack([prob_maps_fp_arr,prob_maps_fn_arr],axis=1)
    return  probs_M.astype(np.float32)

def get_prob_maps_sim(clicks_arr,shape,sigma):
    '''
    generate probability maps based on click information
    '''
    batch_size = clicks_arr.shape[0]
    case_maps = []
    for i in range(batch_size):
        fp_points = clicks_arr[i][:,::2]
        fn_points = clicks_arr[i][:,1::2]
        M_fp = get_prob_map(fp_points, shape, sigma)
        M_fn = get_prob_map(fn_points, shape, sigma)        
        M = np.stack([M_fp,M_fn])
        case_maps.append(M)
    case_maps = np.stack(case_maps) 
    return case_maps.astype(np.float32)

def get_fp_fn_masks_batch(pred, gt, device):
    '''
    get fp mask and fn mask
    '''
    fp_tensor = ((pred>0.5)&(gt==0)).type(torch.uint8)
    fn_tensor = ((pred<=0.5)&(gt==1)).type(torch.uint8)
    fp_batch = fp_tensor.cpu().detach().numpy() 
    fn_batch = fn_tensor.cpu().detach().numpy()
    return fp_batch, fn_batch

def get_largest_cc_mask(arr):
    '''
    get largest connected component mask    
    '''
    # get connected component
    struc = generate_binary_structure(3,3)
    labeled_arr , _ = label(arr, struc)
    assert(labeled_arr.max()!=0)
    # find largest component and get its mask
    largest_cc_mask = labeled_arr == np.argmax(np.bincount(labeled_arr.flat)[1:])+1
    largest_cc_mask = largest_cc_mask.astype(np.uint8)
    return largest_cc_mask, labeled_arr

def get_chamfer_map_click(mask):
    '''
    get chamfer distance based on the mask
    '''
    chamfer_map = distance_transform_cdt(mask)  
    points = np.where(chamfer_map==np.max(chamfer_map))
    lower = 0
    upper = points[0].shape[0] - 1
    index = random.randint(lower, upper)
    click = np.array([arr[index] for arr in points]) 
    return click, chamfer_map

def get_clicks_chamfer_init_without_largets_cc(fp_batch, fn_batch, img_ids, mask_meta):
    '''
    get one click on chamfer distance for a batch data
    '''
    batch_size = fp_batch.shape[0]
    clicks_list = []
    for i in range(batch_size):        
        cur_fp_arr = fp_batch[i]
        cur_fn_arr = fn_batch[i]
        fp_click, _ = get_chamfer_map_click(cur_fp_arr)
        fn_click, _ = get_chamfer_map_click(cur_fn_arr)
        clicks = np.stack([fp_click,fn_click]).swapaxes(1,0)
        clicks_list.append(clicks)
    return clicks_list

def get_weighted_maps(fp_batch, fn_bath):
    '''
    get the initial weighted map, and the average probability map
    '''
    correct_map_init = (fp_batch + fn_bath).astype(np.bool_)
    correct_map_init = ~correct_map_init
    # return correct_map_init.astype(np.float32)
    return correct_map_init.astype(np.uint8)

def save_weighted_maps_masks_npy(weighted_maps,mask_tensor,save_root,img_ids):
    # mask_arrs = mask_tensor.cpu().detach().numpy()
    for i in range(len(img_ids)):
        weighted_arr = weighted_maps[i] # (128,128,64)
        # mask_arr = mask_arrs[i] # (128, 128, 64)
        # mask_name =img_ids[i] + '_mask.npy'
        weighted_map_name = img_ids[i] + '_init_map.npy'
        # mask_path = os.path.join(save_root, mask_name)
        weighted_map_path = os.path.join(save_root, weighted_map_name)
        # np.save(mask_path,mask_arr)
        np.save(weighted_map_path,weighted_arr)
        
def save_weighted_maps_masks_img(weighted_maps,mask_tensor,mask_meta,save_root,img_ids):
    mask_arrs = mask_tensor.cpu().detach().numpy()
    mask_arrs[mask_arrs>0.5]=1
    mask_arrs[mask_arrs<=0.5]=0
    for i in range(len(img_ids)):
        # extract the origin, spacing, and direction
        meta_o = torch.tensor([tensor[i] for tensor in mask_meta['origin']]).numpy()
        meta_s = torch.tensor([tensor[i] for tensor in mask_meta['spacing']]).numpy()
        meta_d = torch.tensor([tensor[i] for tensor in mask_meta['direction']]).numpy()
        weighted_arr = weighted_maps[i] # (128,128,64)
        mask_arr = mask_arrs[i] # (128, 128, 64)
        # save mask probability and intitail weighted map to img.
        mask_img = sitk.GetImageFromArray(mask_arr)
        weighted_map_img = sitk.GetImageFromArray(weighted_arr)
        # mask_img.SetSpacing(meta_s)
        # mask_img.SetOrigin(meta_o)
        # mask_img.SetDirection(meta_d)
        weighted_map_img.SetSpacing(meta_s)
        weighted_map_img.SetOrigin(meta_o)
        weighted_map_img.SetDirection(meta_d)
        # mask_name =img_ids[i] + '_mask.nii.gz'
        weighted_map_name = img_ids[i] + '_init_map.nii.gz'
        # mask_path = os.path.join(save_root, mask_name)
        weighted_map_path = os.path.join(save_root, weighted_map_name)
        # sitk.WriteImage(mask_img,mask_path)
        sitk.WriteImage(weighted_map_img,weighted_map_path)

def sim_clicks_init_batch(img_ids, data, mask, mask_meta, model, save_root, device):
    '''
    simulate clicks before the training for the batch data
    '''
    _, seglayer, = model(torch.unsqueeze(data,1).to(device))
    mask_tensor = torch.sigmoid(seglayer)
    mask_tensor = torch.squeeze(mask_tensor, 1)
    mask = torch.squeeze(mask,1).to(device) 
    fp_batch, fn_batch = get_fp_fn_masks_batch(mask_tensor, mask.to(device), device)
    weighted_maps = get_weighted_maps(fp_batch, fn_batch)
    save_weighted_maps_masks_npy(weighted_maps,mask_tensor,save_root,img_ids)
    save_weighted_maps_masks_img(weighted_maps,mask_tensor,mask_meta,save_root,img_ids)
    
    # get_clicks_chamfer_init_without_largets_cc
    clicks = get_clicks_chamfer_init_without_largets_cc(fp_batch, fn_batch, img_ids, mask_meta)
    return clicks

def fill_the_dict_init(clicks_info_init, clicks_list, key_list):
    '''
    fill the dict with clicks
    '''
    for i in range(len(key_list)):
        clicks_info_init[key_list[i]]  = [clicks_list[i]]

def sim_initial_click_dataset_batch(dataloader, model_init, sigma, save_root, device):
    '''
    simulate initial clicks for the dataset
    '''
    model_init.eval()
    clicks_info_init = {}
    for sim_batch in tqdm(dataloader):
        clicks_list = sim_clicks_init_batch(sim_batch['kidney_id'], sim_batch['data'],sim_batch['mask'],sim_batch['mask_meta'],\
                                            model_init, save_root, device)
        # fill in the dictionary.
        fill_the_dict_init(clicks_info_init,clicks_list,sim_batch['kidney_id'])
    model_init.train()
    return clicks_info_init

def get_batch_clicks(img_keys, clicks_info):
    '''
    get click list
    '''
    all_clicks = []
    for img_key in img_keys:
        all_clicks.append(np.concatenate(clicks_info[img_key],axis=1))
    all_clicks = np.stack(all_clicks)
    return all_clicks

def get_clicks_chamfer_without_largest_cc(fp_batch, fn_batch, img_ids, mask_meta, click_num):
    '''
    get one click on chamfer distance for a batch data
    '''
    batch_size = fp_batch.shape[0]
    clicks_list = []
    for i in range(batch_size):
        cur_fp_arr = fp_batch[i]
        cur_fn_arr = fn_batch[i]
        fp_click, _ = get_chamfer_map_click(cur_fp_arr)
        fn_click, _ = get_chamfer_map_click(cur_fn_arr)
        clicks = np.stack([fp_click,fn_click]).swapaxes(1,0)
        clicks_list.append(clicks)
    return clicks_list

def sim_clicks(batch_data, model, clicks_info, sigma, device, click_num):
    '''
    simulate clicks during the training (4 chanel model)
    data : tensor
    mask : tnesor
    clicks_info : list of array
    '''
    # for key in img_keys:
    train_data_1_c = torch.unsqueeze(batch_data['data'].to(device),1)
    img_ids = batch_data['kidney_id']
    mask_meta = batch_data['mask_meta']
    all_clicks = get_batch_clicks(img_ids,clicks_info) # (3, batch_size *2) numpy arr
    train_maps = torch.tensor(get_prob_maps_sim(all_clicks,batch_data['mask'].shape[1:4],sigma)).to(device)
    train_data = torch.cat((train_data_1_c, train_maps),1)
    _, seglayer = model(train_data)
    pred = torch.sigmoid(seglayer)
    pred = torch.squeeze(pred,1)
    mask = batch_data['mask'].to(device)
    fp_batch, fn_batch = get_fp_fn_masks_batch(pred, mask, device)
    weighted_maps_batch = get_weighted_maps(fp_batch, fn_batch)
    clicks = get_clicks_chamfer_without_largest_cc(fp_batch, fn_batch, img_ids, mask_meta, click_num)
    return clicks, weighted_maps_batch,pred

def fill_the_dict(clicks_info, clicks_list, key_list):
    '''
    fill the dictionary
    '''
    for i in range(len(key_list)):
        clicks_info[key_list[i]].append(clicks_list[i])

def sim_clicks_batch(sim_batch, clicks_info, model, num_points, sigma, res_map_root, device):
    key_list = sim_batch['kidney_id']
    init_w_m_batch = sim_batch['init_weighted_map']
    init_w_m_batch = init_w_m_batch.cpu().detach().numpy()
    mask_meta = sim_batch['mask_meta']
    for k in range(num_points):
        # print(k)
        clicks_list, weighted_maps_batch, pred = sim_clicks(sim_batch, model, clicks_info, sigma, device, k) 
        init_w_m_batch+=weighted_maps_batch
        fill_the_dict(clicks_info, clicks_list, key_list)
        
        # save masks - debug purpose
        # pred = pred.cpu().detach().numpy()
        # pred[pred>0.5]=1
        # pred[pred<=0.5]=0
       
        # for j in range(len(key_list)):
        #     mask_meta_o = torch.tensor([tensor[j] for tensor in mask_meta['origin']]).numpy()
        #     mask_meta_s = torch.tensor([tensor[j] for tensor in mask_meta['spacing']]).numpy()
        #     mask_meta_d = torch.tensor([tensor[j] for tensor in mask_meta['direction']]).numpy()

        #     pred_img = sitk.GetImageFromArray(pred[j])
        #     pred_img.SetSpacing(mask_meta_s)
        #     pred_img.SetOrigin(mask_meta_o)
        #     pred_img.SetDirection(mask_meta_d)

        #     pred_name = key_list[j] + '_pred_' + str(k+1) + '.nii.gz'
        #     pred_path = os.path.join(res_map_root, pred_name)
        #     sitk.WriteImage(pred_img,pred_path)

    # save weighted map
    for i in range(len(key_list)):
        meta_o = torch.tensor([tensor[i] for tensor in mask_meta['origin']]).numpy()
        meta_s = torch.tensor([tensor[i] for tensor in mask_meta['spacing']]).numpy()
        meta_d = torch.tensor([tensor[i] for tensor in mask_meta['direction']]).numpy()
        map_name = key_list[i] + '_res_map.npy'
        img_name = key_list[i] + '_res_map.nii.gz'
        np_map_path = os.path.join(res_map_root, map_name)
        img_path = os.path.join(res_map_root, img_name)
        case_res_img = sitk.GetImageFromArray(init_w_m_batch[i])
        case_res_img.SetSpacing(meta_s)
        case_res_img.SetOrigin(meta_o)
        case_res_img.SetDirection(meta_d)
        sitk.WriteImage(case_res_img,img_path)
        np.save(np_map_path,init_w_m_batch[i])

def sim_clicks_dataset_batch(dataloader,clicks_info_init,model,num_points,sigma,res_map_root,device):
    '''
    simulate  clicks during the trianing
    '''
    model.eval()
    for sim_batch in tqdm(dataloader):
        # print(sim_batch['data'].shape)
        # pdb.set_trace()
        sim_clicks_batch(sim_batch, clicks_info_init, model, num_points, sigma, res_map_root, device)
    model.train()
    return clicks_info_init

####################################--functions are not active--#################################
def get_clicks_chamfer(fp_batch, fn_batch, img_ids, mask_meta, save_root, click_num):
    '''
    get one click on chamfer distance for a batch data
    '''
    batch_size = fp_batch.shape[0]
    clicks_list = []
    for i in range(batch_size):
        largest_fp_mask, labeled_arr_fp = get_largest_cc_mask(fp_batch[i])
        largest_fn_mask, labeled_arr_fn = get_largest_cc_mask(fn_batch[i])
        
        dir_path = os.path.join(save_root,img_ids[i])
        meta = {}
        meta['origin'] = torch.tensor([tensor[i] for tensor in mask_meta['origin']])
        meta['spacing'] = torch.tensor([tensor[i] for tensor in mask_meta['spacing']])
        meta['direction'] = torch.tensor([tensor[i] for tensor in mask_meta['direction']])
        # save fp fn connected component map
        save_fp_fn_arr_to_nii(labeled_arr_fp, labeled_arr_fn, meta, dir_path, click_num+1, 'fp_fn')
        fp_click, c_map_fp = get_chamfer_map_click(largest_fp_mask)
        fn_click, c_map_fn = get_chamfer_map_click(largest_fn_mask)
        # save chamfer map
        save_fp_fn_arr_to_nii(c_map_fp, c_map_fn, meta, dir_path, click_num+1, 'chamfer')
        clicks = np.stack([fp_click,fn_click]).swapaxes(1,0)
        clicks_list.append(clicks)
    return clicks_list

def get_clicks_chamfer_init(fp_batch, fn_batch, img_ids, mask_meta, save_root):
    '''
    get one click on chamfer distance for a batch data
    '''
    batch_size = fp_batch.shape[0]
    clicks_list = []
    for i in range(batch_size):
        largest_fp_mask, labeled_arr_fp = get_largest_cc_mask(fp_batch[i])
        largest_fn_mask, labeled_arr_fn = get_largest_cc_mask(fn_batch[i])
        
        # save fp fn mask for checking
        dir_path = os.path.join(save_root,img_ids[i])
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        # print(mask_meta[i])
        meta = {}
        meta['origin'] = torch.tensor([tensor[i] for tensor in mask_meta['origin']])
        meta['spacing'] = torch.tensor([tensor[i] for tensor in mask_meta['spacing']])
        meta['direction'] = torch.tensor([tensor[i] for tensor in mask_meta['direction']])
        # save fp fn connected component map
        save_fp_fn_arr_to_nii(labeled_arr_fp, labeled_arr_fn, meta, dir_path, 0, 'fp_fn')

        fp_click, c_map_fp = get_chamfer_map_click(largest_fp_mask)
        fn_click, c_map_fn = get_chamfer_map_click(largest_fn_mask)

        # save chamfer map
        save_fp_fn_arr_to_nii(c_map_fp, c_map_fn, meta, dir_path, 0, 'chamfer')

        clicks = np.stack([fp_click,fn_click]).swapaxes(1,0)
        clicks_list.append(clicks)
        
    return 

def fill_the_iou_dict_init(all_ious, ious, key_list):
    '''
    fill the dict with ious
    '''
    for i in range(len(key_list)):
        all_ious[key_list[i]] = [ious[i].cpu().numpy().astype(float)]

def getIOU_two_classes(pred,gt):
    '''
    get IOU between predicted labels and ground truth
    '''
    tp_num = torch.sum((pred==1)&(gt==1),dim = (1,2,3)).float()
    pre_num = torch.sum(pred==1,dim=(1,2,3)).float()
    gt_num = torch.sum(gt==1,dim=(1,2,3)).float()
    iou = tp_num / (pre_num + gt_num - tp_num)
    return iou

def fill_the_iou_dict(all_ious, ious, key_list):
    '''
    fill the iou for each case
    '''
    for i in range(len(key_list)):
        all_ious[key_list[i]].append(ious[i].cpu().numpy().astype(float))

def save_fp_fn_arr_to_nii(fp_arr, fn_arr, mask_meta, save_dir, i, type):
    '''
    save fp_map and fn_map
    '''
    fp_img = sitk.GetImageFromArray(fp_arr)
    fn_img = sitk.GetImageFromArray(fn_arr)

    fp_img.SetSpacing(mask_meta['spacing'].numpy())
    fp_img.SetOrigin(mask_meta['origin'].numpy())
    fp_img.SetDirection(mask_meta['direction'].numpy())
    fn_img.SetSpacing(mask_meta['spacing'].numpy())
    fn_img.SetOrigin(mask_meta['origin'].numpy())
    fn_img.SetDirection(mask_meta['direction'].numpy())

    if type == 'fp_fn':
        fp_file_name = 'fp_' + str(i) + '_map.nii.gz'
        fn_file_name = 'fn_' + str(i) + '_map.nii.gz'
    
    if type == 'chamfer':
        fp_file_name = 'fp_' + str(i) + '_chamfer.nii.gz'
        fn_file_name = 'fn_' + str(i) + '_chamfer.nii.gz'
    
    fp_path = os.path.join(save_dir, fp_file_name)
    fn_path = os.path.join(save_dir, fn_file_name)    
    sitk.WriteImage(fn_img, fn_path)
    sitk.WriteImage(fp_img, fp_path)

def save_click_prob_map(fp_arr,fn_arr,mask_meta,save_root,click_num):
    '''
    save a pair of map
    '''
    fp_img = sitk.GetImageFromArray(fp_arr)
    fn_img = sitk.GetImageFromArray(fn_arr)

    fp_img.SetSpacing(mask_meta['spacing'].numpy())
    fp_img.SetOrigin(mask_meta['origin'].numpy())
    fp_img.SetDirection(mask_meta['direction'].numpy())
    fn_img.SetSpacing(mask_meta['spacing'].numpy())
    fn_img.SetOrigin(mask_meta['origin'].numpy())
    fn_img.SetDirection(mask_meta['direction'].numpy())

    fp_name = 'fp_'+str(click_num)+'_click.nii.gz'
    fn_name = 'fn_'+str(click_num)+'_click.nii.gz'
    fp_path = os.path.join(save_root, fp_name)
    fn_path = os.path.join(save_root, fn_name)

    sitk.WriteImage(fn_img, fn_path)
    sitk.WriteImage(fp_img, fp_path)

def save_click_prob_maps(val_loader_init,clicks_info_val,sigma,save_root):
    '''
    save click prob maps
    '''
    for batch_sim in tqdm(val_loader_init):
        case_ids = batch_sim['kidney_id']
        mask_meta = batch_sim['mask_meta']
        for id in range(len(case_ids)):
            case_id = case_ids[id]
            case_root = os.path.join(save_root,case_id)
            case_arr = clicks_info_val[case_id]

            meta = {}
            meta['origin'] = torch.tensor([tensor[id] for tensor in mask_meta['origin']])
            meta['spacing'] = torch.tensor([tensor[id] for tensor in mask_meta['spacing']])
            meta['direction'] = torch.tensor([tensor[id] for tensor in mask_meta['direction']])

            shape = batch_sim['mask'].shape[1:4]
            for i in range(len(case_arr)):
                click_arr = case_arr[i]
                fp_arr = np.expand_dims(click_arr[:,0],1)
                fn_arr = np.expand_dims(click_arr[:,1],1)
                fp_map = get_prob_map(fp_arr, shape, sigma)   
                fn_map = get_prob_map(fn_arr, shape, sigma)
                save_click_prob_map(fp_map, fn_map, meta, case_root, i)
        
            # save all fore-ground and back-ground clicks in a seperate channel
            clicks_arr = np.concatenate(case_arr,axis=1)
            fp_points = clicks_arr[:,::2]
            fn_points = clicks_arr[:,1::2]
            fp_whole_map = get_prob_map(fp_points, shape, sigma)   
            fn_whole_map = get_prob_map(fn_points, shape, sigma)
            save_click_prob_map(fp_whole_map, fn_whole_map, meta, case_root, fp_points.shape[1])

def sim_clicks_batch_pred(sim_batch, clicks_info, final_clicks_info, model, num_points, sigma, device): 
    key_list = sim_batch['kidney_id']     
    for k in range(num_points):
        clicks_list = sim_clicks(sim_batch, model, clicks_info, k, sigma, device)
        fill_the_dict(clicks_info, clicks_list, key_list)
    
    for img_key in sim_batch['kidney_id']:
        final_clicks_info[img_key] = clicks_info[img_key][-1]       

def sim_clicks_dataset_batch_pred(dataloader,clicks_info_init,model,num_points,sigma,device):
    model.eval()
    final_clicks_info = {}
    for sim_batch in tqdm(dataloader):
        sim_clicks_batch_pred(sim_batch, clicks_info_init, final_clicks_info, model, num_points, sigma, device)
    # print(final_clicks_info)
    # pdb.set_trace()
    return final_clicks_info