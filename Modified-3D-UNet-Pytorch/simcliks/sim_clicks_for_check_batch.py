import torch
from model import Modified3DUNet
from torch.utils.data import DataLoader
# from kidney_dataloader_pred import KidneyDatasetPred
from kidney_dataloader_1_c import KidneyDataset_1_c
import evaluate_tools
from tqdm import tqdm
import SimpleITK as sitk
import numpy as np
import random
from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.morphology import distance_transform_cdt
# from sklearn_extra.cluster import KMedoids
from sklearn.cluster import AgglomerativeClustering


from sklearn.cluster import KMeans
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
    
def get_clicks_chamfer_init_without_largets_cc(fp_batch, fn_batch, img_ids, mask_meta, save_root):
    '''
    get one click on chamfer distance for a batch data
    '''
    batch_size = fp_batch.shape[0]
    clicks_list = []
    for i in range(batch_size):
        # largest_fp_mask, labeled_arr_fp = get_largest_cc_mask(fp_batch[i])
        # largest_fn_mask, labeled_arr_fn = get_largest_cc_mask(fn_batch[i])
        
        cur_fp_arr = fp_batch[i]
        cur_fn_arr = fn_batch[i]
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
        # save_fp_fn_arr_to_nii(labeled_arr_fp, labeled_arr_fn, meta, dir_path, 0, 'fp_fn')
        save_fp_fn_arr_to_nii(cur_fp_arr, cur_fn_arr, meta, dir_path, 0, 'fp_fn')


        # fp_click, c_map_fp = get_chamfer_map_click(largest_fp_mask)
        # fn_click, c_map_fn = get_chamfer_map_click(largest_fn_mask)

        fp_click, c_map_fp = get_chamfer_map_click(cur_fp_arr)
        fn_click, c_map_fn = get_chamfer_map_click(cur_fn_arr)

        # save chamfer map
        save_fp_fn_arr_to_nii(c_map_fp, c_map_fn, meta, dir_path, 0, 'chamfer')

        clicks = np.stack([fp_click,fn_click]).swapaxes(1,0)
        clicks_list.append(clicks)
        
    return clicks_list

def get_clicks_k_means(fp_batch, fn_batch, img_ids, mask_meta, save_root, k_th):
    '''
    simulate clicks based on KMeans clustering algorithm
    '''
    batch_size = fp_batch.shape[0]
    clicks_list = []
    for i in range(batch_size):
        cur_fp_arr = fp_batch[i]
        cur_fn_arr = fn_batch[i]
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
        save_fp_fn_arr_to_nii(cur_fp_arr, cur_fn_arr, meta, dir_path, 0, 'fp_fn')

        # get clicks based on k-means cluster
        fp_pos = np.stack(np.where(cur_fp_arr==1),axis=1)
        fn_pos = np.stack(np.where(cur_fn_arr==1),axis=1)
        # fp_KMeans = KMeans(n_clusters=3, random_state=0).fit(fp_pos.astype(np.uint8)) 
        # fn_KMeans = KMeans(n_clusters=3, random_state=0).fit(fn_pos.astype(np.uint8))
        # fp_clicks = fp_KMeans.cluster_centers_
        # fn_clicks = fn_KMeans.cluster_centers_
        # fp_labels = fp_KMeans.labels_ + 1
        # fn_labels = fn_KMeans.labels_ + 1
        fp_KMeans = AgglomerativeClustering(n_clusters=3).fit(fp_pos.astype(np.uint8)) 
        fn_KMeans = AgglomerativeClustering(n_clusters=3).fit(fn_pos.astype(np.uint8))
        
        pdb.set_trace()
        # KMedoids
       
        clicks = np.stack([fp_clicks,fn_clicks], axis=2).swapaxes(1,0)
        clicks = np.reshape(clicks,(3,-1)).astype(np.uint8)
        clicks_list.append(clicks)

        # save cluster results
        cur_fp_arr[(fp_pos[:,0], fp_pos[:,1],fp_pos[:,2])] = fp_labels
        cur_fn_arr[(fn_pos[:,0], fn_pos[:,1],fn_pos[:,2])] = fn_labels
        save_fp_fn_arr_to_nii(cur_fp_arr, cur_fn_arr, meta, dir_path, k_th + 1, 'k_means')
        return clicks_list

def sim_clicks_init_batch(img_ids, data, mask, mask_meta, model, save_root, device):
    '''
    simulate clicks before the training for the batch data
    '''
    _, seglayer, = model(torch.unsqueeze(data,1).to(device))
    mask_tensor = torch.sigmoid(seglayer)
    mask_tensor = torch.squeeze(mask_tensor, 1)
    
    pred_t = mask_tensor
    mask = torch.squeeze(mask,1).to(device) 
    pred_t[pred_t<=0.5] = 0
    pred_t[pred_t>0.5] = 1
    ious = getIOU_two_classes(pred_t,mask)
    fp_batch, fn_batch = get_fp_fn_masks_batch(mask_tensor, mask.to(device), device)
    # get_clicks_chamfer_init_without_largets_cc
    # clicks = get_clicks_chamfer_init_without_largets_cc(fp_batch, fn_batch, img_ids, mask_meta, save_root)
    clicks = get_clicks_k_means(fp_batch, fn_batch, img_ids, mask_meta, save_root, -1)
    return clicks, ious 

def fill_the_dict_init(clicks_info_init, clicks_list, key_list):
    '''
    fill the dict
    '''
    for i in range(len(key_list)):
        clicks_info_init[key_list[i]]  = [clicks_list[i]]

def fill_the_iou_dict_init(all_ious, ious, key_list):
    for i in range(len(key_list)):
        all_ious[key_list[i]] = [ious[i].cpu().numpy().astype(float)]
    # print(all_ious)
    # pdb.set_trace()

def sim_initial_click_dataset_batch(dataloader, model_init, sigma, save_root, device):
    '''
    simulate initial clicks for the dataset
    '''
    model_init.eval()
    clicks_info_init = {}
    all_ious = {}
    for sim_batch in tqdm(dataloader):
        clicks_list,ious = sim_clicks_init_batch(sim_batch['kidney_id'], sim_batch['data'],sim_batch['mask'],sim_batch['mask_meta'],\
                                            model_init, save_root, device)
        # fill in the dictionary.
        fill_the_dict_init(clicks_info_init,clicks_list,sim_batch['kidney_id'])
        fill_the_iou_dict_init(all_ious,ious,sim_batch['kidney_id'])
        # print(clicks_info_init)
        # clicks_info_init[kidney_id] = [clicks]
    model_init.train()
    return clicks_info_init, all_ious

def get_batch_clicks(img_keys, clicks_info):
    '''
    get click list
    '''
    all_clicks = []
    for img_key in img_keys:
        all_clicks.append(np.concatenate(clicks_info[img_key],axis=1))
    all_clicks = np.stack(all_clicks)
    # print(all_clicks.shape)
    return all_clicks

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

def get_clicks_chamfer_without_largest_cc(fp_batch, fn_batch, img_ids, mask_meta, save_root, click_num):
    '''
    get one click on chamfer distance for a batch data
    '''
    batch_size = fp_batch.shape[0]
    clicks_list = []
    for i in range(batch_size):
        # largest_fp_mask, labeled_arr_fp = get_largest_cc_mask(fp_batch[i])
        # largest_fn_mask, labeled_arr_fn = get_largest_cc_mask(fn_batch[i])
        cur_fp_arr = fp_batch[i]
        cur_fn_arr = fn_batch[i]
        
        dir_path = os.path.join(save_root,img_ids[i])
        meta = {}
        meta['origin'] = torch.tensor([tensor[i] for tensor in mask_meta['origin']])
        meta['spacing'] = torch.tensor([tensor[i] for tensor in mask_meta['spacing']])
        meta['direction'] = torch.tensor([tensor[i] for tensor in mask_meta['direction']])
        # save fp fn connected component map
        # save_fp_fn_arr_to_nii(labeled_arr_fp, labeled_arr_fn, meta, dir_path, click_num+1, 'fp_fn')
        # fp_click, c_map_fp = get_chamfer_map_click(largest_fp_mask)
        # fn_click, c_map_fn = get_chamfer_map_click(largest_fn_mask)、

        fp_click, c_map_fp = get_chamfer_map_click(cur_fp_arr)
        fn_click, c_map_fn = get_chamfer_map_click(cur_fn_arr)
        # save chamfer map
        save_fp_fn_arr_to_nii(c_map_fp, c_map_fn, meta, dir_path, click_num+1, 'chamfer')
        clicks = np.stack([fp_click,fn_click]).swapaxes(1,0)
        clicks_list.append(clicks)
    return clicks_list

def getIOU_two_classes(pred,gt):
    '''
    get IOU between predicted labels and ground truth
    '''
    tp_num = torch.sum((pred==1)&(gt==1),dim = (1,2,3)).float()
    pre_num = torch.sum(pred==1,dim=(1,2,3)).float()
    gt_num = torch.sum(gt==1,dim=(1,2,3)).float()
    iou = tp_num / (pre_num + gt_num - tp_num)
    return iou

def sim_clicks(batch_data, model, clicks_info, sigma, save_root, device, click_num):
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

    # calculate ious
    gt = torch.squeeze(batch_data['mask'],1).to(device)
    pred_t = pred
    pred_t[pred>0.5] = 1
    pred_t[pred<=0.5] = 0
    pred_t.type(torch.uint8)
    ious = getIOU_two_classes(pred_t, gt)
    # print(ious)
    # pdb.set_trace()
    mask = batch_data['mask'].to(device)
    fp_batch, fn_batch = get_fp_fn_masks_batch(pred, mask, device)
    #  img_ids, mask_meta, save_root
    # clicks = get_clicks_chamfer(fp_batch, fn_batch, img_ids, mask_meta, save_root, click_num)
    # clicks = get_clicks_chamfer_without_largest_cc(fp_batch, fn_batch, img_ids, mask_meta, save_root, click_num)
    clicks = get_clicks_k_means(fp_batch, fn_batch, img_ids, mask_meta, save_root, click_num)
    
    return clicks, ious

def fill_the_dict(clicks_info, clicks_list, key_list):
    '''
    fill the dictionary
    '''
    for i in range(len(key_list)):
        clicks_info[key_list[i]].append(clicks_list[i])

def fill_the_iou_dict(all_ious, ious, key_list):
    '''
    fill the iou for each case
    '''
    for i in range(len(key_list)):
        all_ious[key_list[i]].append(ious[i].cpu().numpy().astype(float))


def sim_clicks_batch(sim_batch, clicks_info, all_ious, model, num_points, sigma, save_root, device):
    key_list = sim_batch['kidney_id']
    for k in range(num_points):
        # print(k)
        clicks_list, ious = sim_clicks(sim_batch, model, clicks_info, sigma, save_root, device, k) 
        fill_the_dict(clicks_info, clicks_list, key_list)
        fill_the_iou_dict(all_ious, ious, key_list)
        # print()

def sim_clicks_dataset_batch(dataloader,clicks_info_init,all_ious,model,num_points,sigma,save_root,device):
    '''
    simulate  clicks during the trianing
    '''
    model.eval()
    for sim_batch in tqdm(dataloader):
        sim_clicks_batch(sim_batch, clicks_info_init, all_ious, model, num_points, sigma, save_root, device)
    model.train()
    return clicks_info_init, all_ious

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
    if type == 'k_means':
        fp_file_name = 'fp_' + str(i) + '_k_means.nii.gz'
        fn_file_name = 'fn_' + str(i) + '_k_means.nii.gz'
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
                # print(click_arr.shape)
                # pdb.set_trace()
                fp_arr = np.expand_dims(click_arr[:,0],1)
                fn_arr = np.expand_dims(click_arr[:,1],1)
                # print(fp_arr.shape)
                # pdb.set_trace()
                fp_map = get_prob_map(fp_arr, shape, sigma)   
                fn_map = get_prob_map(fn_arr, shape, sigma)
                save_click_prob_map(fp_map, fn_map, meta, case_root, i)
        
            # save all fore-ground and back-ground clicks in a seperate channel
            clicks_arr = np.concatenate(case_arr,axis=1)
            fp_points = clicks_arr[:,::2]
            fn_points = clicks_arr[:,1::2]
            fp_whole_map = get_prob_map(fp_points, shape, sigma)   
            fn_whole_map = get_prob_map(fn_points, shape, sigma)
            # print(fp_points.shape)
            # pdb.set_trace()
            save_click_prob_map(fp_whole_map, fn_whole_map, meta, case_root, fp_points.shape[1])

def save_multiple_clicks_prob_maps(val_loader_init,clicks_info_val,sigma,save_root):
    '''
    sometimes, multiple clicks are generated for one-time simulation
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
                # print(click_arr.shape)
                # pdb.set_trace()
                # print(click_arr)
                fp_arr = click_arr[:,::2]
                fn_arr = click_arr[:,1::2]
                # print(fp_arr)
                # print(fn_arr)
                # print(fp_arr.shape)
                # pdb.set_trace()
                fp_map = get_prob_map(fp_arr, shape, sigma)   
                fn_map = get_prob_map(fn_arr, shape, sigma)
                save_click_prob_map(fp_map, fn_map, meta, case_root, i)
        
            # save all fore-ground and back-ground clicks in a seperate channel
            clicks_arr = np.concatenate(case_arr,axis=1)
            fp_points = clicks_arr[:,::2]
            fn_points = clicks_arr[:,1::2]
            fp_whole_map = get_prob_map(fp_points, shape, sigma)   
            fn_whole_map = get_prob_map(fn_points, shape, sigma)
            # print(fp_points.shape)
            # pdb.set_trace()
            save_click_prob_map(fp_whole_map, fn_whole_map, meta, case_root, fp_points.shape[1])

###################################################################

def sim_clicks_batch_pred(sim_batch, clicks_info, final_clicks_info, model, num_points, sigma, device): 
    key_list = sim_batch['kidney_id']     
    for k in range(num_points):
        clicks_list = sim_clicks(sim_batch, model, clicks_info, k, sigma, device)
        # print(k)
        # print(clicks_list)
        fill_the_dict(clicks_info, clicks_list, key_list)
    
    # print(clicks_info)
    # pdb.set_trace()
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
###########################################################################################################

def save_map_to_nii(fp_img, fn_img, mask_meta, mask_gt_shape):
    '''
    save fp_map and fn_map
    '''
    fp_img.SetSpacing([x.numpy()[0] for x in mask_meta['spacing']])
    fp_img.SetOrigin([x.numpy()[0] for x in mask_meta['origin']])
    fp_img.SetDirection([x.numpy()[0] for x in mask_meta['direction']])
    fn_img.SetSpacing([x.numpy()[0] for x in mask_meta['spacing']])
    fn_img.SetOrigin([x.numpy()[0] for x in mask_meta['origin']])
    fn_img.SetDirection([x.numpy()[0] for x in mask_meta['direction']])
    # save mdical image
    sitk.WriteImage(fn_img, 'fn_map.nii.gz')
    sitk.WriteImage(fp_img, 'fp_map.nii.gz')

def save_chamfer_map_to_nii(fp_img, fn_img, mask_meta, mask_gt_shape):
    '''
    save chamfer map
    '''
    fp_img.SetSpacing([x.numpy()[0] for x in mask_meta['spacing']])
    fp_img.SetOrigin([x.numpy()[0] for x in mask_meta['origin']])
    fp_img.SetDirection([x.numpy()[0] for x in mask_meta['direction']])
    fn_img.SetSpacing([x.numpy()[0] for x in mask_meta['spacing']])
    fn_img.SetOrigin([x.numpy()[0] for x in mask_meta['origin']])
    fn_img.SetDirection([x.numpy()[0] for x in mask_meta['direction']])
    # save mdical image
    sitk.WriteImage(fn_img, 'fn_chamfer_map.nii.gz')
    sitk.WriteImage(fp_img, 'fp_chamfer_map.nii.gz')

def save_fp_fn_map_to_nii(fp_img, fn_img, mask_meta, mask_gt_shape, i):
    '''
    save fp_map and fn_map
    '''
    fp_img.SetSpacing([x.numpy()[0] for x in mask_meta['spacing']])
    fp_img.SetOrigin([x.numpy()[0] for x in mask_meta['origin']])
    fp_img.SetDirection([x.numpy()[0] for x in mask_meta['direction']])
    fn_img.SetSpacing([x.numpy()[0] for x in mask_meta['spacing']])
    fn_img.SetOrigin([x.numpy()[0] for x in mask_meta['origin']])
    fn_img.SetDirection([x.numpy()[0] for x in mask_meta['direction']])
    # save mdical image
    if i == 100:
        sitk.WriteImage(fn_img, 'fn_0_map.nii.gz')
        sitk.WriteImage(fp_img, 'fp_0_map.nii.gz')
    else:
        fp_file_name = 'fp_' + str(i+1) + '_map.nii.gz'
        fn_file_name = 'fn_' + str(i+1) + '_map.nii.gz'
        sitk.WriteImage(fn_img, fn_file_name)
        sitk.WriteImage(fp_img, fp_file_name)

def save_prob_to_nii(mask, mask_meta, mask_gt_shape, i):
    '''
    save fp_map and fn_map
    '''
    mask = mask.view(mask_gt_shape)
    # numpy array to medical image
    mask_arr = mask.cpu().detach().numpy() 
    mask_img = sitk.GetImageFromArray(mask_arr)
    # set origin, direction, spacing
    mask_img.SetSpacing([x.numpy()[0] for x in mask_meta['spacing']])
    mask_img.SetOrigin([x.numpy()[0] for x in mask_meta['origin']])
    mask_img.SetDirection([x.numpy()[0] for x in mask_meta['direction']])
    # save mdical image
    if i == 100:
        sitk.WriteImage(mask_img, 'click_1_map.nii.gz')
    else:
        sitk.WriteImage(mask_img, 'click_' + str(i+1) + '_map.nii.gz')

def save_mask_to_nii(mask, mask_meta, mask_gt_shape,i):
    '''
    save fp_map and fn_map
    '''
    # mask size
    mask = mask.view(mask_gt_shape)
    # numpy array to medical image
    mask_arr = mask.cpu().detach().numpy() 
    mask_img = sitk.GetImageFromArray(mask_arr)
    # set origin, direction, spacing
    mask_img.SetSpacing([x.numpy()[0] for x in mask_meta['spacing']])
    mask_img.SetOrigin([x.numpy()[0] for x in mask_meta['origin']])
    mask_img.SetDirection([x.numpy()[0] for x in mask_meta['direction']])
    # save mdical image
    sitk.WriteImage(mask_img, 'predmask_' + str(i+1) + '_.nii.gz')