from batchgenerators.dataloading.data_loader import DataLoader
import pandas as pd
import numpy as np
import SimpleITK as sitk
import pdb

class KidneyDatasetAug(DataLoader):
    '''
    kidney dataset for smart annotation
    '''
    def __init__(self, data, batch_size, num_threads_in_multithreaded=4, seed_for_shuffle=1234, return_incomplete=False,\
                 shuffle=True, infinite=True):
        super().__init__(data, batch_size, num_threads_in_multithreaded, seed_for_shuffle, return_incomplete, shuffle,infinite)
        self.indices = list(range(len(data)))
    def generate_train_batch(self):
        idx = self.get_indices()
        patients_for_batch = [self._data.iloc[i] for i in idx]
        img_data = []
        mask_data = []
        case_id = []
        case_meta = []
        
        for row in patients_for_batch:
            k_r_path = row['kidney_resampled_path']
            m_r_path = row['mask_resampled_path']
            img = sitk.ReadImage(k_r_path)
            mask = sitk.ReadImage(m_r_path) 
            img_arr = sitk.GetArrayFromImage(img)
            mask_arr = sitk.GetArrayFromImage(mask)
            # mask_arr[mask_arr==2]=0
            img_data.append(img_arr.astype(np.float32))
            mask_data.append(mask_arr.astype(np.float32))
            # case_id.append(row['kidney_path'].split('/')[7])
            case_id.append(row['case_img_id'])
            cur_meta = {'origin': mask.GetOrigin(), 'direction': mask.GetDirection(), 'spacing': mask.GetSpacing()}
            case_meta.append(cur_meta)
           
        data_batch = np.stack(img_data)
        mask_batch = np.stack(mask_data)
        data_batch = np.expand_dims(data_batch,1)
        mask_batch = np.expand_dims(mask_batch,1)
        batch_sample = {'data': data_batch, 'seg': mask_batch, 'case_id': case_id, 'case_meta': case_meta}
        return batch_sample