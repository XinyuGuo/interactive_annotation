import pandas as pd 
import random

def data_split(kidney_csv_path):
    '''
    split data into train and val
    '''
    kidney_df = pd.read_csv(kidney_csv_path)
    df_len = len(kidney_df)
    indices = list(range(df_len))
    train_indices = random.sample(indices, 189)
    
    kidney_train = []
    mask_train = []
    probs_train = []
    
    kidney_train_r = []
    mask_train_r = []
    probs_train_r = []

    kidney_val = []
    mask_val = []
    probs_val = []

    kidney_val_r = []
    mask_val_r = []
    probs_val_r = []

    for index in range(210):
        row = kidney_df.iloc[index]
        if index in train_indices:
            if row['kidney_0'] != 'no':
                kidney_train.append(row['kidney_0'])
            if row['kidney_1'] != 'no':
                kidney_train.append(row['kidney_1'])     
            if row['mask_0'] != 'no':
                mask_train.append(row['mask_0'])
            if row['mask_1'] != 'no':
                mask_train.append(row['mask_1']) 
            if row['prob_0'] != 'no':
                probs_train.append(row['prob_0'])
            if row['prob_1'] != 'no':
                probs_train.append(row['prob_1'])  
            # 
            if row['kidney_0_resampled'] != 'no':
                kidney_train_r.append(row['kidney_0_resampled'])
            if row['kidney_1_resampled'] != 'no':
                kidney_train_r.append(row['kidney_1_resampled'])
            if row['mask_0_resampled'] != 'no':
                mask_train_r.append(row['mask_0_resampled'])
            if row['mask_1_resampled'] != 'no':
                mask_train_r.append(row['mask_1_resampled']) 
            if row['prob_resampled_0'] != 'no':
                probs_train_r.append(row['prob_resampled_0'])
            if row['prob_resampled_1'] != 'no':
                probs_train_r.append(row['prob_resampled_1'])
            
        else:
            if row['kidney_0'] != 'no':
                kidney_val.append(row['kidney_0'])
            if row['kidney_1'] != 'no':
                kidney_val.append(row['kidney_1'])     
            if row['mask_0'] != 'no':
                mask_val.append(row['mask_0'])
            if row['mask_1'] != 'no':
                mask_val.append(row['mask_1']) 
            if row['prob_0'] != 'no':
                probs_val.append(row['prob_0'])
            if row['prob_1'] != 'no':
                probs_val.append(row['prob_1']) 
            # 
            if row['kidney_0_resampled'] != 'no':
                kidney_val_r.append(row['kidney_0_resampled'])
            if row['kidney_1_resampled'] != 'no':
                kidney_val_r.append(row['kidney_1_resampled'])
            if row['mask_0_resampled'] != 'no':
                mask_val_r.append(row['mask_0_resampled'])
            if row['mask_1_resampled'] != 'no':
                mask_val_r.append(row['mask_1_resampled'])
            if row['prob_resampled_0'] != 'no':
                probs_val_r.append(row['prob_resampled_0'])
            if row['prob_resampled_1'] != 'no':
                probs_val_r.append(row['prob_resampled_1'])

    csv_dic_train = {'kidney_path': kidney_train, 'mask_path': mask_train, 'prob_path': probs_train, \
        'kidney_resampled_path': kidney_train_r, 'mask_resampled_path': mask_train_r, 'prob_resampled_path': probs_train_r}
    csv_dic_val = {'kidney_path': kidney_val, 'mask_path': mask_val, 'prob_path': probs_val, \
        'kidney_resampled_path': kidney_val_r, 'mask_resampled_path': mask_val_r, 'prob_resampled_path': probs_val_r} 
    train_df = pd.DataFrame(csv_dic_train)
    val_df = pd.DataFrame(csv_dic_val)
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    val_df = val_df.sample(frac=1).reset_index(drop=True)
    train_df.to_csv('train.csv', index=False)
    val_df.to_csv('val.csv',index=False)

kidney_csv_path = 'kidney_data.csv'
data_split(kidney_csv_path)