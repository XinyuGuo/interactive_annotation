import pandas as pd
import pdb
import os

def add_new_clos(train_df, val_df):
    '''
    add new columns to csv files 
    '''
    train_10_paths = []
    val_10_paths = []
    prob_0_10_name = 'prob_map_0_resampled_10.nii.gz'
    prob_1_10_name = 'prob_map_1_resampled_10.nii.gz'

    for _, row in train_df.iterrows():
        prob_path =row['prob_resampled_path']
        prob_name = os.path.split(prob_path)[-1]
        prob_root = os.path.split(prob_path)[0]
        if '1' in prob_name:
            train_10_paths.append(os.path.join(prob_root, prob_1_10_name))
            # print(os.path.join(prob_root, prob_1_10_name))
        if '0' in prob_name:
            train_10_paths.append(os.path.join(prob_root, prob_0_10_name))
            # print(os.path.join(prob_root, prob_0_10_name))
        # pdb.set_trace()
    train_df['prob_resmapled_10'] = train_10_paths

    for _, row in val_df.iterrows():
        val_prob_path =row['prob_resampled_path']
        val_prob_name = os.path.split(val_prob_path)[-1]
        val_prob_root = os.path.split(val_prob_path)[0]
        if '1' in val_prob_name:
            val_10_paths.append(os.path.join(val_prob_root, prob_1_10_name))
            # print(os.path.join(prob_root, prob_1_10_name))
        if '0' in val_prob_name:
            val_10_paths.append(os.path.join(val_prob_root, prob_0_10_name))
            # print(os.path.join(prob_root, prob_0_10_name))
        # pdb.set_trace()
    val_df['prob_resmapled_10'] = val_10_paths

    train_df.to_csv('train_sigma_10.csv',index=False)    
    val_df.to_csv('val_sigma_10.csv',index=False)   
     
# script
train_csv_path = '/data/ccusr/xinyug/annotation/train.csv'
val_csv_path = '/data/ccusr/xinyug/annotation/val.csv'

train_df = pd.read_csv(train_csv_path)
val_df = pd.read_csv(val_csv_path)
add_new_clos(train_df, val_df)