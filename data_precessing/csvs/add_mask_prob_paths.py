import pandas as pd
import pdb
import os

def add_new_clos(train_df, val_df):
    '''
    add new columns to csv files 
    '''
    train_pred_mask_paths = []
    train_pred_prob_paths = []
    val_pred_mask_paths = []
    val_pred_prob_paths = []

    pred_mask_name_0 = 'pred_mask_0.nii.gz'
    pred_mask_name_1 = 'pred_mask_1.nii.gz'
    pred_prob_name_0 = 'pred_prob_0.nii.gz'
    pred_prob_name_1 = 'pred_prob_1.nii.gz'

    for _, row in train_df.iterrows():
        prob_path =row['prob_resampled_path']
        prob_name = os.path.split(prob_path)[-1]
        prob_root = os.path.split(prob_path)[0]
        if '1' in prob_name:
            train_pred_mask_paths.append(os.path.join(prob_root, pred_mask_name_1))
            train_pred_prob_paths.append(os.path.join(prob_root, pred_prob_name_1))
            # print(os.path.join(prob_root, prob_1_10_name))
        if '0' in prob_name:
            train_pred_mask_paths.append(os.path.join(prob_root, pred_mask_name_0))
            train_pred_prob_paths.append(os.path.join(prob_root, pred_prob_name_0))
            # print(os.path.join(prob_root, prob_0_10_name))
        # pdb.set_trace()
    train_df['pred_mask_paths'] = train_pred_mask_paths
    train_df['pred_prob_paths'] = train_pred_prob_paths

    for _, row in val_df.iterrows():
        val_prob_path =row['prob_resampled_path']
        val_prob_name = os.path.split(val_prob_path)[-1]
        val_prob_root = os.path.split(val_prob_path)[0]
        if '1' in val_prob_name:
            val_pred_mask_paths.append(os.path.join(val_prob_root, pred_mask_name_1))
            val_pred_prob_paths.append(os.path.join(val_prob_root, pred_prob_name_1))
            # print(os.path.join(prob_root, prob_1_10_name))
        if '0' in val_prob_name:
            val_pred_mask_paths.append(os.path.join(val_prob_root, pred_mask_name_0))
            val_pred_prob_paths.append(os.path.join(val_prob_root, pred_prob_name_0))
            # print(os.path.join(prob_root, prob_0_10_name))
        # pdb.set_trace()
    val_df['pred_mask_paths'] = val_pred_mask_paths
    val_df['pred_prob_paths'] = val_pred_prob_paths

    train_df.to_csv('train_0122_20.csv',index=False)    
    val_df.to_csv('val_0122_20.csv',index=False)   
     
# script
train_csv_path = '/data/ccusr/xinyug/annotation/train_new.csv'
val_csv_path = '/data/ccusr/xinyug/annotation/val_new.csv'

train_df = pd.read_csv(train_csv_path)
val_df = pd.read_csv(val_csv_path)
add_new_clos(train_df, val_df)