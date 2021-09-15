import pandas as pd
import os
import pdb

def add_case_id(csv_df):
    '''
    add case id
    '''
    case_img_ids = []
    for _, row in csv_df.iterrows():
        kidney_path = row['kidney_path']
        case_id = kidney_path.split('/')[7]
        img_id = kidney_path.split('/')[8]
        img_id = img_id.split('.')[0].split('_')[-1]
        case_img_id = case_id + '_' + img_id
        case_img_ids.append(case_img_id)
    csv_df['case_img_id'] = case_img_ids
    # print(csv_df)
    # pdb.set_trace()
    return csv_df

csv_path_train = 'train.csv'
csv_path_val = 'val.csv'

csv_path_train_df = pd.read_csv(csv_path_train)
csv_path_val_df = pd.read_csv(csv_path_val)

train_df = add_case_id(csv_path_train_df)
val_df = add_case_id(csv_path_val_df)

train_df.to_csv('train_new.csv', index=False)
val_df.to_csv('val_new.csv', index=False)