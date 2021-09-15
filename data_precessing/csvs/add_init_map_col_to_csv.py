import pandas as pd
import os
import pdb

def add_colums(ori_csv_path, filepath):
    '''
    add colums to csv
    '''
    csv_df = pd.read_csv(ori_csv_path)
    data_root = '../kidney/initial_simulations'
    map_paths = []
    for _, row in csv_df.iterrows():
        kidney_path = row['kidney_path']
        case_name = kidney_path.split('/')[7]
        case_id = ['1' if '1' in os.path.split(kidney_path)[-1] else '0'][0]
        case_id = case_name + '_' + case_id
        case_weighted_map_name = case_id + '_init_map.npy'
        case_weighted_map_path = os.path.join(data_root, case_weighted_map_name)
        map_paths.append(case_weighted_map_path)
        print(case_id)
    csv_df['init_weighted_path'] = map_paths
    csv_df.to_csv(filepath,index=False)

ori_csv_train_path = 'train_0122_20.csv'
ori_csv_val_path = 'val_0122_20.csv'
new_file_path_train = 'train_weighted_dice.csv'
new_file_path_val = 'val_weighted_dice.csv'

add_colums(ori_csv_train_path,new_file_path_train)
add_colums(ori_csv_val_path, new_file_path_val)