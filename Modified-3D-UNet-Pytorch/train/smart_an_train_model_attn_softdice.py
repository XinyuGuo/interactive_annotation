from kidney_dataloader_1_c_aug_2_classes import KidneyDataset, KidneyDataset_standardization, KidneyDataset_standardization_2
from kidney_loss_factory import SoftDiceLoss_MultiClasses, CrossEntropyLoss, Generalised_SoftDiceLoss_MultiClasses
import evaluate_tools
from weights_initialization import init_weights
# ******************
import os
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from model_attn_2 import Modified3DUNet_ATTN
import paths
import time
import pdb
# ********************

csv_train_path = '../train_multiC_aug.csv'
csv_val_path ='../val_weighted_dice_2.csv'
# csv_train_path = '../train_multiC_aug_test.csv'
# csv_val_path = '../val_multiC_aug_test.csv'

kidney_dataset_train = KidneyDataset_standardization_2(csv_train_path)
kidney_dataset_val = KidneyDataset_standardization_2(csv_val_path)

train_loader  = DataLoader(kidney_dataset_train, batch_size = 6, shuffle=True, num_workers=4)
val_loader = DataLoader(kidney_dataset_val, batch_size = 6, shuffle = True, num_workers = 2)

def datestr():
	now = time.localtime()
	return '{:04}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)

curtime = datestr()

# logging train and val loss
log_save_dir = '/data/ccusr/xinyug/annotation/Modified-3D-UNet-Pytorch/results/unet_attn_generalised_softdice'
if not os.path.exists(log_save_dir):
    os.mkdir(log_save_dir)

train_loss_file_name = 'train_loss.csv'
val_loss_file_name = 'val_loss.csv'
train_csv_writer = evaluate_tools.get_csv_writer(log_save_dir,train_loss_file_name)
val_csv_writer = evaluate_tools.get_csv_writer(log_save_dir,val_loss_file_name)
# 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Loading the model
in_channels = 1
n_classes = 3
base_n_filter = 16

# train a new model
model = Modified3DUNet_ATTN(in_channels, n_classes, base_n_filter).to(device)
init_weights(model,'xavier_normal')
# criterion = SoftDiceLoss_MultiClasses(0.000001)
criterion = Generalised_SoftDiceLoss_MultiClasses(0.000001)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)
check_points_save = '/data/ccusr/xinyug/annotation/Modified-3D-UNet-Pytorch/checkpoints/unet_attn_generalised_softdice'
if not os.path.exists(check_points_save):
    os.mkdir(check_points_save)

# training 
running_loss = 0
val_loss_best = 65535
model.train()
for i in range(100000):
    print('**** epoch ',i,' ****')
    running_loss = 0
    for train_batch in tqdm(train_loader):
        train_data = train_batch['data'].to(device) 
        train_data = torch.unsqueeze(train_data,1)
        train_label = train_batch['mask'].to(device)
        optimizer.zero_grad()
        output = model(train_data)
        output = output[1]
        # print(output.shape)
        # print(train_label.shape)
        # pdb.set_trace()
        loss = criterion(output, train_label)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
        # pdb.set_trace()
    print('training loss : %.3f'%(running_loss/len(train_loader)))
    evaluate_tools.write_row_csv(train_csv_writer,running_loss/len(train_loader))

    if i % 1 == 0 : # generate val loss and save model 
        model.eval() # model in evaluation mode
        run_val_loss = 0
        with torch.no_grad():
            for val_batch in tqdm(val_loader):               
                val_data = val_batch['data'].to(device) 
                val_data = torch.unsqueeze(val_data,1)
                val_label = val_batch['mask'].to(device)
                # val_label = val_batch['mask'].view(1,-1).squeeze(0).to(device)
                output_val = model(val_data)
                logits_val = output_val[1]
                val_loss = criterion(logits_val, val_label)
                run_val_loss+=val_loss.item()
            val_loss_final = run_val_loss/len(val_loader)    
            print('val loss : %.3f'%(val_loss_final))
            evaluate_tools.write_row_csv(val_csv_writer,val_loss_final)
            # save model
            if val_loss_final < val_loss_best:
                val_loss_best = val_loss_final
                print('**** best val loss is : %.3f ******'%(val_loss_best))
                evaluate_tools.save_checkpoint_with_date(model, loss, i, curtime, check_points_save)
                print ('********** model saved !************')
        model.train()