import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np
import pandas as pd
import csv
import pdb

def save_checkpoint(model, epoch_num, save_dir):
    model_dict = {
        'state_dict':model.state_dict()
    }
    filename = 'checkpoint_'+str(epoch_num) + '_'+'.pth.tar'
    modelpath = os.path.join(save_dir,filename)
    torch.save(model_dict,modelpath)

def save_checkpoint_with_date(model, loss, epoch_num, date, save_dir):
    '''
    save model
    '''
    model_dict = {
        'loss' : loss,
        'state_dict':model.state_dict()
    }
    filename = 'checkpoint_'+str(epoch_num) + '_' + date + '.pth.tar'
    modelpath = os.path.join(save_dir,filename)
    torch.save(model_dict,modelpath)


def load_checkpoint(model, modelpath):
    '''
    load the model checkpoint
    '''
    model_params = torch.load(modelpath)
  
    # load parameters to the model.
    model.load_state_dict(model_params['state_dict'])
    model.eval()
    return model

def load_checkpoint_with_date(model, modelpath):
    '''
    load the model checkpoint
    '''
    model_params = torch.load(modelpath)
  
    # load parameters to the model.
    model.load_state_dict(model_params['state_dict'])
    model.eval()
    return model

def load_checkpoint_1c_2_3c(model, modelpath, device):
    '''
    load the model checkpoint
    '''
    model_info= torch.load(modelpath)
    state_dict = model_info['state_dict']
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key == 'conv3d_c1_1.weight':
            # extra_1_c = torch.tensor(np.zeros(value.shape).astype(np.float32)).to(device)
            # extra_2_c = torch.tensor(np.zeros(value.shape).astype(np.float32)).to(device)
            new_value = torch.cat((value,value,value),1)
            new_state_dict[key] = new_value
            continue   
        new_state_dict[key] = value
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

def save_loss_csv(train_loss_l, val_loss_l, save_dir):
    '''
    save train and val list to csv files
    '''
    train_dict = {'train_loss': train_loss_l}
    val_dict = {'val_loss': val_loss_l}

    train_loss_path = os.path.join(save_dir, 'train_loss.csv')
    val_loss_path = os.path.join(save_dir, 'val_loss.csv')
    
    df_train = pd.DataFrame(train_dict)
    df_val = pd.DataFrame(val_dict)
    
    df_train.to_csv(train_loss_path)
    df_val.to_csv(val_loss_path)

def plot_loss(val_csv, train_csv):
    '''
    plot train / validation curve
    '''
    val_df = pd.read_csv(val_csv)
    train_df = pd.read_csv(train_csv)
    train_epochs =list(range(len(train_df)))
    val_epochs = list(range(len(val_df)))
    # train_axis = list(filter(lambda x:x%5 == 0,train_epochs))
    # val_axis = list(filter(lambda x:x%5 ==0,val_epochs))
    train_loss = train_df['train_loss'].tolist()
    val_loss = val_df['val_loss'].tolist()
    plt.title('Modified 3D UNet Loss Curve')

    plt.plot(train_epochs, train_loss, color='green', label='training loss')
    plt.plot(val_epochs, val_loss, color='red', label='val loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    loss_fig_path = os.path.join(os.path.split(val_csv)[0], 'loss.png')
    plt.savefig(loss_fig_path)

def getIOU_two_classes(pred,gt):
    '''
    get IOU between predicted labels and ground truth
    '''
    tp_num = torch.sum((pred==1)&(gt==1)).float()
    pre_num = torch.sum(pred==1)
    gt_num = torch.sum(gt==1)

    # sample_num = torch.tensor(pred.shape[0])
    iou = tp_num / (pre_num + gt_num - tp_num)
    return iou

def write_row_csv(csv_writer,value):
    csv_writer.writerow([value])

def get_csv_writer(root,name):
    csv_path = os.path.join(root,name)
    csv_file = open(csv_path, 'w')
    csv_writer = csv.writer(csv_file)
    if 'train' in name:
        csv_writer.writerow(['train_loss'])
    if 'val' in name:
        csv_writer.writerow(['val_loss'])
    return csv_writer