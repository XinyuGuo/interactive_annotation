import torch.nn as nn
import torch
import numpy as np
import pdb
import torch.nn.functional as F

class CrossEntropyLoss(nn.Module):
    '''
    cross-entropy loss
    '''
    def __init__(self, weight):
        super(CrossEntropyLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=weight)
    def forward(self, pred, target):
        ce_loss=self.ce_loss(pred,target)
        return ce_loss

class SoftDiceLoss(nn.Module):
    '''
    pred : 1D tensor FloatTensor containing logits
    target : 1D FloatTensor containing data labels
    '''
    def __init__(self, smooth):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
        # self.sig = torch.nn.Sigmoid()
    def forward(self, pred, target):
        pred_label = torch.sigmoid(pred)
        # pred_label = self.sig(pred)
        tp_tensor = pred_label.mul(target)
        numerator = 2*tp_tensor.sum() + self.smooth
        denominator = pred_label.sum() + target.sum() + self.smooth
        coefficient = numerator / denominator
        return 1-coefficient

class SoftDiceLoss_MultiClasses(nn.Module):
    '''
    dice loss for the multiclass segmentation task
    '''
    def __init__(self, smooth):
        super(SoftDiceLoss_MultiClasses, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = F.softmax(pred,dim=1)
        # construct pred onehot 
        _, max_index = torch.max(pred,1)
        pred_labels = pred.data.clone().zero_()
        pred_labels.scatter_(1, max_index.unsqueeze(1),1)
        # construct ground truth one-hot
        onehot_target = pred.data.clone().zero_()
        onehot_target.scatter_(1, target.unsqueeze(1), 1)
        # calculate the multi-class dice loss
        num =  (pred*onehot_target).sum(dim=4).sum(dim=3).sum(dim=2)
        den1 = (pred*pred_labels).sum(dim=4).sum(dim=3).sum(dim=2)
        den2 = onehot_target.sum(dim=4).sum(dim=3).sum(dim=2)
        dice = (2*num)/(den1+den2+self.smooth)
        dice = dice.sum()/((dice.size(0) * dice.size(1)))
        return 1-dice

class SoftDiceLoss_MultiClasses_Plus_CrossEntropy(nn.Module):
    def __init__(self, weight,smooth):
        super(SoftDiceLoss_MultiClasses_Plus_CrossEntropy, self).__init__()
        self.smooth = smooth
        self.ce_loss = nn.CrossEntropyLoss(weight=weight)
    def forward(self, pred, target):
        pred = F.softmax(pred,dim=1)
        # construct pred onehot 
        _, max_index = torch.max(pred,1)
        pred_labels = pred.data.clone().zero_()
        pred_labels.scatter_(1, max_index.unsqueeze(1),1)
        # construct ground truth one-hot
        onehot_target = pred.data.clone().zero_()
        onehot_target.scatter_(1, target.unsqueeze(1), 1)
        # calculate the multi-class dice loss
        num =  (pred*onehot_target).sum(dim=4).sum(dim=3).sum(dim=2)
        den1 = (pred*pred_labels).sum(dim=4).sum(dim=3).sum(dim=2)
        den2 = onehot_target.sum(dim=4).sum(dim=3).sum(dim=2)
        dice = (2*num)/(den1+den2+self.smooth)
        dice = dice.sum()/((dice.size(0) * dice.size(1)))
        loss  = ((1-dice) + self.ce_loss(pred,target))*0.5
        return loss

class Generalised_SoftDiceLoss_MultiClasses(nn.Module):
    def __init__(self, smooth):
        super(Generalised_SoftDiceLoss_MultiClasses, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = F.softmax(pred,dim=1)
        # construct pred onehot 
        _, max_index = torch.max(pred,1)
        pred_labels = pred.data.clone().zero_()
        pred_labels.scatter_(1, max_index.unsqueeze(1),1)
        # construct ground truth one-hot
        onehot_target = pred.data.clone().zero_()
        onehot_target.scatter_(1, target.unsqueeze(1), 1)
        # calculate the multi-class dice loss
        num =  (pred*onehot_target).sum(dim=4).sum(dim=3).sum(dim=2)
        den1 = (pred*pred_labels).sum(dim=4).sum(dim=3).sum(dim=2)
        den2 = onehot_target.sum(dim=4).sum(dim=3).sum(dim=2)
        class_weight  = den2**2 + self.smooth
        num = num/class_weight
        div = (den1+den2+self.smooth)/class_weight
        # dice = (2*num)/(den1+den2+self.smooth)
        dice = 2*num/div
        dice = dice.sum()/((dice.size(0) * dice.size(1)))
        return 1-dice
        
class BCEPlusSoftDice(nn.Module):
    def __init__(self, smooth, weight):
        super(BCEPlusSoftDice, self).__init__()
        self.smooth = smooth
        self.weight = weight
        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(pos_weight = weight)

    def forward(self, pred, target):
        # soft dice 
        pred_label = torch.sigmoid(pred)
        tp_tensor = pred_label.mul(target)
        numerator = 2*tp_tensor.sum() + self.smooth
        denominator = pred_label.sum() + target.sum() + self.smooth
        coefficient = numerator / denominator
        softdice = 1-coefficient
        bceloss = self.BCEWithLogitsLoss(pred, target)
        # print(bceloss)
        # print(softdice)
        return (bceloss + softdice)
        # bce

class SoftDicePlusIou(nn.Module): 
    '''
    pred : 1D tensor FloatTensor containing logits
    target : 1D FloatTensor containing data labels 
    '''
    def __init__(self, smooth):
        super(SoftDicePlusIou, self).__init__()
        self.smooth = smooth
        # self.sig = torch.nn.Sigmoid()
    def forward(self, pred, target):
        # soft dice 
        pred_label = torch.sigmoid(pred)
        tp_tensor = pred_label.mul(target)
        numerator = 2*tp_tensor.sum() + self.smooth
        denominator = pred_label.sum() + target.sum() + self.smooth
        coefficient = numerator / denominator
        softdice = 1-coefficient

        # iou
        pred[pred>0.5] = 1
        pred[pred<=0.5] = 0
        tp_num = torch.sum((pred==1)&(target==1)).float()
        pre_num = torch.sum(pred==1)
        gt_num = torch.sum(target==1)
        iou = tp_num / (pre_num + gt_num - tp_num)
        
        # print(iou)
        return softdice + (1-iou)

class WeightedDiceLoss(nn.Module):
    '''
    calculate the weighted dice loss based on the weighted map
    '''
    def __init__(self, bias, class_num, device):
        super(WeightedDiceLoss, self).__init__()
        self.bias = bias
        self.class_num = class_num
        self.device = device
    
    def forward(self, pred, target, weighted_target):
        '''
        weighted loss function
        '''
        unique = torch.unique(weighted_target).to(self.device)
        target = target.view(1,-1).squeeze(0)
        pred = pred.view(1,-1).squeeze(0)
        weighted_target = weighted_target.view(1,-1).squeeze(0)
        class_nums = torch.tensor(np.array(list(range(self.class_num))).astype(np.float32)).to(self.device)
        numerator = torch.tensor(0.0).to(self.device)
        denominator = torch.tensor(0.0).to(self.device)
        for i in class_nums:
            for j in unique:
                lw_number, label_weighted= self.get_label_w(i,j,target,weighted_target)
                numerator+=torch.sum(pred[label_weighted])/(lw_number**2)
                denominator_temp = torch.add(pred,label_weighted.type(torch.cuda.FloatTensor))
                denominator+= torch.sum(denominator_temp[label_weighted.type(torch.cuda.BoolTensor)])/(lw_number**2)
        weighted_dice_loss = 1 - (2*numerator/denominator)
        # pdb.set_trace()
        return weighted_dice_loss

    def get_label_w(self, label, weight, target, weighted_target):
        label_weighted = (target==label)&(weighted_target==weight)  
        return torch.sum(label_weighted).type(torch.cuda.FloatTensor), label_weighted.to(self.device)