import torch 

index = torch.tensor([[0,1,2],[0,0,1],[0,1,2],[1,0,2]])
# print(index.shape)
onehot = torch.zeros((4,3,3))
# print(index.unsqueeze(1).shape)
print (onehot.scatter_(1,index.unsqueeze(1),1))