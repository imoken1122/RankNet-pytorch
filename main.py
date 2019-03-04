from PIL import Image
from preprocessing import creat_DataLoader
from loss_function import CrossEntropyLoss
import torch as th
from torch import nn,optim
from torch.utils.data import (DataLoader,TensorDataset,Dataset)
import torchvision as tv
from torchvision import datasets,models,transforms
import torchvision.transforms as tf
from torch.autograd import Variable as V
import numpy as np
import matplotlib.pyplot as plt



def train(model,data_loader,gpu=False):
  model.train()
  run_loss = 0.0
  for d in data_loader:
    opt.zero_grad()
    x_i, x_j,u_i, u_j = map(V, d)
    if gpu:
      x_i, x_j,u_i, u_j = x_i.cuda(), x_j.cuda(),u_i.cuda(), u_j.cuda()
    
    s_i = model(x_i)
    s_j = model(x_j)

    loss = criterion(s_i,s_j ,u_i,u_j)
    run_loss += loss.data
    loss.backward()
    opt.step()
  train_loss = run_loss/len(data_loader)
  return train_loss
   
    
def vaild(model,data_loader,gpu=False):
  model.eval()
  run_loss = 0.0
  with th.no_grad():
    for d in data_loader:
      opt.zero_grad()
      x_i, x_j,u_i, u_j = map(V, d)
      
      if gpu:
        x_i, x_j,u_i, u_j = x_i.cuda(), x_j.cuda(),u_i.cuda(), u_j.cuda()

      s_i = model(x_i)
      s_j = model(x_j)
      loss = criterion(s_i,s_j ,u_i,u_j)
      run_loss += loss.data
      
  eval_loss = run_loss/len(data_loader)
  return eval_loss

def main():
  data_loader,image_data = creat_DataLoader()

  model = models.resnet18(pretrained=True)
  for p in model.parameters():
    p.requires_grad = False 

  model.fc = nn.Linear(model.fc.in_features,1)
  model = model.cuda()
  criterion = CrossEntropyLoss()
  opt = optim.Adam(model.fc.parameters())
  max_epoch = 10
  tr_loss,te_loss = [],[]
  print("training start")

  for epoch in range(max_epoch):
    train_loss = train(model,data_loader["train"],True)
    eval_loss = vaild(model,data_loader["test"],True)

    tr_loss.append(train_loss)
    te_loss.append(eval_loss)

    if epoch % 2 == 0:
      print(f"epoch:{epoch} , loss:{train_loss} , val_loss:{eval_loss}")
      
  print("trainning complete!! ")
if __name__ == "__main__" :
  main()
