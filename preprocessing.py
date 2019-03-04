from PIL import Image
from torch.utils.data import Dataset,
import torch as th
import numpy as np
from torchvision import datasets,transforms

class Pairwise(Dataset):
    def __init__(self, X, U):
        self.X = X
        self.U = U

    def __len__(self):
        return len(self.U)

    def __getitem__(self, i):
        x_i = self.X[i]
        u_i = self.U[i]

        idx = list(range(len(self.U)))
        idx.remove(i)
        j = np.random.choice(idx)

        x_j = self.X[j]
        u_j = self.U[j]

        return x_i, x_j, u_i, u_j
      
data_transforms = {

     "train" : transforms.Compose([
                transforms.Resize((226,226)),
                transforms.RandomCrop(224), 
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
      "test" : transforms.Compose([
                transforms.Resize((226,226)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])
  }
def load_data(imageset,state):
  all_img = []
  label = []
  for i in range(len(imageset)):
    img_path,y = imageset.imgs[i]
    img = Image.open(img_path).convert("RGB")
    img = data_transforms[state](img)
    all_img.append(img)
    label.append(y)
  return [th.stack(all_img), th.FloatTensor(label)]


def creat_DataLoader():
  PATH="cat/"
  state = ["train","test"]
  image_data = {x : datasets.ImageFolder(PATH+x, data_transforms[x]) for x in state}
  #tr_x, tr_y = load_data(image_data["train"], data_transforms["train"])
  #te_x,te_y = load_data(image_data["test"], data_transforms["test"])
  d = {x : load_data(image_data[x],x ) for x in state}
  #train_data = Pairwise(tr_x, tr_y) 
  #vaild_data = Pairwise(te_x, te_y)
  train_test_data = {x : Pairwise(d[x][0] ,d[x][1]) for x in state}
  sf = {"train":True,"test":True}
  data_loader = {x:DataLoader(train_test_data[x],batch_size = 10,shuffle =sf[x],num_workers=4) for x in state}
  return data_loader

