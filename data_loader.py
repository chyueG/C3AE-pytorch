from __future__ import print_function,division
import os
import torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader
import cv2
import numpy as np
import random

class FaceDataset(Dataset):
    def __init__(self,im_list,root_dir,transform=None):
        with open(im_list,"r") as fd:
            self.im_info = fd.readlines()
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.im_info)



    def __getitem__(self,idx):
        im_info = self.im_info[idx]

        info    = im_info.strip().split(" ")
        im_path = info[0]
        if os.path.exists(os.path.join(self.root_dir,"0",im_path)) and os.path.exists(os.path.join(self.root_dir,"1",im_path)) and os.path.exists(os.path.join(self.root_dir,"2",im_path)):
            pass
        else:
            idx = random.randint(0,5000)
            im_info = self.im_info[idx]
            info = im_info.strip().split(" ")
            im_path = info[0]


        age     = int(info[1])



        if age>100:
            age = 30
        #gender  = float(info[2])
        #landmark_x  = list(map(lambda x:int(info[x]),[3,5,7,9,11]))
        #landmark_y  = list(map(lambda x:int(info[x]),[4,6,8,10,12]))

        #im_path,label,_ = im_info.strip().split()
        ims     = []

        for i in ["0","1","2"]:
            im      = self.read_im(os.path.join(self.root_dir,i,im_path))
            ims.append(im)
        label   = age
        label_vec = label_to_vector(label)
        sample  = {"image":ims,"label":label,"label_vec":label_vec}

        if self.transform:
            sample = self.transform(sample)

        return sample
    def read_im(self,path):
        image = cv2.imread(path)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        return image

    """
    def landmark_crop(self,landmarks):
        with open(landmarks,"r") as f:
            for line in f.readlines():
                info = line.strip().split(" ")
                im_name = info[0]
                x_axis = list(map(lambda x:info[x],[1,3,5,7,9]))
                xmin   = min(x_axis)
                xmax   = max(x_axis)
                y_axis = list(map(lambda x:info[x],[2,4,6,8,10]))
                ymin   = min(y_axis)
                ymax   = max(y_axis)
    """
class Resize(object):
    def __init__(self,new_size):
        self.new_size = new_size

    def __call__(self,sample):
        ims = sample["image"]
        label = sample["label"]
        label_vec = sample["label_vec"]
        im = list(map(lambda x :cv2.resize(x,self.new_size),ims))
        return {"image":im,"label":label,"label_vec":label_vec}



class ToTensor(object):
    def __call__(self,sample):
        ims,label,label_vec = sample["image"],sample["label"],sample["label_vec"]
        ims = list(map(lambda x:x.transpose((2,0,1)),ims))
        ims       = torch.cat(list(map(lambda x:torch.from_numpy(x),ims)))
        label    = torch.tensor(label,dtype=torch.float32)
        ims      = ims.type_as(label)
        label_vec = torch.tensor(label_vec,dtype=torch.float32)
        return {"image":ims,"label":label,"label_vec":label_vec}

def label_to_vector(label,interval=10):
    #:print("age label is {}".format(label))
    z1      = interval * np.floor(label/interval)
    z2      = interval * np.ceil(label/interval)
    lambda1 = 1 - (label - z1)/interval
    lambda2 = 1 - (z2 - label)/interval
    age_vec = [0]*12
    age_vec[int(z1//interval)] = lambda1
    age_vec[int(z2//interval)] = lambda2
    return age_vec

