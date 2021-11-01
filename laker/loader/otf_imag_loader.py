"""
This module implements on-the-fly data augmentation loader

"""
from random import randint
import queue
from threading import Thread
import numpy as np

import torch
import torch.nn as nn
import cv2
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

def get_img_info(data_dir):        
    data_info = list()        # data_dir 是训练集、验证集或者测试集的路径        
    for root, dirs, _ in os.walk(data_dir):  
        for sub_dir in dirs:                # 文件列表                
            img_names = os.listdir(os.path.join(root, sub_dir))      
            # 取出 jpg 结尾的文件
            img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))                       # 遍历图片                
            for i in range(len(img_names)):
                img_name = img_names[i]                    # 图片的绝对路径 
                path_img = os.path.join(root, sub_dir, img_name)
                data_info.append(path_img)
    return data_info

class AEImagSet(torch.utils.data.Dataset):

    def __init__(self, data_dir, max_len, seq_len = 150, transform=None):
        super(AEImagSet, self).__init__()
        self.data_info = get_img_info(data_dir)
        if transform is not None:
            self.transform = transform
        else:
            #self.transform = transforms.Compose([
            #                       transforms.Resize((640, 480)), 
            #                       transforms.ToTensor()
            #                                    ])
            #self.transform = nn.Sequential(
            #                    transforms.ConvertImageDtype(torch.float)
            #                 )
            pass
        self.max_len = max_len
        self.seq_len = seq_len

    def __getitem__(self, index):        # 通过 index 读取样本        
        path_imgs = self.data_info[index * self.seq_len : (index + 1) * self.seq_len]
        Imgs = list()
        for idx in range(self.seq_len):
            #img = Image.open(path_imgs[idx]).convert('RGB')     # 0~255        
            img = cv2.imread(path_imgs[idx])
            img = cv2.resize(img, (480, 640)).transpose(2, 0, 1) # H: 640, W:480
            #if self.transform is not None:
            #   img = self.transform(img)   # 在这里做transform，转为tensor等等
            Imgs.append(img)
        return Imgs

    def __len__(self):
        return len(self.data_info) // self.seq_len 


def dataloader(data_dir, args):
    imagSet = AEImagSet(
                     data_dir= data_dir,\
                     max_len = args.seq_len,\
                     seq_len = args.seq_len 
                   )
    dataloader_ = DataLoader(imagSet, batch_size=args.batch_size, shuffle=True)
    for i, dataBatch in enumerate(dataloader_):
        ## dataBatch: seq_len, B, C, H, W
        datalst = []
        for data in dataBatch[0:]:
           datalst.append(data.unsqueeze(1))
        dataBatch_ = tuple(datalst) 
        dataBatch_ = torch.cat(dataBatch_, dim=1)
        ## bsz, seq_len, C, H, W
        yield dataBatch_

def register(parser):
    """
    register loader arguments
    """
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Traing batch size')
    parser.add_argument('--seq_len', type=int, default=100,
                        help='the sequence of training images')
