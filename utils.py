import os 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision as trv
from PIL import Image
import random

VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]

def label2image(pred):
    colormap = torch.tensor(VOC_COLORMAP) 
    X = pred.long() 
    return colormap[X, :].numpy()

def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1.
    else:
        center = factor - 0.5
    
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels, kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight

class VOC_dataset(torch.utils.data.Dataset):
    def __init__(self, VOC_DIR, train=True, crop_size=(320, 480)):
        self.voc_dir = VOC_DIR
        self.train = train
        self.crop_size = crop_size
        self.img_paths, self.label_paths = self.load_img_path()
        self.transform = trv.transforms.Compose([trv.transforms.ConvertImageDtype(torch.float32),
                         trv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
    def load_img_path(self):
        if self.train:
            txt_fname = os.path.join(self.voc_dir, "ImageSets", "Segmentation", "train.txt")
        else:
            txt_fname = os.path.join(self.voc_dir, "ImageSets", "Segmentation", "val.txt")
        with open(txt_fname, "r") as f:
            img_name_list = f.read().split()
        img_path_list = []
        label_path_list = []
        for img_name in img_name_list:
            img_path = os.path.join(self.voc_dir, "JPEGImages", img_name+".jpg")
            label_path = os.path.join(self.voc_dir, "SegmentationClass", img_name+".png")
            img = trv.io.read_image(img_path)
            if (img.shape[1] > self.crop_size[0] and img.shape[2] > self.crop_size[1]):
                img_path_list.append(img_path)
                label_path_list.append(label_path)
            
        img_paths   = np.array(img_path_list)
        label_paths = np.array(label_path_list)
        return img_paths, label_paths
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img = trv.io.read_image(self.img_paths[idx])
        img = self.transform(img)
        label = trv.io.read_image(self.label_paths[idx])
        rect = trv.transforms.RandomCrop.get_params(img, self.crop_size)
        img   = trv.transforms.functional.crop(img, *rect)
        label = trv.transforms.functional.crop(label, *rect)
        label = torch.squeeze(label)
        label = torch.where(label<22, label, 0.)
        label = label.long()
        return img, label
    
    def show_images(self):
        num_rows = 2
        num_cols = 5 
        scale = 4.
        figsize = (num_cols * scale, num_rows * scale)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
        fig.suptitle("VOC Image Segmentation", fontsize=20)
        fig.subplots_adjust(hspace=0.6, top=0.8)
        plt.tight_layout()
        axes = axes.flatten()
        total_imgs = self.__len__()
        index = np.arange(total_imgs)
        np.random.shuffle(index)
        index = index[:num_cols]
        mode = trv.io.image.ImageReadMode.RGB
        
        for i, idx in enumerate(index):
            img = trv.io.read_image(self.img_paths[idx])
            img = img.permute(1, 2, 0)
            axes[i].imshow(img)
            axes[i].axes.get_xaxis().set_visible(False)
            axes[i].axes.get_yaxis().set_visible(False)
        
        for i, idx in enumerate(index):
            img = trv.io.read_image(self.label_paths[idx], mode=mode)
            img = img.permute(1, 2, 0)
            axes[i+num_cols].imshow(img)
            axes[i+num_cols].axes.get_xaxis().set_visible(False)
            axes[i+num_cols].axes.get_yaxis().set_visible(False)
            
        plt.show()
        

class ResNet18_FCN(nn.Module):
    def __init__(self, NUM_CLASSES):
        super(ResNet18_FCN, self).__init__()
        self.num_classes = NUM_CLASSES
        self.Conv_base = self.build_ResNet18()
        self.final_conv = nn.Conv2d(512, NUM_CLASSES, kernel_size=1)
        self.transpose_conv = nn.ConvTranspose2d(NUM_CLASSES, NUM_CLASSES, 
                                                 kernel_size=64, padding=16, stride=32)
        self.weight_init()
    
    def build_ResNet18(self):
        resnet18 = trv.models.resnet18(weights=trv.models.ResNet18_Weights.IMAGENET1K_V1)
        Conv_base = nn.Sequential(*list(resnet18.children())[:-2])
        return Conv_base
    
    def weight_init(self):
        torch.nn.init.xavier_uniform_(self.final_conv.weight)
        W = bilinear_kernel(self.num_classes, self.num_classes, 64)
        self.transpose_conv.weight.data.copy_(W)
        
    def forward(self, img):
        x = self.Conv_base(img)
        x = self.final_conv(x)
        x = self.transpose_conv(x)
        return x

