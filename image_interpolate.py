# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 18:06:22 2022

@author: acmor
"""
###############################################################################

import os
import cv2
import torch
import argparse
from os import listdir
from torch.nn import functional as F
import warnings

###############################################################################

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False) # deactivates gradient calculation
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
def load_model(modelDir:str):
    try:
        try:
            from model.RIFE_HDv3 import Model
            model = Model()
            model.load_model(modelDir, -1)
            print("Loaded v3.x HD model.")
        except:
            from train_log.RIFE_HDv2 import Model
            model = Model()
            model.load_model(modelDir, -1)
            print("Loaded v2.x HD model.")
    except:
        from model.RIFE_HD import Model
        model = Model()
        model.load_model(modelDir, -1)
        print("Loaded v1.x HD model")
    model.eval()
    model.device()
    return model

def interpolate_img(img_0:str,img_1:str,exp:int,model,ratio=0,rthreshold=0.02,
                    rmaxcycles=8):


    img0 = cv2.imread(img_0, cv2.IMREAD_UNCHANGED)
    img1 = cv2.imread(img_1, cv2.IMREAD_UNCHANGED)
    img0 =(torch.tensor(img0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    img1 =(torch.tensor(img1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
        
    n, c, h, w = img0.shape
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
    padding = (0, pw - w, 0, ph - h)
    img0 = F.pad(img0, padding)
    img1 = F.pad(img1, padding)
    
    if  ratio:
        img_list = [img0]
        img0_ratio = 0.0
        img1_ratio = 1.0
        if ratio <= img0_ratio + rthreshold / 2:
            middle = img0
        elif ratio >= img1_ratio - rthreshold / 2:
            middle = img1
        else:
            tmp_img0 = img0
            tmp_img1 = img1
            for inference_cycle in range(rmaxcycles):
                middle = model.inference(tmp_img0, tmp_img1)
                middle_ratio = ( img0_ratio + img1_ratio ) / 2
                if ratio - (rthreshold / 2) <= middle_ratio <= ratio + (rthreshold / 2):
                    break
                if  ratio > middle_ratio:
                    tmp_img0 = middle
                    img0_ratio = middle_ratio
                else:
                    tmp_img1 = middle
                    img1_ratio = middle_ratio
        img_list.append(middle)
        img_list.append(img1)
    else:
        img_list = [img0, img1]
        for i in range(exp):
            tmp = []
            for j in range(len(img_list) - 1):
                mid = model.inference(img_list[j], img_list[j + 1])
                tmp.append(img_list[j])
                tmp.append(mid)
            tmp.append(img1)
            img_list = tmp
    return img_list,h,w
def save_images(img_list:list,img_0:str,output_path:str,h:int,w:int):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    for i in range(len(img_list)):
        if img_0.endswith('.exr') and img_1.endswith('.exr'):
            cv2.imwrite('{}/img{}.exr'.format(output_path,i), (img_list[i][0]).cpu().numpy().transpose(1, 2, 0)[:h, :w], [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
        else:
            cv2.imwrite('{}/img{}.png'.format(output_path,i), (img_list[i][0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])


def interpolate_folder(files_path:str,output_path:str,exp:int,ratio=0,rthreshold=0.02,
                    rmaxcycles=8,modelDir='train_log'):
    images=[img for img in listdir(files_path) if img.endswith('.jpg')]  
    model=load_model(modelDir)
    interpolated=list()
    for i in range(len(images)-1):
        three,h,w=interpolate_img(files_path+'/'+images[i],files_path+'/'+images[i+1],
                       exp, model)
        if i==len(images)-2:
            interpolated=[*interpolated,*three]
        else:
            interpolated=[*interpolated,*three[:-1]]
    save_images(interpolated,files_path+'/'+images[0],output_path,h,w)
        
if __name__ == "__main__":

    interpolate_folder('./ratios/','./output_trial/',exp=1,ratio=0.2,rthreshold=0.2)