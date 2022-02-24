# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 18:06:22 2022

@author: acmor
LICENCE-MIT
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

def img_to_tensor(img_0:str)->torch.tensor:
    img0 = cv2.imread(img_0, cv2.IMREAD_UNCHANGED)
    img0 =(torch.tensor(img0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    return img0

def interpolate_images(img_folder:str,output_path:str,exp:int,modelDir='train_log'):
    
    model=load_model(modelDir)
    img_list=[img_to_tensor(img_folder+'/'+img) 
              for img in listdir(img_folder) if img.endswith('.png')]
    
    img0=img_list[0]
    n, c, h, w = img0.shape
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
    padding = (0, pw - w, 0, ph - h)
    padded_images = [F.pad(img0, padding) for img0 in img_list]
    img_list=padded_images
    for i in range(exp):
        tmp = []
        for j in range(len(img_list) - 1):
            mid = model.inference(img_list[j], img_list[j + 1])
            tmp.append(img_list[j])
            tmp.append(mid)
        tmp.append(img_list[-1])
        img_list = tmp
    save_images(img_list,output_path,h,w)

def save_images(img_list:list,output_path:str,h:int,w:int):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    for i in range(len(img_list)):
        cv2.imwrite('{}/{}.png'.format(output_path,i), (img_list[i][0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])

def make_video(image_folder: str,png:bool,fps:int):
    video_name = image_folder + "/transition.mp4"
    term='.png'
    if png: term='.png'
    images = [img for img in listdir(image_folder) if img.endswith(term)]
    names=[int(i[:-4]) for i in images]
    file_names=sorted(names)
   # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    frame = cv2.imread(image_folder + "/" + images[0])
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name,0, fps, (width, height))
    for file in file_names:
        video.write(cv2.imread(image_folder + "/" + str(file)+term))
    cv2.destroyAllWindows()
    video.release()
    
if __name__ == "__main__":

    interpolate_images('./data-related/trial/','./data-related/outtrial_2/',exp=2)
    #make_video('./output_exp3/', True, 8)