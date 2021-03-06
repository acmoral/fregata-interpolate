# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 18:06:22 2022

@author: acmor
LICENCE-MIT from arxiv2020-RIFE 's repository 
"""
###############################################################################

import os
import cv2
import torch
import argparse
from os import listdir
from torch.nn import functional as F
import warnings
import datetime
import pandas as pd
import re
import tempfile

###############################################################################

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False) # deactivates gradient calculation
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
def load_model(modelDir:str):
    """
    This function loads the weights trained from the repository of arxiv2020-RIFE
    the weights are stored locally in the folder modelDir
    Parameters
    ----------
    modelDir : str
        The directory where the pretrained model is at.

    Returns
    -------
    model : TYPE
        An object from the class Model which stores the pretrained weights of
        the neural net from RIFE.

    """
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
    """
    This functions transforms an image located at PATH=img_0 returning a 
    torch tensor

    Parameters
    ----------
    img_0 : str
       Th location of the image to be converted to a tensor object in torcj??h

    Returns
    -------
    img0 : torch.tensor
        An object of torch of type tensor which stores numbers in uint8 

    """
    img0 = cv2.imread(img_0, cv2.IMREAD_UNCHANGED)
    img0 =(torch.tensor(img0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    return img0

def Update_df(j:int,add_up:int,df,mid):
    """
    

    Parameters
    ----------
    j : int
        The day_0 from which the next image was generated
    df : TYPE
        the dataframe where all info from the set of iamges is stored
    mid : TYPE
       the tf.tensor to be stored from the newly generated image

    Returns
    -------
    Updated dataframe with new image

    """
    date=df['dates'].iloc[j]
    date_num=df['day_num'].iloc[j]+add_up
    date_mid=date+datetime.timedelta(days=add_up)
    dic={'file_name':'generated','tensors':mid,'dates':date_mid,'day_num':date_num}
    df=df.append(dic,ignore_index=True)
    df=df.sort_values(by=['dates'])
    return df

def save_images(df:pd.DataFrame,output_path:str,h:int,w:int):
    """
    

    Parameters
    ----------
    df: pd.DataFrame
        The dataframe stores the tensors to be stored as images
        and the dates of the images
    output_path : str
        the output path of generated images
    h : int
        the height of the image
    w : int
        with of the image

    Returns
    -------
    updated data frame contaning the new names of the files, this is fed to 
    the video generating function

    """
    img_list=list(df['tensors'])
    names=list(df['dates'])
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    for i in range(len(img_list)):
        if (names[i].hour,names[i].minute,names[i].second)==(00,00,00):
            new_name=str(i)
        else:
            new_name=str(i)+'int'
        df['file_name'].iloc[i]=new_name
        cv2.imwrite('{}/{}.png'.format(output_path,new_name), 
                    (img_list[i][0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])
    df=df.sort_values(by=['dates'])
    return df 
 
def GetSort_dates(image_folder:str):
    """
    This function sorts the names of the images according to their date

    Parameters
    ----------
    image_folder : str
        The path to the folder that contains the images to be interpolated and 
        that need to be sorted by date

    Returns
    -------
    file_names : pd.DataFrame
        The dataframe containing the names of the files, the dates of the files

    """
    images = [img for img in listdir(image_folder) if img.endswith('.png')]
    names=[i[:-4] for i in images]
    file_names=pd.DataFrame(names, columns=['file_name'])
    dates=[]
    for file in file_names['file_name']:
       dates.append(datetime.datetime.strptime(file[11:19],'%Y%m%d'))
    file_names['dates']=dates
    file_names=file_names.sort_values(by=['dates'])
    return file_names

def interpolate_video(img_folder:str,output_path:str,exp:int,fps:int,
                      cdays=True,video=True,modelDir='train_log'):
    """
    This function returns an amount of interpolated images equal to the number
    of difference in dates between two images. So if there is date 2020/01/01 and
    next image is 2020/01/05 the function will generate 3 images. 
    The parameter exp allows to specify n number of intermediate images between two days
    

    Parameters
    ----------
    img_folder : str
        Location/path of input images
    output_path : str
       Location/path of  output generated images
    modelDir : tf.keras model
        The path of loaded pretrained model's weights. The default is 'train_log'.
    exp : int 
        The amount of intermediate images to generate
    fps : int
        Frames per second to generate the output video
    cdays: bool
        If true t generated firs the amount of iamges corresponging to missing days 
        in between two images
    video: bool
        if true generates a video
    modelDir:
        Where the model wrights are at
        
    Returns
    -------
    None.

    """
    model=load_model(modelDir)
    dated_df=GetSort_dates(img_folder)
    
    if listdir(output_path):
        video_files=True
    if video_files:
          dated_df=dated_df.tail(2)
          dated_df.reset_index(drop=True,inplace=True)
          os.rename(output_path+'/transition.mp4',output_path+'/temp.mp4')
    else:
        print('No video file exists. A new one is being generated')
        
    file_names=list(dated_df['file_name'])
    img_list=[img_to_tensor(img_folder+'/'+img+'.png') for img in file_names]
    
    dates= list(dated_df['dates']-dated_df['dates'].iloc[0])
    dated_df['day_num']=[date.days for date in dates]
    
    img0=img_list[0]
    n, c, h, w = img0.shape
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
    padding = (0, pw - w, 0, ph - h)
    padded_images = [F.pad(img0, padding) for img0 in img_list]
    img_list=padded_images
    dated_df['tensors']=img_list
    if cdays:
        lst = list(range(0,dated_df['day_num'].iloc[-1]+1))   
        while dated_df['day_num'].tolist() != lst: 
            img_list=list(dated_df['tensors'])
            for j in range(len(img_list) - 1):
                date_num0=dated_df['day_num'].iloc[j]
                date_num1=dated_df['day_num'].iloc[j+1]
                if date_num0+1==date_num1:
                    pass
                else:
                    mid = model.inference(img_list[j], img_list[j+1])
                    add_up=int((date_num1-date_num0)/2)
                    dated_df=Update_df(j,add_up,dated_df, mid)
                
    image_list=dated_df['tensors']    
                 
    if exp:
        for i in range(exp):
            for j in range(len(image_list) - 1):
               mid = model.inference(image_list[j], image_list[j+1])
               add_up=(1/2)**exp
               dated_df=Update_df(2*j,add_up,dated_df, mid)
    if video:
        makeSave_video(dated_df,output_path, fps,h,w,video_files)
             
def makeSave_video(df:pd.DataFrame,output_path:str,fps:int,h:int,w:int,old_videos:bool):
    """
    It takes the frames generated and saves it into a video, if a video already
    exists it reads the old video's frames and appends the last generated frames
    then it erases the old video.

    Parameters
    ----------
    image_folder : str
       Location of images to be interpolated
    png : bool
        Is the image png or jpg
    fps : int
        Amount of fps to generate the video in mp4 format
    h: int
        Height of video based on the height of the image
    w: int
        Width of wideo based on the width of the image
    old_videos: bool
        False if there is no old video
    Returns
    -------
    None.

    """
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    video_name = output_path + "/transition.mp4"
    frames =list(df['tensors'])
    frames=[(img[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
            for img in frames]
    video = cv2.VideoWriter(video_name,0, fps, (w,h))
    print('writing video [...]')
    if old_videos:
        curr_v = cv2.VideoCapture(output_path+'/temp.mp4') # reads old video 
        while curr_v.isOpened():
            r, frame = curr_v.read()    
            if not r:
                break
            video.write(frame)            
    for frame in frames:
        video.write(frame)
    
    cv2.destroyAllWindows()
    curr_v.release()
    video.release()
    os.remove(output_path+'/temp.mp4')

if __name__ == "__main__":
    interpolate_video('./data-related/T18PVS_Chl/','./data-related/T18PVS_OneEachDay/',exp=1,fps=9,cdays=True)
