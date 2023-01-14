from PIL import Image,ImageDraw
import numpy as np
import glob
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from sklearn.linear_model import LinearRegression
from xml.dom import minidom
import random
import requests
import argparse
import cv2
from io import BytesIO
import os
import itertools
import json
from skimage.transform import rescale, resize


params={
    "wide_img_path":'./T_W/W/',
    "termal_img_path":"./T_W/T/",
    
    "termal_n_samples":10,
    
    "folders":['train','val'],
    "folder_kind":["input","template","label","template_original"],
    "file_type":".png",
    
    "output_region":300,
    "img_size":(700,860),
}



#create respective folders
for (f,t) in itertools.product(params['folders'],params['folder_kind']):
    folder=f"./{f}_{t}/"
    if not os.path.exists(str(folder)):
        os.makedirs(folder)



img_path_list=glob.glob(f"{params['wide_img_path']}*.png")


#generate samples

index=0
output_region=params["output_region"]
termal_img_path=params["termal_img_path"]
img_size_h,img_size_w=params["img_size"]


#
for i in img_path_list:
    index=index+1
    
    wide_img_name=i.split('/')[-1]
    img_wide=plt.imread(i)
    
    #adap to termal name and location 
    termal_img_name=wide_img_name[:-6]+""+wide_img_name[-4:]
    img_termal=plt.imread(termal_img_path+termal_img_name)
    

    img_name=wide_img_name[:-6]
    img_size_h=img_termal.shape[0]
    img_size_w=img_termal.shape[1]
    
    if (img_size_h < 700) or (img_size_w <700):
        continue
    
    for rand_num in range(50):
        
        center_x=random.randint(output_region+50,img_size_h-output_region-50)
        center_y=random.randint(output_region+50,img_size_w-output_region-50)

        #wide image cut input image
        squre_img=img_wide[(center_y-output_region):(center_y+output_region),
                        (center_x-output_region):(center_x+output_region),:3]
        squre_img=resize(squre_img,(192,192))

        if rand_num<40:
            plt.imsave('./train_input/'+str(rand_num)+'_'+img_name+'.png',squre_img)
        else:
            plt.imsave('./val_input/'+str(rand_num)+'_'+img_name+'.png',squre_img)
            
            
        #termal image cut input image
        squre_img=img_termal[(center_y-output_region):(center_y+output_region),
                        (center_x-output_region):(center_x+output_region),:3]
        squre_img=resize(squre_img,(192,192))

        if rand_num<40:
            plt.imsave('./train_template_original/'+str(rand_num)+'_'+img_name+'.png',squre_img)
        else:
            plt.imsave('./val_template_original/'+str(rand_num)+'_'+img_name+'.png',squre_img)
            
        
        #warp random corners to [[32,32],[159,32],[32,159],[159,159]]
        top_left_box_u=random.randint(0,63)
        top_left_box_v=random.randint(0,63)

        top_right_box_u=random.randint(128,191)
        top_right_box_v=random.randint(0,63)

        bottom_left_box_u=random.randint(0,63)
        bottom_left_box_v=random.randint(128,191)

        bottom_right_box_u=random.randint(128,191)
        bottom_right_box_v=random.randint(128,191)

         # prepare source and target four points
        src_points=[
            [top_left_box_u,top_left_box_v],
            [top_right_box_u,top_right_box_v],
            [bottom_left_box_u,bottom_left_box_v],
            [bottom_right_box_u,bottom_right_box_v]
        ]

        tgt_points=[[32,32],[159,32],[32,159],[159,159]]

        src_points=np.reshape(src_points,[4,1,2])
        tgt_points=np.reshape(tgt_points,[4,1,2])

        # find homography
        h_matrix, status = cv2.findHomography(src_points, tgt_points,0)


        warped_termal = cv2.warpPerspective(squre_img,h_matrix,(192,192))

        if rand_num<40:
            plt.imsave('./train_template/'+str(rand_num)+'_'+img_name+'.png',warped_termal[32:160,32:160,:])
        else:
            plt.imsave('./val_template/'+str(rand_num)+'_'+img_name+'.png',warped_termal[32:160,32:160,:])
        
        #save the used corners for gt homography
        label = {}
        label['location'] = []

        label['location'].append({
                'top_left_u':top_left_box_u,
                'top_left_v': top_left_box_v
            })
        label['location'].append({
                'top_right_u':top_right_box_u,
                'top_right_v':top_right_box_v
            })
        label['location'].append({
                'bottom_left_u':bottom_left_box_u,
                'bottom_left_v':bottom_left_box_v
            })
        label['location'].append({
                'bottom_right_u':bottom_right_box_u,
                'bottom_right_v':bottom_right_box_v
            })

        if rand_num<40:
            with open('./train_label/'+str(rand_num)+'_'+img_name+'_label.txt', 'w') as outfile:
                json.dump(label, outfile)
        else:
            with open('./val_label/'+str(rand_num)+'_'+img_name+'_label.txt', 'w') as outfile:
                json.dump(label, outfile)