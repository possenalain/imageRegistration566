
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

import random



params={
    "paths":'../T_W',
    "sequences":{
        '0018':{
                "h":550,
                "w":860,
                "samples_per_each":7,
                "val_count":2,
                "first_s":560,
                "out_region":260
            }, 
        '0027':{
                "h":1080,
                "w":1620,
                "samples_per_each":5,
                "val_count":1,
                "first_s":1050,
                "out_region":500
            },  
        '0009':{
                "h":700,
                "w":860,
                "samples_per_each":5,
                "val_count":1,
                "first_s":1000,
                "out_region":330
            },
        '0003':{
                "h":700,
                "w":860,
                "samples_per_each":10,
                "val_count":1,
                "first_s":300,
                "out_region":330
            }, 
        '0010':{
                "h":700,
                "w":860,
                "samples_per_each":4,
                "val_count":1,
                "first_s":2400,
                "out_region":330
            }
    },
    "folders":['train','val'],
    "folder_kind":["input","template","label","template_original"],
    "file_type":".png"
}



#create folders
for (f,t) in itertools.product(params['folders'],params['folder_kind']):
    folder=f"../{f}_{t}/"
    if not os.path.exists(str(folder)):
        os.makedirs(folder)


# ###  Generate data


for (seq_id,sequence_params) in params["sequences"].items():
    
    print(f"\n\nsequence------------------- {seq_id}")
    
    foldersWT = glob.glob(f"{params['paths']}/{seq_id}/*")
    foldersW=[wf for wf in foldersWT if not wf.endswith("T")]

    for fid,rgb_folder in enumerate(foldersW):
        t_folder=rgb_folder+"_T"

        print(rgb_folder,t_folder)

        rgb_images_path_list=glob.glob(f"{rgb_folder}/*.png")
        random.shuffle(rgb_images_path_list)
        

        for iid,rgb_img_path in enumerate(rgb_images_path_list[:sequence_params["first_s"]]):
            rgb_img_name=rgb_img_path.split('/')[-1]
            rgb_img=plt.imread(rgb_img_path)
            
            iidfid=f"{seq_id}_{fid}_{iid}-"
            
            #termal image name is exactly the same
            termal_img_name=rgb_img_name
            termal_img=None
            try:
                termal_img=plt.imread(f"{t_folder}/{termal_img_name}")
            except:
                print(f"no T for {t_folder}/{termal_img_name}")
                continue
            #print (f"{i} :{rgb_img_name},{rgb_img.shape}, {termal_img_name},{termal_img.shape}")
            
            #generate and distribute samples
            #generateAndSaveSamples(seq_id,sequence_params)
            
            
            img_name=rgb_img_name[:-4]
            img_size_h=sequence_params["h"]
            img_size_w=sequence_params["w"]

            samples_per_each=sequence_params["samples_per_each"]
            val_count=sequence_params["val_count"]
            val_start_at=samples_per_each-val_count
            output_region=sequence_params["out_region"]

            for rand_num in range(samples_per_each):

                center_x=random.randint(output_region+10,img_size_w-output_region-10)
                center_y=random.randint(output_region+10,img_size_h-output_region-10)

                #wide image cut input image
                squre_img=rgb_img[(center_y-output_region):(center_y+output_region),
                                (center_x-output_region):(center_x+output_region),:3]
                squre_img=resize(squre_img,(192,192))

                if rand_num<val_start_at:
                    plt.imsave('../train_input/'+iidfid+str(rand_num)+'_'+img_name+'.png',squre_img)
                else:
                    plt.imsave('../val_input/'+iidfid+str(rand_num)+'_'+img_name+'.png',squre_img)


                #termal image cut input image
                squre_img=termal_img[(center_y-output_region):(center_y+output_region),
                                (center_x-output_region):(center_x+output_region),:3]
                squre_img=resize(squre_img,(192,192))

                if rand_num<val_start_at:
                    plt.imsave('../train_template_original/'+iidfid+str(rand_num)+'_'+img_name+'.png',squre_img)
                else:
                    plt.imsave('../val_template_original/'+iidfid+str(rand_num)+'_'+img_name+'.png',squre_img)


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

                if rand_num<val_start_at:
                    plt.imsave('../train_template/'+iidfid+str(rand_num)+'_'+img_name+'.png',warped_termal[32:160,32:160,:])
                else:
                    plt.imsave('../val_template/'+iidfid+str(rand_num)+'_'+img_name+'.png',warped_termal[32:160,32:160,:])

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

                if rand_num<val_start_at:
                    with open('../train_label/'+iidfid+str(rand_num)+'_'+img_name+'_label.txt', 'w') as outfile:
                        json.dump(label, outfile)
                else:
                    with open('../val_label/'+iidfid+str(rand_num)+'_'+img_name+'_label.txt', 'w') as outfile:
                        json.dump(label, outfile)

