
import os
from PIL import Image
import matplotlib.pyplot as plt
import glob
import numpy as np
import random
import tensorflow as tf
import cv2
import sys
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import json
import glob
import ntpath

import skimage.io as io
import scipy.io as sio

from skimage.io import imsave, imread
from skimage import img_as_ubyte
from skimage.transform import rescale, resize

sys.path.append("../")


class data_loader_SkyData():

    def __init__(self,dataset_name='train'):
        
        self.dataset_name=dataset_name

        if dataset_name=='train':
            self.img_path=glob.glob('../Dataset/SkyData/train_input/*')
            self.input_path='../Dataset/SkyData/train_input/'
            self.label_path='../Dataset/SkyData/train_label/'
            self.template_path='../Dataset/SkyData/train_template/'

        elif dataset_name=='val':
            self.img_path=glob.glob('../Dataset/SkyData/val_input/*')
            self.input_path='../Dataset/SkyData/val_input/'
            self.label_path='../Dataset/SkyData/val_label/'
            self.template_path='../Dataset/SkyData/val_template/'


        else:
            print ("no data load")
        random.shuffle(self.img_path)
        print (len(self.img_path))
        
        self.start=0

    def datasetSize(self):
        return len(self.img_path)

    def channel_norm(self,img):
        img=np.squeeze(img)
        for i in range(3):
            temp_max=np.max(img[:,:,i])
            temp_min=np.min(img[:,:,i])
            img[:,:,i] =(img[:,:,i]-temp_min)/(temp_max-temp_min+0.000001)
        return img

    def data_read_batch(self,batch_size=8):
        input_all=[]
        label_u=[]
        label_v=[]
        template_all=[]

        if self.start>len(self.img_path)-1:
            input_all=[0]
            label_u=[1]
            label_v=[2]
            template_all=[3]
            return np.asarray(input_all).astype(np.float32),np.asarray(label_u).astype(np.float32),np.asarray(label_v).astype(np.float32),np.asarray(template_all).astype(np.float32)

        for i in range(batch_size):

            if self.start>(len(self.img_path)-1):
                input_all=[0]
                label_u=[1]
                label_v=[2]
                template_all=[3]
                break




            img_name = self.img_path[self.start].split('/')[-1]

            self.start = self.start + 1

            input_img=plt.imread(self.input_path+img_name)/255.0
            input_img=self.channel_norm(input_img)

            template_img=plt.imread(self.template_path+img_name)/255.0
            template_img=self.channel_norm(template_img)

            with open(self.label_path+img_name[:(len(img_name)-4)]+'_label.txt', 'r') as outfile:
                data = json.load(outfile)

            u_list=[data['location'][0]['top_left_u'],data['location'][1]['top_right_u'],data['location'][3]['bottom_right_u'],data['location'][2]['bottom_left_u']]
            v_list=[data['location'][0]['top_left_v'],data['location'][1]['top_right_v'],data['location'][3]['bottom_right_v'],data['location'][2]['bottom_left_v']]

            input_all.append(input_img)
            label_u.append(u_list)
            label_v.append(v_list)
            template_all.append(template_img)

        return np.asarray(input_all).astype(np.float32),np.asarray(label_u).astype(np.float32),np.asarray(label_v).astype(np.float32),np.asarray(template_all).astype(np.float32)
