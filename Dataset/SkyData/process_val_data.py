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
import json
from skimage.transform import rescale, resize



img_path_list=glob.glob('./val_input/*')

#rename validation images to just numbers

for i in range(len(img_path_list)):
    
    img_name=img_path_list[i].split('/')[-1]
    
    path_input='./val_input/'+img_name
    path_template='./val_template/'+img_name
    path_template_original='./val_template_original/'+img_name
    path_label='./val_label/'+img_name[:(len(img_name)-4)]+'_label.txt'

    os.rename(path_input,'./val_input/'+str(i)+'.png')
    os.rename(path_template,'./val_template/'+str(i)+'.png')
    os.rename(path_template_original,'./val_template_original/'+str(i)+'.png')
    os.rename(path_label,'./val_label/'+str(i)+'_label.txt')

