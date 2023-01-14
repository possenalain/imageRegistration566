
import sys
sys.path.append('../')
from data_read import *
from utils import *
from net import *
import matplotlib.pyplot as plt
from backbone.backboneUtils import  *
import numpy as np
import pandas as pd
import argparse

import os


parser = argparse.ArgumentParser()

parser.add_argument('--dataset_name', action="store", dest= "dataset_name",default="SkyData",help='SkyData')
parser.add_argument('--epoch_eval', action="store", dest="epoch_eval", type=int, default=128 ,help='eval from which epoch')
parser.add_argument('--samples_to_test', action="store", dest="samples_to_test", type=int, default=5,help='samples to test')

parser.add_argument('--batch_size', action="store", dest="batch_size", type=int, default=16, help='batch_size')
parser.add_argument('--backbone', action="store", dest="backbone", type=int, default=6,help='backbone')

input_parameters = parser.parse_args()


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=35000)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)


save_path='./checkpoints/'+input_parameters.dataset_name+'/'


if not(os.path.exists('./checkpoints')):
    os.makedirs('./checkpoints')
if not(os.path.exists('./checkpoints/'+input_parameters.dataset_name)):
    os.makedirs('./checkpoints/'+input_parameters.dataset_name)
if not(os.path.exists('./results/')):
    os.makedirs('./results/')
if not(os.path.exists(save_path)):
    os.makedirs(save_path)




#backbone
load_path_backbone = './backbone/model/'

backbone_input = ResNet_first_input()
backbone_template = ResNet_first_template()

backbone_input.load_weights(load_path_backbone + 'epoch_' + str(input_parameters.backbone) + "input_full")
backbone_template.load_weights(load_path_backbone + 'epoch_' + str(input_parameters.backbone) + "template_full")

#head
regression_network=Net_first()
regression_network.load_weights(save_path + 'epoch_'+str(input_parameters.epoch_eval))



if input_parameters.dataset_name=='SkyData':
    data_loader_caller=data_loader_SkyData('val')

errors = []
error_homography=[]
error_ace=[]
error_total=[]

samples_to_test=data_loader_caller.datasetSize()
#samples_to_test = input_parameters.samples_to_test
for iters in range(samples_to_test):
    input_img, u_list, v_list, template_img = data_loader_caller.data_read_batch(batch_size=1)
    if len(np.shape(input_img)) < 2:
        break
    ## stage 1
    input_feature_one = backbone_input.call(input_img, training=False)
    template_feature_one = backbone_template.call(template_img, training=False)

    ## stage 2


    input_img_grey=input_feature_one
    template_feature_one=template_feature_one
    template_img_grey = tf.image.pad_to_bounding_box(template_feature_one, 32, 32, 192, 192)

    network_input=tf.concat([template_img_grey,input_img_grey],axis=-1)
    #ready network input



    homography_vector = regression_network.call(network_input, training=False)
    gt_vector = gt_motion_rs(u_list, v_list, batch_size=1)
    homography_loss = tf.reduce_mean(tf.math.sqrt((homography_vector - gt_vector) ** 2))
    corner_error = corner_loss(1, homography_vector, u_list, v_list)

    loss= homography_loss + corner_error*0.0001

    ##logs
    error_ace.append(np.float(corner_error))
    error_homography.append(np.float(homography_loss))
    error_total.append(np.float(loss))

    #print(f"Hg:{homography_vector.numpy()} \n Ht: {gt_vector}")


ErrorDict = {"ace": error_ace, "homography": error_homography, "total": error_total}

dtframedf = pd.DataFrame(ErrorDict)
dtframedf.to_csv(f'./results/ResultsAce{input_parameters.epoch_eval}.csv')