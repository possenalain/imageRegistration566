
import sys
sys.path.append('../')
from data_read import *
from net import *
from utils import *
from backbone.backboneUtils import  *

import matplotlib.pyplot as plt
import numpy as np

import argparse
import pandas as pd
import os


print(tf.executing_eagerly())

parser = argparse.ArgumentParser()


parser.add_argument('--dataset_name', action="store", dest= "dataset_name",default="SkyData",help='SkyData')
parser.add_argument('--learning_rate', action="store", dest="learning_rate", type=float, default=0.0000005 ,help='learning_rate')
parser.add_argument('--batch_size', action="store", dest="batch_size", type=int, default=32, help='batch_size')
parser.add_argument('--save_eval_f', action="store", dest="save_eval_f", type=int, default=1600000,help='save and eval after how many iterations')
parser.add_argument('--epoch_start', action="store", dest="epoch_start", type=int, default=1,help='train from which epoch')
parser.add_argument('--epoch_decay', action="store", dest="epoch_decay", type=int, default=30,help='how many epoch to lower lr by 2')
parser.add_argument('--epoch_num', action="store", dest="epoch_num", type=int, default=50,help='how many epochs to train')

parser.add_argument('--backbone', action="store", dest="backbone", type=int, default=20,help='backbone')


input_parameters = parser.parse_args()


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=36000)])
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
if not(os.path.exists(save_path)):
    os.makedirs(save_path)
if not(os.path.exists('./logs/')):
    os.makedirs('./logs/')

lr=input_parameters.learning_rate


#backbone
load_path_backbone = './backbone/model/'

backbone_input = ResNet_first_input()
backbone_template = ResNet_first_template()

backbone_input.load_weights(load_path_backbone + 'epoch_' + str(input_parameters.backbone) + "input_full")
backbone_template.load_weights(load_path_backbone + 'epoch_' + str(input_parameters.backbone) + "template_full")


#Head
regression_network=Net_first()


if input_parameters.epoch_start>1:
    #load weights
    regression_network.load_weights(save_path + 'epoch_'+str(input_parameters.epoch_start-1))




for current_epoch in range(input_parameters.epoch_start-1,input_parameters.epoch_start+input_parameters.epoch_num-1):

    error_homography = []
    error_ace = []
    error_total = []

    if input_parameters.dataset_name=='SkyData':
        data_loader_caller=data_loader_SkyData('train')


    if current_epoch>0 and current_epoch%input_parameters.epoch_decay==0:
        if lr > 0.0000001:
            lr=lr*0.9

    #file = open(f"./logs/epoch-{current_epoch + 1}.txt", "w")
    optimizer = tf.keras.optimizers.Adam(lr=lr,beta_1=0.9)

    print("Starting epoch " + str(current_epoch+1))
    print("Learning rate is " + str(lr)) 


    for iters in range(data_loader_caller.datasetSize()+1):
        input_img,u_list,v_list,template_img=data_loader_caller.data_read_batch(batch_size=input_parameters.batch_size)
        

        if len(np.shape(input_img))<2:
          regression_network.save_weights(save_path +'epoch_'+str(current_epoch+1))
          break

        ## stage 1

        input_feature_one = backbone_input.call(input_img, training=False)
        template_feature_one = backbone_template.call(template_img, training=False)

        ## stage 2


        #input_img=input_img[:,:,:,:3]
        #template_img=template_img[:,:,:,:3]
        #input_img_grey=tf.image.rgb_to_grayscale(input_img)
        
        #template_img_new=tf.image.pad_to_bounding_box(template_img, 32, 32, 192, 192)
        
        #template_img_grey=tf.image.rgb_to_grayscale(template_img_new)




        input_img_grey=input_feature_one
        template_feature_one=template_feature_one
        #input_img_grey = input_feature_one[:, :, :, :3]
        #template_feature_one = template_feature_one[:, :, :, :3]
        template_img_grey = tf.image.pad_to_bounding_box(template_feature_one, 32, 32, 192, 192)


        
        network_input=tf.concat([template_img_grey,input_img_grey],axis=-1)
        #ready network input



        with tf.GradientTape() as tape:
            homography_vector=regression_network.call(network_input)

            gt_vector=gt_motion_rs(u_list,v_list,batch_size=input_parameters.batch_size)
            homography_loss= tf.reduce_mean(tf.math.sqrt((homography_vector - (gt_vector))**2))
            corner_error=corner_loss(input_parameters.batch_size,homography_vector,u_list,v_list)

            loss = (homography_loss) * (corner_error)

        all_parameters=regression_network.trainable_variables
           
        grads = tape.gradient(loss, all_parameters)
        #grads=[tf.clip_by_value(i,-0.1,0.1) for i in grads]
        optimizer.apply_gradients(zip(grads, all_parameters))

        #log the loss
        log = f"{iters} " \
              f"- H loss:{homography_loss} " \
              f"- corner loss:{corner_error} " \
              f"- loss:{loss} "
        error_ace.append(np.float(corner_error))
        error_homography.append(np.float(homography_loss))
        error_total.append(np.float(loss))

        print(log)
        #file.write(str(log) + '\n')

        #save the model
        if (iters*input_parameters.batch_size)%input_parameters.save_eval_f==0 and iters>0:
            regression_network.save_weights(save_path +'epoch_'+str(current_epoch+1)+str(iters))

    #file.close()
    ErrorDict = {"ace": error_ace, "homography": error_homography, "total": error_total}

    dtframedf = pd.DataFrame(ErrorDict)
    dtframedf.to_csv(f'./logs/{current_epoch + 1}.csv')