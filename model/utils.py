import sys
sys.path.append('../')
from data_read import *
from net import *
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv



import math

import argparse

import os


def corner_loss(batch_size, network_output, u_list, v_list, top_left_u=0, top_left_v=0, bottom_right_u=127,
                bottom_right_v=127):
    four_conner = [[top_left_u, top_left_v, 1], [bottom_right_u, top_left_v, 1], [bottom_right_u, bottom_right_v, 1],
                   [top_left_u, bottom_right_v, 1]]
    four_conner = np.asarray(four_conner)
    four_conner = np.transpose(four_conner)
    four_conner = np.expand_dims(four_conner, axis=0)
    four_conner = np.tile(four_conner, [batch_size, 1, 1]).astype(np.float32)
    four_conner = tf.dtypes.cast(four_conner, tf.float32)

    extra = tf.ones((batch_size, 1))
    network_output= network_output
    predicted_matrix = tf.concat([network_output, extra], axis=-1)

    predicted_matrix = tf.reshape(predicted_matrix, [batch_size, 3, 3])

    new_four_points = tf.matmul(predicted_matrix, four_conner)

    new_four_points_scale = new_four_points[:, 2:, :]
    new_four_points = new_four_points / new_four_points_scale

    u_predict = new_four_points[:, 0, :]
    v_predict = new_four_points[:, 1, :]
    average_conner = tf.math.pow(u_predict - u_list, 2) + tf.math.pow(v_predict - v_list, 2)
    # print (np.shape(average_conner))
    average_conner = tf.reduce_mean(tf.math.sqrt(average_conner))

    return average_conner
def gt_motion_rs(u_list, v_list, batch_size=1):
    # prepare source and target four points
    matrix_list = []
    for i in range(batch_size):
        src_points = [[0, 0], [127, 0], [127, 127], [0, 127]]

        # tgt_points=[[32*2+1,32*2+1],[160*2+1,32*2+1],[160*2+1,160*2+1],[32*2+1,160*2+1]]

        tgt_points = np.concatenate([u_list[i:(i + 1), :], v_list[i:(i + 1), :]], axis=0)
        tgt_points = np.transpose(tgt_points)
        tgt_points = np.expand_dims(tgt_points, axis=1)

        src_points = np.reshape(src_points, [4, 1, 2])
        tgt_points = np.reshape(tgt_points, [4, 1, 2])

        # find homography
        h_matrix, status = cv2.findHomography(src_points, tgt_points, 0)

        matrix_list.append(np.squeeze(np.reshape(h_matrix, (1, 9))[:, :8]))
    return np.asarray(matrix_list).astype(np.float32)


###visualization
def padImage(image, destShape):
    in_size = image.shape
    out_size = destShape

    channels = image.shape[2]

    padd_size = out_size[0] - in_size[0]
    left_top = math.ceil(padd_size / 2)
    right_bottom = int(padd_size / 2)

    paddedImage = np.array([np.pad(image[:, :, t], (left_top, right_bottom), 'constant') for t in range(channels)])
    paddedImage = np.stack([t for t in paddedImage], axis=2)
    return paddedImage

def pileAndConcatenate(showcase):
    destSizeInput = showcase["input_fmap"]["input"].shape[1:]
    detSizeTemplate = showcase["template_fmap"]["template"].shape[1:]

    new_showcase = {
        "sample": showcase["sample"],
        "input_fmap": {
            "input": padImage(showcase["input_fmap"]["input"][0, :, :, :3], destSizeInput)
        },

        "template_fmap": {
            "template": padImage(showcase["template_fmap"]["template"][0, :, :, :3], detSizeTemplate)
        },
        "H": {
            "gt_vector": showcase["H"]["gt_vector"],
            "predicted": showcase["H"]["predicted"]
        },
        "UV": {
            "u_list": showcase["UV"]["u_list"],
            "v_list": showcase["UV"]["v_list"]
        },
        "corner_error": showcase["corner_error"]
    }

    return new_showcase

def registerAndSave(new_showcase):
    #     predicted_H=new_showcase["H"]["predicted"]
    #     predicted_H=np.squeeze(predicted_H)

    gt_matrix = np.concatenate((new_showcase["H"]["gt_vector"], [[1]]), -1).reshape(3, 3)
    predicted_matrix = np.concatenate((new_showcase["H"]["predicted"].numpy(), [[1]]), -1).reshape(3, 3)

    predicted_H = predicted_matrix

    template = new_showcase["template_fmap"]["template"]
    input = new_showcase["input_fmap"]["input"]

    # plt.imshow(input)
    # plt.show()

    warped = cv2.warpPerspective(template, np.squeeze(predicted_H), (192, 192))

    fused = (.4) * warped + 1 * input
    # plt.imshow(fused)
    # plt.show()

    new_showcase["UV"]

    corners = []

    for i, u in enumerate(new_showcase["UV"]["u_list"][0]):
        c = [u, new_showcase["UV"]["v_list"][0][i]]
        corners.append(c)

    src = [corners[0], corners[1], corners[3], corners[2]]

    src_points = src
    tgt_points = [[32, 32], [159, 32], [32, 159], [159, 159]]

    src_points = np.reshape(src_points, [4, 1, 2])
    tgt_points = np.reshape(tgt_points, [4, 1, 2])

    # find homography
    h1, status = cv2.findHomography(src_points, tgt_points, 0)

    src_points = [[32, 32], [159, 32], [32, 159], [159, 159]]
    tgt_points = [[0, 0], [127, 0], [0, 127], [127, 127]]

    src_points = np.reshape(src_points, [4, 1, 2])
    tgt_points = np.reshape(tgt_points, [4, 1, 2])

    # find homography
    h2, status = cv2.findHomography(src_points, tgt_points, 0)

    h = np.matmul(h2, h1)

    #     InverseH=inv(h)
    InverseH = gt_matrix
    warpedgt = cv2.warpPerspective(template, np.squeeze(InverseH), (192, 192))

    fusedgt = (1) * warpedgt + 1 * input
    fusedImg = np.hstack([fusedgt, fused])
    # plt.imshow(fusedImg)
    # plt.show()

    if not (os.path.exists('./results/registration')):
        print("creating one")
        os.makedirs('./registration/')
    fusedImg = cv2.convertScaleAbs(fusedImg, alpha=(255.0))
    fusedImg = fusedImg * 255

    cv2.imwrite(f"./results/registration/registered-{new_showcase['sample']}.png", fusedImg)