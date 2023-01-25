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


# Visualization
def initial_motion():
    # prepare source and target four points
    matrix_list = []
    for i in range(1):
        src_points = [[0, 0], [127, 0], [127, 127], [0, 127]]

        tgt_points = [[32, 32], [160, 32], [160, 160], [32, 160]]

        src_points = np.reshape(src_points, [4, 1, 2])
        tgt_points = np.reshape(tgt_points, [4, 1, 2])

        # find homography
        h_matrix, status = cv2.findHomography(src_points, tgt_points, 0)
        matrix_list.append(h_matrix)
    return np.asarray(matrix_list).astype(np.float32)

def construct_matrix(initial_matrix, scale_factor, batch_size):
    # scale_factor size_now/(size to get matrix)
    initial_matrix = tf.cast(initial_matrix, dtype=tf.float32)

    scale_matrix = np.eye(3) * scale_factor
    scale_matrix[2, 2] = 1.0
    scale_matrix = tf.cast(scale_matrix, dtype=tf.float32)
    scale_matrix_inverse = tf.linalg.inv(scale_matrix)

    scale_matrix = tf.expand_dims(scale_matrix, axis=0)
    scale_matrix = tf.tile(scale_matrix, [batch_size, 1, 1])

    scale_matrix_inverse = tf.expand_dims(scale_matrix_inverse, axis=0)
    scale_matrix_inverse = tf.tile(scale_matrix_inverse, [batch_size, 1, 1])

    final_matrix = tf.matmul(tf.matmul(scale_matrix, initial_matrix), scale_matrix_inverse)
    return final_matrix


def average_cornner_error(batch_size, predicted_matrix, u_list, v_list, top_left_u=0, top_left_v=0, bottom_right_u=127,
                          bottom_right_v=127):
    four_conner = [[top_left_u, top_left_v, 1], [bottom_right_u, top_left_v, 1], [bottom_right_u, bottom_right_v, 1],
                   [top_left_u, bottom_right_v, 1]]
    four_conner = np.asarray(four_conner)
    four_conner = np.transpose(four_conner)
    four_conner = np.expand_dims(four_conner, axis=0)
    four_conner = np.tile(four_conner, [batch_size, 1, 1]).astype(np.float32)

    new_four_points = tf.matmul(predicted_matrix, four_conner)

    new_four_points_scale = new_four_points[:, 2:, :]
    new_four_points = new_four_points / new_four_points_scale

    u_predict = new_four_points[:, 0, :]
    v_predict = new_four_points[:, 1, :]

    average_conner = tf.reduce_mean(tf.sqrt(tf.math.pow(u_predict - u_list, 2) + tf.math.pow(v_predict - v_list, 2)))

    return average_conner


def calculate_feature_map(input_tensor):
    bs, height, width, channel = tf.shape(input_tensor)
    path_extracted = tf.image.extract_patches(input_tensor, sizes=(1, 3, 3, 1), strides=(1, 1, 1, 1),
                                              rates=(1, 1, 1, 1), padding='SAME')
    path_extracted = tf.reshape(path_extracted, (bs, height, width, channel, 9))
    path_extracted_mean = tf.math.reduce_mean(path_extracted, axis=3, keepdims=True)

    # path_extracted_mean=tf.tile(path_extracted_mean,[1,1,1,channel,1])
    path_extracted = path_extracted - path_extracted_mean
    path_extracted_transpose = tf.transpose(path_extracted, (0, 1, 2, 4, 3))
    variance_matrix = tf.matmul(path_extracted_transpose, path_extracted)

    tracevalue = tf.linalg.trace(variance_matrix)
    row_sum = tf.reduce_sum(variance_matrix, axis=-1)
    max_row_sum = tf.math.reduce_max(row_sum, axis=-1)
    min_row_sum = tf.math.reduce_min(row_sum, axis=-1)
    mimic_ratio = (max_row_sum + min_row_sum) / 2.0 / tracevalue

    return tf.expand_dims(mimic_ratio, axis=-1)
