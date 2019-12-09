#Network에 사용되는 Function 정의

import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import numpy as np
import random
from scipy import misc

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def str2bool(x):
    return x.lower() in ('true')

def normalize(X_train, X_test):

    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_test

###########################
#data augmentation func
###########################
def _random_rotate(batch):
    for i in range(len(batch)):
        k = random.randint(0,3)
        if bool(random.getrandbits(1)):
            batch[i] = np.rot90(batch[i],k,axes = (0,1))
    return batch

def _random_salt_pepper(batch):
    batch_copy = batch.copy()
    
    salt_vs_pepper = 0.2
    amount = 0.04
    num_salt = np.ceil(32*32*amount*salt_vs_pepper)
    num_pepper = np.ceil(32*32*amount*(1-salt_vs_pepper))

    if bool(random.getrandbits(1)):
        for batch in batch_copy:
            coords = [np.random.randint(0,i-1,int(num_salt)) for i in batch.shape]
            batch_copy[coords[0],coords[1],:] = 255
       
            coords = [np.random.randint(0,i-1,int(num_pepper)) for i in batch.shape]
            batch_copy[coords[0],coords[1],:] = 0
    
    return batch_copy
def _random_brightness(batch):
    for i in range(len(batch)):
        k = random.randint(0,25)
        if bool(random.getrandbits(1)):
            batch[i] += k
    return batch
	
def _random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])

    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                      mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                       nw:nw + crop_shape[1]]
    return new_batch

def _random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch

def data_augmentation(batch, img_size):
    batch = _random_flip_leftright(batch)
    #batch = _random_rotate(batch)
    #batch = _random_brightness(batch)
    #batch = _random_salt_pepper(batch)
    batch = _random_crop(batch, [img_size, img_size], 4)
    return batch