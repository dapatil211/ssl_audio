import tarfile, shutil
import zipfile
import sys, os, urllib.request, tarfile, glob
import numpy as np
import cv2
import librosa
import librosa.core
import librosa.feature
import librosa.display
import matplotlib.pyplot as plt
import random
from utils import *
from tensorflow import keras
import logging
import constants

logger = logging.getLogger(__name__)

# wav file Input
def file_load(wav_name, mono=False):
    try:
        return librosa.load(wav_name, sr=constants.SAMPLING_RATE, mono=mono)
    except:
        logger.error("file_broken or not exists!! : {}".format(wav_name))
        raise ValueError()


def make_data(folder_name):
    result = []
    all_name = glob.glob(folder_name)
    for name in all_name:
        result.append(file_load(name)[0])
    return np.array(result)


def make_data_img(folder_name):
    result = []
    all_name = glob.glob(folder_name)
    for name in all_name:
        # result.append(file_load(name)[0])
        result.append(cv2.resize(to_sp(file_load(name)[0]), (224, 224)))
    return np.array(result)


# change wave data to stft
def to_sp(x, n_fft=512, hop_length=256):
    stft = librosa.stft(x, n_fft=n_fft, hop_length=hop_length)
    sp = librosa.amplitude_to_db(np.abs(stft))
    return sp


def to_img(x):
    result = []
    for i in range(len(x)):
        result.append(cv2.resize(to_sp(x[i]), (224, 224)))
    return np.array(result)


def pad_array(arr):
    M = max(len(a) for a in arr)
    return np.array([a + [np.nan] * (M - len(a)) for a in arr])


def to_img2(x):
    result = cv2.resize(to_sp(x), (224, 224))
    return np.array(result)


"""def create_data_downstream(x_normal, x_anomaly):
    X_total = np.concatenate((x_normal, x_anomaly))
    # make label
    y_total = np.zeros(len(X_total), dtype=int)
    y_total[len(x_normal):] = 1
    
    #y_total = keras.utils.to_categorical(y_total)
    
    #normalize data
    X_batch = []

    for j in range(len(X_total)):
        img = to_img2(X_total[j])
        img = normalization(img)
        X_batch.append(img) 
    X_total = np.array(X_batch)
    X_total = np.expand_dims(X_total, axis=-1)
    return X_total, y_total"""


def normalization(x, max_=None, min_=None, mean=None, sigma=None):
    # max_ = 15.6#np.max(x_train)
    # min_ = -74.7#np.min(x_train)
    # mean = 133.4
    # sigma = 31.595366

    result = cv2.resize(x, (224, 224))
    result = (result - min_) / (max_ - min_)
    return (result * 255 - mean) / sigma


def create_data_downstream_labels(
    x_normal, x_anomaly, max_=None, min_=None, mean=None, sigma=None
):
    X_total = np.concatenate((x_normal, x_anomaly))
    # make label
    y_total = np.zeros(len(X_total), dtype=int)
    y_total[len(x_normal) :] = 1

    # y_total = keras.utils.to_categorical(y_total)

    # normalize data
    X_total = create_data_downstream(X_total, max_, min_, mean, sigma)
    return X_total, y_total


def create_data_downstream(x_data, max_=None, min_=None, mean=None, sigma=None):
    # normalize data
    X_batch = []

    for j in range(len(x_data)):
        img = to_img2(x_data[j])
        img = normalization(img, max_, min_, mean, sigma)
        X_batch.append(img)
    x_data = np.array(X_batch)
    x_data = np.expand_dims(x_data, axis=-1)
    return x_data


# train_path = os.path.join(path, 'train', 'spoofing_attacks')
# dev_path = os.path.join(path, 'dev', 'spoofing_attacks')
from datetime import datetime


def create_dataset(path, max_=None, min_=None, mean=None, sigma=None):

    start = datetime.now()
    attack_number = 0
    y_all = []
    x_all = []
    for attack_class in CLASSES:
        path_folder = os.path.join(path, attack_class)
        # x_train = make_data(path_folder + "/*")
        # X_train_img = to_img(x_train)
        X_train_img = make_data_img(path_folder + "/*")
        y_train = np.full((1, len(X_train_img)), attack_number, dtype=int)
        if x_all == []:
            x_all = X_train_img
        else:
            x_all = np.concatenate((x_all, X_train_img), axis=0)
        # y_all = np.hstack((y_all, y_train))
        if y_all == []:
            y_all = y_train
        else:
            y_all = np.concatenate((y_all, y_train), axis=1)
        attack_number += 1
        print("Finished..." + attack_class)
    # then on all data
    """max_ = np.max(X_train_img) #15.6#
    min_ = np.min(X_train_img)

    X_train = (X_train_img-min_)/(max_-min_)

    mean = np.mean(X_train) #133.4
    sigma = np.std(X_train, dtype=np.float64) #31.595366

    X_train = (X_train*255-mean)/sigma

    X_train = np.expand_dims(X_train, axis=-1)
    
    y_all = tf.keras.utils.to_categorical(y_all, num_classes=len(CLASSES))
    print(X_train.shape)
    return X_train, y_all"""
    if max_ is None:
        max_ = np.max(x_all)  # 15.6#
        min_ = np.min(x_all)

    x_all = (x_all - min_) / (max_ - min_)

    if mean is None:
        mean = np.mean(x_all)  # 133.4
        sigma = np.std(x_all, dtype=np.float64)  # 31.595366

    x_all = (x_all * 255 - mean) / sigma

    x_all = np.expand_dims(x_all, axis=-1)

    # y_all = tf.keras.utils.to_categorical(y_all, num_classes=len(CLASSES))
    y_all = y_all.reshape(-1, 1)
    y_all = keras.utils.to_categorical(y_all, num_classes=len(CLASSES))
    print(x_all.shape, y_all.shape)
    end = datetime.now()
    print("#Time to execute: " + str(end - start))
    return x_all, y_all, max_, min_, mean, sigma
