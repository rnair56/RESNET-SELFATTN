import os
import cv2
import numpy as np
from glob import glob
from scipy.io import loadmat
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def read_image(image_path, mask=False):
    image = tf.io.read_file(image_path)
    if mask:
        image = tf.image.decode_png(image, channels=1)
        image.set_shape([None, None, 1])
        image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
    else:
        image = tf.image.decode_png(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
        image = image / 127.5 - 1
    return image


def load_data(image_list, mask_list):
    image = read_image(image_list)
    mask = read_image(mask_list, mask=True)
    return image, mask


def data_generator(image_list, mask_list):
    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    return dataset


if __name__ == "__main__":
    celabAHQ_path = "/Users/rahulnair/Desktop/personal/UB/courses/cse 673 Comp Vision/Assig2_Segmentation/CelebAMask-HQ"
    ##Parse the data according to required format
    ##

    IMAGE_SIZE = 512
    BATCH_SIZE = 4
    NUM_CLASSES = 20
    DATA_DIR = celabAHQ_path
    NUM_TRAIN_IMAGES = 1000
    NUM_VAL_IMAGES = 50

    train_images = sorted(glob(os.path.join(DATA_DIR, "CelebA-HQ-img/*")))[:NUM_TRAIN_IMAGES]
    train_masks = sorted(glob(os.path.join(DATA_DIR, "CelebAMask-HQ-mask-anno/*")))[:NUM_TRAIN_IMAGES]
    val_images = sorted(glob(os.path.join(DATA_DIR, "CelebA-HQ-img/*")))[
                 NUM_TRAIN_IMAGES: NUM_VAL_IMAGES + NUM_TRAIN_IMAGES
                 ]
    val_masks = sorted(glob(os.path.join(DATA_DIR, "CelebAMask-HQ-mask-anno/*")))[
                NUM_TRAIN_IMAGES: NUM_VAL_IMAGES + NUM_TRAIN_IMAGES
                ]

    train_dataset = data_generator(train_images, train_masks)
    val_dataset = data_generator(val_images, val_masks)

    print("Train Dataset:", train_dataset)
    print("Val Dataset:", val_dataset)
