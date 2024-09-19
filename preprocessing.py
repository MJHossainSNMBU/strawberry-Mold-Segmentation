
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split

np.random.seed(42)
tf.random.set_seed(42)

IMG_H = 512
IMG_W = 512

def load_dataset(train_image_dir, train_mask_dir, valid_image_dir, valid_mask_dir, test_image_dir, test_mask_dir):
    train_x = sorted(glob(os.path.join(train_image_dir, "*.png")))
    train_y = sorted(glob(os.path.join(train_mask_dir, "*.png")))

    valid_x = sorted(glob(os.path.join(valid_image_dir, "*.png")))
    valid_y = sorted(glob(os.path.join(valid_mask_dir, "*.png")))

    test_x = sorted(glob(os.path.join(test_image_dir, "*.png")))
    test_y = sorted(glob(os.path.join(test_mask_dir, "*.png")))

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def read_image(path):
    path = path.decode()
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (IMG_W, IMG_H))
    image = image / 255.0
    image = image.astype(np.float32)
    return image

def read_mask(path):
    path = path.decode()
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (IMG_W, IMG_H))
    mask = mask / 255.0
    mask = mask.astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)
    return mask

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([IMG_H, IMG_W, 3])
    y.set_shape([IMG_H, IMG_W, 1])
    return x, y

def tf_dataset(X, Y, batch=2):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    ds = ds.map(tf_parse).batch(batch).prefetch(10)
    return ds

# Example usage directories
train_image_dir = '/content/train2/images'
train_mask_dir = '/content/train2/masks'
valid_image_dir = '/content/val2/images'
valid_mask_dir = '/content/val2/masks'
test_image_dir = '/content/test2/images'
test_mask_dir = '/content/test2/masks'

(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(train_image_dir, train_mask_dir, valid_image_dir, valid_mask_dir, test_image_dir, test_mask_dir)
