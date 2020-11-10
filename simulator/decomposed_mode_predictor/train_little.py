import os
import sys
sys.path.append('..')
import numpy as np
import tensorflow as tf
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from Car import World, square_hall_right

num_lidar_rays = 21
num_epochs = 15
middle_layer_size = 256

straight = np.loadtxt(os.path.join('..', 'training_data', 'straight.csv'), delimiter=',')
square_right = np.loadtxt(os.path.join('..', 'training_data', 'right90.csv'), delimiter=',')
square_left = np.loadtxt(os.path.join('..', 'training_data', 'left90.csv'), delimiter=',')
sharp_right = np.loadtxt(os.path.join('..', 'training_data', 'right120.csv'), delimiter=',')
sharp_left = np.loadtxt(os.path.join('..', 'training_data', 'left120.csv'), delimiter=',')
num_straight, dim1 = straight.shape
num_square_right, dim2 = square_right.shape
num_square_left, dim3 = square_left.shape
num_sharp_right, dim4 = sharp_right.shape
num_sharp_left, dim5 = sharp_left.shape

def sample_size(array_list, size):
    avg_size = size // len(array_list)
    num_plus1 = size % len(array_list)
    for a in array_list:
        len_a, _ = a.shape
        assert(len_a > avg_size)
        np.random.shuffle(a)
    return np.concatenate([a[: avg_size + int(i < num_plus1)] for i, a in enumerate(array_list)])

def load_straight_data():
    return (
            np.concatenate([straight, sample_size([square_right, square_left, sharp_right, sharp_left], num_straight)]),
            np.concatenate([np.full(num_straight, 0), np.full(num_straight, 1)])
            )

def load_square_right_data():
    return (
            np.concatenate([square_right, sample_size([straight, square_left, sharp_right, sharp_left], num_square_right)]),
            np.concatenate([np.full(num_square_right, 0), np.full(num_square_right, 1)])
            )

def load_square_left_data():
    return (
            np.concatenate([square_left, sample_size([straight, square_right, sharp_right, sharp_left], num_square_left)]),
            np.concatenate([np.full(num_square_left, 0), np.full(num_square_left, 1)])
            )

def load_sharp_right_data():
    return (
            np.concatenate([sharp_right, sample_size([straight, square_right, square_left, sharp_right], num_sharp_right)]),
            np.concatenate([np.full(num_sharp_right, 0), np.full(num_sharp_right, 1)])
            )

def load_sharp_left_data():
    return (
            np.concatenate([sharp_left, sample_size([straight, square_right, square_left, sharp_left], num_sharp_left)]),
            np.concatenate([np.full(num_sharp_left, 0), np.full(num_sharp_left, 1)])
            )

def train_little(train_data, train_labels):
    class_weights = dict(enumerate(sklearn.utils.class_weight.compute_class_weight(
            'balanced', classes=np.unique(train_labels), y=train_labels
            )))
    nn = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(num_lidar_rays,)),
        tf.keras.layers.Dense(middle_layer_size, activation='tanh'),
        tf.keras.layers.Dense(middle_layer_size, activation='tanh'),
        tf.keras.layers.Dense(1, activation='sigmoid')
        ])
    nn.compile(
            #optimizer='adam',
            loss='binary_crossentropy',
            metrics=[
                #'accuracy',
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall()
                ]
            )
    nn.fit(
            train_data, train_labels,
            class_weight=class_weights,
            epochs=num_epochs,
            verbose=1
            )
    return nn

if __name__ == '__main__':
    assert(all([dim_i == num_lidar_rays for dim_i in [dim1, dim2, dim3, dim4, dim5]]))

    #straight_train_data, straight_train_labels, = load_straight_data()
    #straight_model = train_little(straight_train_data, straight_train_labels)
    #straight_model.save('straight_little.h5')

    square_right_train_data, square_right_train_labels, = load_square_right_data()
    square_right_model = train_little(square_right_train_data, square_right_train_labels)
    square_right_model.save('square_right_little.h5')

    #square_left_train_data, square_left_train_labels, = load_square_left_data()
    #square_left_model = train_little(square_left_train_data, square_left_train_labels)
    #square_left_model.save('square_left_little.h5')

    #sharp_right_train_data, sharp_right_train_labels, = load_sharp_right_data()
    #sharp_right_model = train_little(sharp_right_train_data, sharp_right_train_labels)
    #sharp_right_model.save('sharp_right_little.h5')

    #sharp_left_train_data, sharp_left_train_labels, = load_sharp_left_data()
    #sharp_left_model = train_little(sharp_left_train_data, sharp_left_train_labels)
    #sharp_left_model.save('sharp_left_little.h5')
