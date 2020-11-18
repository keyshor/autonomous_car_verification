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
middle_layer_size = 32

straight_begin_raw = np.loadtxt(os.path.join('..', 'training_data', 'straight_begin.csv'), delimiter=',')
square_right_begin_raw = np.loadtxt(os.path.join('..', 'training_data', 'right90_begin.csv'), delimiter=',')
square_left_begin_raw = np.loadtxt(os.path.join('..', 'training_data', 'left90_begin.csv'), delimiter=',')
sharp_right_begin_raw = np.loadtxt(os.path.join('..', 'training_data', 'right120_begin.csv'), delimiter=',')
sharp_left_begin_raw = np.loadtxt(os.path.join('..', 'training_data', 'left120_begin.csv'), delimiter=',')
straight_interior = np.loadtxt(os.path.join('..', 'training_data', 'straight_interior.csv'), delimiter=',')
square_right_interior = np.loadtxt(os.path.join('..', 'training_data', 'right90_interior.csv'), delimiter=',')
square_left_interior = np.loadtxt(os.path.join('..', 'training_data', 'left90_interior.csv'), delimiter=',')
sharp_right_interior = np.loadtxt(os.path.join('..', 'training_data', 'right120_interior.csv'), delimiter=',')
sharp_left_interior = np.loadtxt(os.path.join('..', 'training_data', 'left120_interior.csv'), delimiter=',')

def truncate_min(array_list):
    min_len = min([a.shape[0] for a in array_list])
    for a in array_list:
        np.random.shuffle(a)
    return min_len, [a[:min_len] for a in array_list]

num_begin, begin_list = truncate_min([straight_begin_raw, square_right_begin_raw, square_left_begin_raw, sharp_right_begin_raw, sharp_left_begin_raw])
straight_begin = begin_list[0]
square_right_begin = begin_list[1]
square_left_begin = begin_list[2]
sharp_right_begin = begin_list[3]
sharp_left_begin = begin_list[4]

interior_sample_size = 3 * num_begin

def truncate(a, n):
    assert(a.shape[0] >= n)
    np.random.shuffle(a)
    return a[:n]

def balanced_sample(array_list):
    min_len = min([a.shape[0] for a in array_list])
    for a in array_list:
        np.random.shuffle(a)
    return (
            np.concatenate([a[:min_len] for a in array_list]),
            np.concatenate([np.full(min_len, i) for i in range(len(array_list))])
            )

def load_straight_data():
    interior_points = np.concatenate([straight_begin, truncate(straight_interior, interior_sample_size)])
    goal_points = np.concatenate([
        square_right_begin,
        square_left_begin,
        sharp_right_begin,
        sharp_left_begin
        ])
    return balanced_sample([interior_points, goal_points])

def load_square_right_data():
    interior_points = np.concatenate([square_right_begin, truncate(square_right_interior, interior_sample_size)])
    goal_points = np.concatenate([
        straight_begin,
        square_left_begin,
        sharp_right_begin,
        sharp_left_begin
        ])
    return balanced_sample([interior_points, goal_points])

def load_square_left_data():
    interior_points = np.concatenate([square_left_begin, truncate(square_left_interior, interior_sample_size)])
    goal_points = np.concatenate([
        straight_begin,
        square_right_begin,
        sharp_right_begin,
        sharp_left_begin
        ])
    return balanced_sample([interior_points, goal_points])

def load_sharp_right_data():
    interior_points = np.concatenate([sharp_right_begin, truncate(sharp_right_interior, interior_sample_size)])
    goal_points = np.concatenate([
        straight_begin,
        square_right_begin,
        square_left_begin,
        sharp_left_begin
        ])
    return balanced_sample([interior_points, goal_points])

def load_sharp_left_data():
    interior_points = np.concatenate([sharp_left_begin, truncate(sharp_left_interior, interior_sample_size)])
    goal_points = np.concatenate([
        straight_begin,
        square_right_begin,
        square_left_begin,
        sharp_right_begin
        ])
    return balanced_sample([interior_points, goal_points])

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
    straight_train_data, straight_train_labels, = load_straight_data()
    straight_model = train_little(straight_train_data, straight_train_labels)
    straight_model.save('straight_little.h5')

    square_right_train_data, square_right_train_labels, = load_square_right_data()
    square_right_model = train_little(square_right_train_data, square_right_train_labels)
    square_right_model.save('square_right_little.h5')

    square_left_train_data, square_left_train_labels, = load_square_left_data()
    square_left_model = train_little(square_left_train_data, square_left_train_labels)
    square_left_model.save('square_left_little.h5')

    sharp_right_train_data, sharp_right_train_labels, = load_sharp_right_data()
    sharp_right_model = train_little(sharp_right_train_data, sharp_right_train_labels)
    sharp_right_model.save('sharp_right_little.h5')

    sharp_left_train_data, sharp_left_train_labels, = load_sharp_left_data()
    sharp_left_model = train_little(sharp_left_train_data, sharp_left_train_labels)
    sharp_left_model.save('sharp_left_little.h5')
