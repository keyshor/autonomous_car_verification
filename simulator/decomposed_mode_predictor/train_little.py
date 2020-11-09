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
        tf.keras.layers.Dense(32, activation='tanh'),
        tf.keras.layers.Dense(32, activation='tanh'),
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
            verbose=0
            )
    return nn

def plot_predictions(model, neg_label, test_data_file, test_position_file, xy_file, xh_file, yh_file):
    test_data = np.loadtxt(os.path.join('..', 'training_data', test_data_file), delimiter=',')
    test_position = np.loadtxt(os.path.join('..', 'training_data', test_position_file), delimiter=',')
    test_pred = model.predict(test_data).ravel().round()
    test_labels = np.not_equal(test_position[:, 3].ravel(), neg_label).astype(int)
    label_pos = np.equal(test_labels, 1)
    label_neg = np.equal(test_labels, 0)
    pred_pos = np.equal(test_pred, 1)
    pred_neg = np.equal(test_pred, 0)
    true_pos = test_position[np.logical_and(label_pos, pred_pos)]
    false_pos = test_position[np.logical_and(label_neg, pred_pos)]
    true_neg = test_position[np.logical_and(label_neg, pred_neg)]
    false_neg = test_position[np.logical_and(label_pos, pred_neg)]

    xy_fig, xy_ax = plt.subplots()
    xh_fig, xh_ax = plt.subplots()
    yh_fig, yh_ax = plt.subplots()
    #plt.tick_params(labelsize=20)
    for array, label, color in [
            (true_pos, 'true pos', 'g'),
            (true_neg, 'true neg', 'b'),
            (false_pos, 'false pos', 'r'),
            (false_neg, 'false neg', 'm')
            ]:
        xy_ax.plot(
                array[:,0], array[:,1],
                f'{color}.', label=label,
                markersize=1
                )
        xh_ax.plot(
                array[:,0], array[:,2],
                f'{color}.', label=label,
                markersize=1
                )
        yh_ax.plot(
                array[:,1], array[:,2],
                f'{color}.', label=label,
                markersize=1
                )
    xy_ax.legend(markerscale=10)
    xh_ax.legend(markerscale=10)
    yh_ax.legend(markerscale=10)
    xy_ax.set_xlabel('x coordinate')
    xy_ax.set_ylabel('y coordinate')
    xh_ax.set_xlabel('x coordinate')
    xh_ax.set_ylabel('heading')
    yh_ax.set_xlabel('y coordinate')
    yh_ax.set_ylabel('heading')
    xy_fig.savefig(xy_file)
    xh_fig.savefig(xh_file)
    yh_fig.savefig(yh_file)

    num_tp, _ = true_pos.shape
    num_pp = np.sum(pred_pos.astype(int))
    num_lp = np.sum(label_pos.astype(int))
    print(f'precision = {num_tp / num_pp}, recall = {num_tp / num_lp}')

if __name__ == '__main__':
    assert(all([dim_i == num_lidar_rays for dim_i in [dim1, dim2, dim3, dim4, dim5]]))
    straight_train_data, straight_train_labels, = load_straight_data()
    square_right_train_data, square_right_train_labels, = load_square_right_data()
    square_left_train_data, square_left_train_labels, = load_square_left_data()
    sharp_right_train_data, sharp_right_train_labels, = load_sharp_right_data()
    sharp_left_train_data, sharp_left_train_labels, = load_sharp_left_data()
    straight_model = train_little(straight_train_data, straight_train_labels)
    square_right_model = train_little(square_right_train_data, square_right_train_labels)
    square_left_model = train_little(square_left_train_data, square_left_train_labels)
    sharp_right_model = train_little(sharp_right_train_data, sharp_right_train_labels)
    sharp_left_model = train_little(sharp_left_train_data, sharp_left_train_labels)
    plot_predictions(straight_model, 0, 'right90_test.csv', 'right90_test_position.csv',
            'straight_xy.png', 'straight_xh.png', 'straight_yh.png')
    plot_predictions(square_right_model, 1, 'right90_test.csv', 'right90_test_position.csv',
            'square_right_xy.png', 'square_right_xh.png', 'square_right_yh.png')
    plot_predictions(square_left_model, 2, 'left90_test.csv', 'left90_test_position.csv',
            'square_left_xy.png', 'square_left_xh.png', 'square_left_yh.png')
    plot_predictions(sharp_right_model, 3, 'right120_test.csv', 'right120_test_position.csv',
            'sharp_right_xy.png', 'sharp_right_xh.png', 'sharp_right_yh.png')
    plot_predictions(sharp_left_model, 4, 'left120_test.csv', 'left120_test_position.csv',
            'sharp_left_xy.png', 'sharp_left_xh.png', 'sharp_left_yh.png')
    straight_model.save('straight_little.h5')
    square_right_model.save('square_right_little.h5')
    square_left_model.save('square_left_little.h5')
    sharp_right_model.save('sharp_right_little.h5')
    sharp_left_model.save('sharp_left_little.h5')
