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
num_epochs = 30

def balanced_sample(array_list):
    min_len = min([a.shape[0] for i in array_list])
    for a in array_list:
        np.random.shuffle(a)
    return (
            np.concatenate([a[:min_len] for a in array_list]),
            np.concatenate([np.full(min_len, i) for i in range(len(array_list))])
            )

def load_train_data():
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
    assert(all([dim_i == num_lidar_rays for dim_i in [dim1, dim2, dim3, dim4, dim5]]))
    return balanced_sample([straight, square_right, squale_left, sharp_right, sharp_left])

def train_big(train_data, train_labels):
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
            epochs=num_epochs,
            verbose=1
            )
    return nn

def plot_predictions(model):
    test_data = np.loadtxt(os.path.join('..', 'training_data', 'right90_test.csv'), delimiter=',')
    test_position = np.loadtxt(os.path.join('..', 'training_data', 'right90_test_position.csv'), delimiter=',')
    test_pred = model.predict(test_data).ravel().round()
    test_labels = np.not_equal(test_position[:, 3].ravel(), 1).astype(int)
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
    xy_fig.savefig('right_little_xy.png')
    xh_fig.savefig('right_little_xh.png')
    yh_fig.savefig('right_little_yh.png')

    num_tp, _ = true_pos.shape
    num_pp = np.sum(pred_pos.astype(int))
    num_lp = np.sum(label_pos.astype(int))
    print(f'precision = {num_tp / num_pp}, recall = {num_tp / num_lp}')

if __name__ == '__main__':
    train_data, train_labels, = load_train_data()
    m = train_right_little(train_data, train_labels)
    plot_predictions(m)
