import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def plot_predictions(model, test_data_file, test_position_file, xy_file, xh_file, yh_file):
    test_data = np.loadtxt(os.path.join('..', 'training_data', test_data_file), delimiter=',')
    test_position = np.loadtxt(os.path.join('..', 'training_data', test_position_file), delimiter=',')
    test_pred = np.argmax(model.predict(test_data), axis=1)
    test_labels = test_position[:, 3].ravel()
    label_straight = np.equal(test_labels, 0)
    label_square_right = np.equal(test_labels, 1)
    label_square_left = np.equal(test_labels, 2)
    label_sharp_right = np.equal(test_labels, 3)
    label_sharp_left = np.equal(test_labels, 4)
    pred_straight = np.equal(test_pred, 0)
    pred_square_right = np.equal(test_pred, 1)
    pred_square_left = np.equal(test_pred, 2)
    pred_sharp_right = np.equal(test_pred, 3)
    pred_sharp_left = np.equal(test_pred, 4)
    true_straight = test_position[np.logical_and(label_straight, pred_straight)]
    true_square_right = test_position[np.logical_and(label_square_right, pred_square_right)]
    true_square_left = test_position[np.logical_and(label_square_left, pred_square_left)]
    true_sharp_right = test_position[np.logical_and(label_sharp_right, pred_sharp_right)]
    true_sharp_left = test_position[np.logical_and(label_sharp_left, pred_sharp_left)]
    false_straight = test_position[np.logical_and(np.logical_not(label_straight), pred_straight)]
    false_square_right = test_position[np.logical_and(np.logical_not(label_square_right), pred_square_right)]
    false_square_left = test_position[np.logical_and(np.logical_not(label_square_left), pred_square_left)]
    false_sharp_right = test_position[np.logical_and(np.logical_not(label_sharp_right), pred_sharp_right)]
    false_sharp_left = test_position[np.logical_and(np.logical_not(label_sharp_left), pred_sharp_left)]

    xy_fig, xy_ax = plt.subplots()
    xh_fig, xh_ax = plt.subplots()
    yh_fig, yh_ax = plt.subplots()
    #plt.tick_params(labelsize=20)
    for array, label, color in [
            (true_straight, 'true straight', 'g'),
            (true_square_right, 'true square right', 'b'),
            (true_square_left, 'true square left', 'c'),
            (true_sharp_right, 'true sharp right', 'lime'),
            (true_sharp_left, 'true sharp left', 'slategray'),
            (false_straight, 'false straight', 'r'),
            (false_square_right, 'false square right', 'm'),
            (false_square_left, 'false square left', 'y'),
            (false_sharp_right, 'false sharp right', 'orange'),
            (false_sharp_left, 'false sharp left', 'brown'),
            ]:
        xy_ax.plot(
                array[:,0], array[:,1],
                color=color, marker='.', linestyle='None', label=label,
                markersize=1
                )
        xh_ax.plot(
                array[:,0], array[:,2],
                color=color, marker='.', linestyle='None', label=label,
                markersize=1
                )
        yh_ax.plot(
                array[:,1], array[:,2],
                color=color, marker='.', linestyle='None', label=label,
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

if __name__ == '__main__':
    m = tf.keras.models.load_model('big.h5')
    plot_predictions(m, 'right90_test.csv', 'right90_test_position.csv',
            'square_right_xy.png', 'square_right_xh.png', 'square_right_yh.png')
    plot_predictions(m, 'left90_test.csv', 'left90_test_position.csv',
            'square_left_xy.png', 'square_left_xh.png', 'square_left_yh.png')
    plot_predictions(m, 'right120_test.csv', 'right120_test_position.csv',
            'sharp_right_xy.png', 'sharp_right_xh.png', 'sharp_right_yh.png')
    plot_predictions(m, 'left120_test.csv', 'left120_test_position.csv',
            'sharp_left_xy.png', 'sharp_left_xh.png', 'sharp_left_yh.png')
