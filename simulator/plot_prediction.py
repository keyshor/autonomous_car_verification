import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def plot_predictions(model):
    test_data = np.loadtxt(
            os.path.join('training_data', 'right_test.csv'),
            delimiter=','
            )
    test_position = np.loadtxt(
            os.path.join('training_data', 'right_test_position.csv'),
            delimiter=','
            )
    test_pred = np.argmax(model.predict(test_data), axis=1)
    test_labels = test_position[:, 3].ravel()
    label_straight = np.equal(test_labels, 0)
    label_right = np.equal(test_labels, 1)
    label_left = np.equal(test_labels, 2)
    pred_straight = np.equal(test_pred, 0)
    pred_right = np.equal(test_pred, 1)
    pred_left = np.equal(test_pred, 2)
    true_straight = test_position[np.logical_and(label_straight, pred_straight)]
    true_right = test_position[np.logical_and(label_right, pred_right)]
    true_left = test_position[np.logical_and(label_left, pred_left)]
    false_straight = test_position[np.logical_and(np.logical_not(label_straight), pred_straight)]
    false_right = test_position[np.logical_and(np.logical_not(label_right), pred_right)]
    false_left = test_position[np.logical_and(np.logical_not(label_left), pred_left)]

    xy_fig, xy_ax = plt.subplots()
    xh_fig, xh_ax = plt.subplots()
    yh_fig, yh_ax = plt.subplots()
    #plt.tick_params(labelsize=20)
    for array, label, color in [
            (true_straight, 'true straight', 'g'),
            (true_right, 'true right', 'b'),
            (true_left, 'true left', 'c'),
            (false_straight, 'false straight', 'r'),
            (false_right, 'false right', 'm'),
            (false_left, 'false left', 'y'),
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
    xy_fig.savefig('right_turn_xy.png')
    xh_fig.savefig('right_turn_xh.png')
    yh_fig.savefig('right_turn_yh.png')

if __name__ == '__main__':
    model = tf.keras.models.load_model('modepredictor.h5')
    plot_predictions(model)
