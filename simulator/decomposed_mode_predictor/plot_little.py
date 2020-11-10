import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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
    model = tf.keras.models.load_model('square_right_little.h5')
    neg_label = 1
    plot_predictions(model, neg_label, 'right90_test.csv', 'right90_test_position.csv',
            'square_right_xy.png', 'square_right_xh.png', 'square_right_yh.png')
    plot_predictions(model, neg_label, 'left90_test.csv', 'left90_test_position.csv',
            'square_left_xy.png', 'square_left_xh.png', 'square_left_yh.png')
    plot_predictions(model, neg_label, 'right120_test.csv', 'right120_test_position.csv',
            'sharp_right_xy.png', 'sharp_right_xh.png', 'sharp_right_yh.png')
    plot_predictions(model, neg_label, 'left120_test.csv', 'left120_test_position.csv',
            'sharp_left_xy.png', 'sharp_left_xh.png', 'sharp_left_yh.png')
