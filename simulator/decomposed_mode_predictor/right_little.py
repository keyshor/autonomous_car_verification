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

def load_data():
    not_goal = np.loadtxt('right_little_notgoal.csv', delimiter=',')
    goal = np.loadtxt('right_little_goal.csv', delimiter=',')
    notgoal_xy = np.loadtxt('right_little_notgoal_xy.csv', delimiter=',')
    goal_xy = np.loadtxt('right_little_goal_xy.csv', delimiter=',')
    neg, dim1 = not_goal.shape
    pos, dim2 = goal.shape
    neg_xy, dim3 = notgoal_xy.shape
    pos_xy, dim4 = goal_xy.shape
    assert(dim1 == num_lidar_rays and dim2 == num_lidar_rays)
    assert(neg_xy == neg and pos_xy == pos)
    assert(dim3 == 2 and dim4 == 2)
    total_data = np.concatenate([not_goal, goal])
    total_labels = np.concatenate([np.full(neg, 0), np.full(pos, 1)])
    total_xy = np.concatenate([notgoal_xy, goal_xy])
    return train_test_split(total_data, total_labels, total_xy)

def train_right_little(train_data, train_labels):
    class_weights = sklearn.utils.class_weight.compute_class_weight(
            'balanced', classes=np.unique(train_labels), y=train_labels
            )
    nn = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(num_lidar_rays,)),
        tf.keras.layers.Dense(32, activation='sigmoid'),
        tf.keras.layers.Dense(32, activation='sigmoid'),
        tf.keras.layers.Dense(1, activation='sigmoid')
        ])
    nn.compile(
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall()
                ]
            )
    nn.fit(
            train_data, train_labels,
            class_weight=dict(enumerate(class_weights)),
            epochs=num_epochs,
            validation_data=(test_data, test_labels),
            verbose=0
            )
    return nn

def plot_results(model, test_data, test_labels, test_xy, filename='right_little.png'):
    test_pred = m.predict(test_data).ravel().round()
    label_pos = np.equal(test_labels, 1)
    label_neg = np.equal(test_labels, 0)
    pred_pos = np.equal(test_pred, 1)
    pred_neg = np.equal(test_pred, 0)
    true_pos = test_xy[np.logical_and(label_pos, pred_pos)]
    false_pos = test_xy[np.logical_and(label_neg, pred_pos)]
    true_neg = test_xy[np.logical_and(label_neg, pred_neg)]
    false_neg = test_xy[np.logical_and(label_pos, pred_neg)]

    hallWidths, hallLengths, turns = square_hall_right(1.5)
    cur_dist_s = 0.75
    cur_dist_f = 12
    cur_heading = 0
    w = World(hallWidths, hallLengths, turns,
            0.75, 10, 0, 2.4, 70, 0.1, 115, num_lidar_rays)

    plt.figure(figsize=(12, 10))
    w.plotHalls()
    plt.ylim((-1, 11))
    plt.xlim((-1.75, 15))
    plt.tick_params(labelsize=20)
    for array, label, color in [
            (true_pos, 'true pos', 'g'),
            (true_neg, 'true neg', 'b'),
            (false_pos, 'false pos', 'r'),
            (false_neg, 'false neg', 'm')
            ]:
        plt.plot(
                array[:,0], array[:,1],
                f'{color}.', label=label,
                markersize=1
                )
    plt.legend(markerscale=10)
    plt.savefig(filename)

if __name__ == '__main__':
    train_data, test_data, train_labels, test_labels, train_xy, test_xy = load_data()
    m = train_right_little(train_data, train_labels)
    plot_results(m, test_data, test_labels, test_xy)
