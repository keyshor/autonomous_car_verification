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

def sample_size(a, b, size):
    len_a, _ = a.shape
    len_b, _ = b.shape
    assert len_a + len_b >= size
    np.random.shuffle(a)
    np.random.shuffle(b)
    if len_a <= size // 2:
        return np.concatenate([a, b[: size - len_a, :]])
    elif len_b <= size // 2:
        return np.concatenate([a[: size - len_b, :], b])
    else:
        return np.concatenate([a[: size // 2, :], b[: size - size // 2, :]])

def load_train_data():
    straight = np.loadtxt('straight.csv', delimiter=',')
    right = np.loadtxt('right.csv', delimiter=',')
    left = np.loadtxt('left.csv', delimiter=',')
    num_straight, dim1 = straight.shape
    num_right, dim2 = right.shape
    num_left, dim3 = left.shape
    assert(all([dim_i == num_lidar_rays for dim_i in [dim1, dim2, dim3]]))
    total_data = np.concatenate([right, sample_size(straight, left, num_right)])
    total_labels = np.concatenate([np.full(num_right, 0), np.full(num_right, 1)])
    return total_data, total_labels

def train_right_little(train_data, train_labels):
    class_weights = dict(enumerate(sklearn.utils.class_weight.compute_class_weight(
            'balanced', classes=np.unique(train_labels), y=train_labels
            )))
    nn = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(num_lidar_rays,)),
        tf.keras.layers.Dense(64, activation='tanh'),
        tf.keras.layers.Dense(64, activation='tanh'),
        tf.keras.layers.Dense(1, activation='sigmoid')
        ])
    nn.compile(
            #optimizer='adam',
            loss='binary_crossentropy'#,
            #metrics=[
            #    'accuracy',
            #    tf.keras.metrics.Precision(),
            #    tf.keras.metrics.Recall()
            #    ]
            )
    nn.fit(
            train_data, train_labels,
            class_weight=class_weights,
            epochs=num_epochs,
            verbose=0
            )
    return nn

def plot_predictions(model):
    test_data = np.loadtxt('right_test.csv', delimiter=',')
    test_position = np.loadtxt('right_test_position.csv', delimiter=',')
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
