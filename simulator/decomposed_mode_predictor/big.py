import numpy as np
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split

num_lidar_rays = 21

def train_big():
    straight = np.loadtxt('big_straight.csv', delimiter=',')
    right = np.loadtxt('big_right.csv', delimiter=',')
    left = np.loadtxt('big_left.csv', delimiter=',')
    num_straight, dim1 = straight.shape
    num_right, dim2 = right.shape
    num_left, dim3 = left.shape
    assert(all([dim_i == num_lidar_rays for dim_i in [dim1, dim2, dim3]]))
    total_data = np.concatenate([straight, right, left])
    total_labels = np.concatenate([
        np.full((num_straight,), 0),
        np.full((num_right,), 1),
        np.full((num_left,), 2)
        ])
    train_data, test_data, train_labels, test_labels = train_test_split(total_data, total_labels)
    class_weights = sklearn.utils.class_weight.compute_class_weight('balanced', np.unique(train_labels), train_labels)
    m = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(num_lidar_rays,)),
        tf.keras.layers.Dense(32, activation='sigmoid'),
        tf.keras.layers.Dense(32, activation='sigmoid'),
        tf.keras.layers.Dense(3, activation='softmax')
        ])
    m.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    m.fit(train_data, train_labels,
            validation_data=(test_data, test_labels),
            epochs=15, class_weight=dict(enumerate(class_weights))
            )
    return m

if __name__ == '__main__':
    m = train_big()
