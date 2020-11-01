import numpy as np
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split

num_lidar_rays = 21
num_epochs = 15
goal_weight = 50

def train_right_little():
    not_goal = np.loadtxt('right_little_notgoal.csv', delimiter=',')
    goal = np.loadtxt('right_little_goal.csv', delimiter=',')
    neg, dim1 = not_goal.shape
    pos, dim2 = goal.shape
    assert(dim1 == num_lidar_rays and dim2 == num_lidar_rays)
    total_data = np.concatenate([not_goal, goal])
    total_labels = np.concatenate([np.full(neg, 0), np.full(pos, 1)])
    train_data, test_data, train_labels, test_labels, = \
            train_test_split(total_data, total_labels)
    class_weights = sklearn.utils.class_weight.compute_class_weight(
            'balanced', np.unique(train_labels), train_labels
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
            validation_data=(test_data, test_labels)
            )
    return nn

if __name__ == '__main__':
    m = train_right_little()
