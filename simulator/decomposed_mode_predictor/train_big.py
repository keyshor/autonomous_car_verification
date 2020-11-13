import os
import numpy as np
import tensorflow as tf

num_lidar_rays = 21
num_epochs = 15
middle_layer_size = 32

def balanced_sample(array_list):
    min_len = min([a.shape[0] for a in array_list])
    for a in array_list:
        np.random.shuffle(a)
    return (
            np.concatenate([a[:min_len] for a in array_list]),
            np.concatenate([np.full(min_len, i) for i in range(len(array_list))])
            )

def load_train_data():
    straight_begin = np.loadtxt(os.path.join('..', 'training_data', 'straight_begin.csv'), delimiter=',')
    square_right_begin = np.loadtxt(os.path.join('..', 'training_data', 'right90_begin.csv'), delimiter=',')
    square_left_begin = np.loadtxt(os.path.join('..', 'training_data', 'left90_begin.csv'), delimiter=',')
    sharp_right_begin = np.loadtxt(os.path.join('..', 'training_data', 'right120_begin.csv'), delimiter=',')
    sharp_left_begin = np.loadtxt(os.path.join('..', 'training_data', 'left120_begin.csv'), delimiter=',')
    num_straight_begin, dim1 = straight_begin.shape
    num_square_right_begin, dim2 = square_right_begin.shape
    num_square_left_begin, dim3 = square_left_begin.shape
    num_sharp_right_begin, dim4 = sharp_right_begin.shape
    num_sharp_left_begin, dim5 = sharp_left_begin.shape
    assert(all([dim_i == num_lidar_rays for dim_i in [dim1, dim2, dim3, dim4, dim5]]))
    return balanced_sample([
        straight_begin,
        square_right_begin,
        square_left_begin,
        sharp_right_begin,
        sharp_left_begin
        ])

def train_big(train_data, train_labels):
    nn = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(num_lidar_rays,)),
        tf.keras.layers.Dense(middle_layer_size, activation='tanh'),
        tf.keras.layers.Dense(middle_layer_size, activation='tanh'),
        tf.keras.layers.Dense(5, activation='softmax')
        ])
    nn.compile(
            #optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=[
                'accuracy'#,
                #tf.keras.metrics.Precision(),
                #tf.keras.metrics.Recall()
                ]
            )
    nn.fit(
            train_data, train_labels,
            epochs=num_epochs,
            verbose=1
            )
    return nn

if __name__ == '__main__':
    train_data, train_labels, = load_train_data()
    m = train_big(train_data, train_labels)
    m.save('big.h5')
