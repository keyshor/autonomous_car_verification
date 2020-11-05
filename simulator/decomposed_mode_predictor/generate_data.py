import sys
import csv
sys.path.append('..')
from Car import World
from Car import square_hall_right
from Car import square_hall_left
from Car import trapezoid_hall_sharp_right
from Car import triangle_hall_sharp_right
from Car import triangle_hall_equilateral_right
import numpy as np
import random
def normalize(s):
    mean = [2.5]
    spread = [5.0]
    return (s - mean) / spread

def generate_data(iter_batch=200):
    width = 1.5
    (hallWidths, hallLengths, turns) = square_hall_right(width)
    car_V = 2.4
    episode_length = 65
    time_step = 0.1
    state_feedback = False
    lidar_field_of_view = 115
    lidar_num_rays = 21
    lidar_noise = 0
    missing_lidar_rays = 0

    car_dist_s = width/2
    car_dist_f = 5 + width
    car_heading = 0
    heading_range = 0.3
    pos_range = 0.3
    offset = 0.025
    iter_batch = 200

    straight_file = open('straight.csv', mode='x', newline='')
    right_file = open('right.csv', mode='x', newline='')
    left_file = open('left.csv', mode='x', newline='')
    right_test_file = open('right_test.csv', mode='x', newline='')
    left_test_file = open('left_test.csv', mode='x', newline='')
    right_test_position_file = open('right_test_position.csv', mode='x', newline='')
    left_test_position_file = open('left_test_position.csv', mode='x', newline='')
    straight = csv.writer(straight_file)
    right = csv.writer(right_file)
    left = csv.writer(left_file)
    right_test = csv.writer(right_test_file)
    left_test = csv.writer(left_test_file)
    right_test_position = csv.writer(right_test_position_file)
    left_test_position = csv.writer(left_test_position_file)

    cur_dist_s = width/2
    cur_dist_f = 7
    cur_heading = 0

    #train_data = []
    #train_labels = []

    label0 = 0
    label1 = 0
    label2 = 0
    #labels: 1 - straight, 2 - right, 3 - left

    diff = 1
    diff2 = 0.5

    while cur_dist_f >= 5 - diff:
        for i in range(iter_batch):
            car_dist_s = cur_dist_s + np.random.uniform(-pos_range, pos_range)
            car_dist_f = cur_dist_f
            car_heading = cur_heading + np.random.uniform(-heading_range, heading_range)
            w = World(hallWidths, hallLengths, turns,\
                    car_dist_s, car_dist_f, car_heading, car_V,\
                    episode_length, time_step, lidar_field_of_view,\
                    lidar_num_rays, lidar_noise, missing_lidar_rays, state_feedback=state_feedback)
            obs = w.scan_lidar()
            label = 0
            #label0+=1
            #train_data.append(normalize(obs))
            #train_labels.append(label)
            straight.writerow(normalize(obs))
            right_test.writerow(normalize(obs))
            right_test_position.writerow([w.car_global_x, w.car_global_y, w.car_global_heading, label])
        cur_dist_f -= offset

    while cur_dist_f >= width:
        for i in range(iter_batch):
            car_dist_s = cur_dist_s + np.random.uniform(-pos_range, pos_range)
            car_dist_f = cur_dist_f
            car_heading = cur_heading + np.random.uniform(-heading_range, heading_range)
            w = World(hallWidths, hallLengths, turns,\
                    car_dist_s, car_dist_f, car_heading, car_V,\
                    episode_length, time_step, lidar_field_of_view,\
                    lidar_num_rays, lidar_noise, missing_lidar_rays, state_feedback=state_feedback)
            obs = w.scan_lidar()
            label = 1
            #label1+=1
            #train_data.append(normalize(obs))
            #train_labels.append(label)
            right.writerow(normalize(obs))
            right_test.writerow(normalize(obs))
            right_test_position.writerow([w.car_global_x, w.car_global_y, w.car_global_heading, label])
        cur_dist_f -= offset
    cur_heading = -np.pi/4

    while cur_dist_f >= width/2 - pos_range:
        for i in range(iter_batch):
            car_dist_s = cur_dist_s + np.random.uniform(-pos_range, pos_range)
            car_dist_f = cur_dist_f
            car_heading = cur_heading + np.random.uniform(-heading_range, heading_range)
            w = World(hallWidths, hallLengths, turns,\
                    car_dist_s, car_dist_f, car_heading, car_V,\
                    episode_length, time_step, lidar_field_of_view,\
                    lidar_num_rays, lidar_noise, missing_lidar_rays, state_feedback=state_feedback)
            obs = w.scan_lidar()
            label = 1
            #label1+=1
            #train_data.append(normalize(obs))
            #train_labels.append(label)
            right.writerow(normalize(obs))
            right_test.writerow(normalize(obs))
            right_test_position.writerow([w.car_global_x, w.car_global_y, w.car_global_heading, label])
        cur_dist_f -= offset

    cur_dist_f = width/2

    while cur_dist_s <= width:
        for i in range(iter_batch):
            car_dist_s = cur_dist_s 
            car_dist_f = cur_dist_f + np.random.uniform(-pos_range, pos_range)
            car_heading = cur_heading + np.random.uniform(-heading_range, heading_range)
            w = World(hallWidths, hallLengths, turns,\
                    car_dist_s, car_dist_f, car_heading, car_V,\
                    episode_length, time_step, lidar_field_of_view,\
                    lidar_num_rays, lidar_noise, missing_lidar_rays, state_feedback=state_feedback)
            obs = w.scan_lidar()
            label = 1
            #label1+=1
            #train_data.append(normalize(obs))
            #train_labels.append(label)
            right.writerow(normalize(obs))
            right_test.writerow(normalize(obs))
            right_test_position.writerow([w.car_global_x, w.car_global_y, w.car_global_heading, label])
        cur_dist_s += offset

    cur_heading = -np.pi/2
    cur_dist_s = width/2

    while cur_dist_s <= width + diff2:
        for i in range(iter_batch):
            car_dist_s = cur_dist_s 
            car_dist_f = cur_dist_f + np.random.uniform(-pos_range, pos_range)
            car_heading = cur_heading + np.random.uniform(-heading_range, heading_range)
            w = World(hallWidths, hallLengths, turns,\
                    car_dist_s, car_dist_f, car_heading, car_V,\
                    episode_length, time_step, lidar_field_of_view,\
                    lidar_num_rays, lidar_noise, missing_lidar_rays, state_feedback=state_feedback)
            obs = w.scan_lidar()
            label = 1
            #label1+=1
            #train_data.append(normalize(obs))
            #train_labels.append(label)
            right.writerow(normalize(obs))
            right_test.writerow(normalize(obs))
            right_test_position.writerow([w.car_global_x, w.car_global_y, w.car_global_heading, label])
        cur_dist_s += offset

    while cur_dist_s <= 4:
        for i in range(iter_batch):
            car_dist_s = cur_dist_s 
            car_dist_f = cur_dist_f + np.random.uniform(-pos_range, pos_range)
            car_heading = cur_heading + np.random.uniform(-heading_range, heading_range)
            w = World(hallWidths, hallLengths, turns,\
                    car_dist_s, car_dist_f, car_heading, car_V,\
                    episode_length, time_step, lidar_field_of_view,\
                    lidar_num_rays, lidar_noise, missing_lidar_rays, state_feedback=state_feedback)
            obs = w.scan_lidar()
            label = 0
            #label0+=1
            #train_data.append(normalize(obs))
            #train_labels.append(label)
            straight.writerow(normalize(obs))
            right_test.writerow(normalize(obs))
            right_test_position.writerow([w.car_global_x, w.car_global_y, w.car_global_heading, label])
        cur_dist_s += offset

#For left turn

    (hallWidths, hallLengths, turns) = square_hall_left(width)
    cur_dist_f = 5 - diff - offset
    cur_heading = 0
    cur_dist_s = width/2
    while cur_dist_f >= width:
        for i in range(iter_batch):
            car_dist_s = cur_dist_s + np.random.uniform(-pos_range, pos_range)
            car_dist_f = cur_dist_f
            car_heading = cur_heading + np.random.uniform(-heading_range, heading_range)
            w = World(hallWidths, hallLengths, turns,\
                    car_dist_s, car_dist_f, car_heading, car_V,\
                    episode_length, time_step, lidar_field_of_view,\
                    lidar_num_rays, lidar_noise, missing_lidar_rays, state_feedback=state_feedback)
            obs = w.scan_lidar()
            label = 2
            #label2+=1
            #train_data.append(normalize(obs))
            #train_labels.append(label)
            left.writerow(normalize(obs))
            left_test.writerow(normalize(obs))
            left_test_position.writerow([w.car_global_x, w.car_global_y, w.car_global_heading, label])
        cur_dist_f -= offset
    cur_heading = np.pi/4

    while cur_dist_f >= width/2 - pos_range:
        for i in range(iter_batch):
            car_dist_s = cur_dist_s + np.random.uniform(-pos_range, pos_range)
            car_dist_f = cur_dist_f
            car_heading = cur_heading + np.random.uniform(-heading_range, heading_range)
            w = World(hallWidths, hallLengths, turns,\
                    car_dist_s, car_dist_f, car_heading, car_V,\
                    episode_length, time_step, lidar_field_of_view,\
                    lidar_num_rays, lidar_noise, missing_lidar_rays, state_feedback=state_feedback)
            obs = w.scan_lidar()
            label = 2
            #label2+=1
            #train_data.append(normalize(obs))
            #train_labels.append(label)
            left.writerow(normalize(obs))
            left_test.writerow(normalize(obs))
            left_test_position.writerow([w.car_global_x, w.car_global_y, w.car_global_heading, label])
        cur_dist_f -= offset

    cur_dist_f = width/2

    while cur_dist_s <= width:
        for i in range(iter_batch):
            car_dist_s = cur_dist_s 
            car_dist_f = cur_dist_f + np.random.uniform(-pos_range, pos_range)
            car_heading = cur_heading + np.random.uniform(-heading_range, heading_range)
            w = World(hallWidths, hallLengths, turns,\
                    car_dist_s, car_dist_f, car_heading, car_V,\
                    episode_length, time_step, lidar_field_of_view,\
                    lidar_num_rays, lidar_noise, missing_lidar_rays, state_feedback=state_feedback)
            obs = w.scan_lidar()
            label = 2
            #label2+=1
            #train_data.append(normalize(obs))
            #train_labels.append(label)
            left.writerow(normalize(obs))
            left_test.writerow(normalize(obs))
            left_test_position.writerow([w.car_global_x, w.car_global_y, w.car_global_heading, label])
        cur_dist_s += offset

    cur_heading = np.pi/2

    cur_dist_s = width/2

    while cur_dist_s <= width + diff2:
        for i in range(iter_batch):
            car_dist_s = cur_dist_s 
            car_dist_f = cur_dist_f + np.random.uniform(-pos_range, pos_range)
            car_heading = cur_heading + np.random.uniform(-heading_range, heading_range)
            w = World(hallWidths, hallLengths, turns,\
                    car_dist_s, car_dist_f, car_heading, car_V,\
                    episode_length, time_step, lidar_field_of_view,\
                    lidar_num_rays, lidar_noise, missing_lidar_rays, state_feedback=state_feedback)
            obs = w.scan_lidar()
            label = 2
            #label2 +=1
            #train_data.append(normalize(obs))
            #train_labels.append(label)
            left.writerow(normalize(obs))
            left_test.writerow(normalize(obs))
            left_test_position.writerow([w.car_global_x, w.car_global_y, w.car_global_heading, label])
        cur_dist_s += offset

    #print(label0, label1, label2)
    ## exit()
    #return train_data, train_labels    

def main(argv):
    lidar_num_rays = 21
    train_data, train_labels = generate_data(400)
    train_data = np.array(train_data)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation="tanh", input_shape=(lidar_num_rays,)),
        tf.keras.layers.Dense(64, activation="tanh"),
        tf.keras.layers.Dense(3, activation="softmax")
        ])

    model.compile(
        optimizer='adam',  # Optimizer
        # Loss function to minimize
        loss='sparse_categorical_crossentropy',
        # List of metrics to monitor
        metrics=['accuracy'],
    )
    
    train_data, train_labels = shuffle(train_data, train_labels)
    train_len = int(len(train_labels))
    model.fit(train_data[:train_len], train_labels[:train_len], epochs=15)
    # model.evaluate(train_data[train_len:], train_labels[train_len:])
    model.save("modepredictor.h5")

if __name__ == '__main__':
    #main(sys.argv[1:])
    generate_data(400)
