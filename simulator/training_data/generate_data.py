import sys
import csv
sys.path.append('..')
from Car import World
from Car import square_hall_right
from Car import square_hall_left
from Car import triangle_hall_equilateral_right
from Car import triangle_hall_equilateral_left
import numpy as np

width = 1.5
car_V = 2.4
episode_length = 65
time_step = 0.1
state_feedback = False
lidar_field_of_view = 115
lidar_num_rays = 21
lidar_noise = 0
missing_lidar_rays = 0

heading_range = 0.3
pos_range = 0.3
offset = 0.025
middle_square_heading_range = np.pi / 4 - heading_range
middle_sharp_heading_range = np.pi / 3 - heading_range

def normalize(s):
    mean = [2.5]
    spread = [5.0]
    return (s - mean) / spread

def right_random120(world, dist_s, dist_f, cur_heading, pos_range, heading_range):
    sin30 = np.sin(np.radians(30))
    cos30 = np.cos(np.radians(30))
    side_disp = dist_f + np.random.uniform(-pos_range, pos_range)
    world.set_state_local(
            dist_s * cos30 - side_disp * sin30,
            dist_s * sin30 + side_disp * cos30,
            cur_heading + np.random.uniform(-heading_range, heading_range)
            )

def left_random120(world, dist_s, dist_f, cur_heading, pos_range, heading_range):
    sin30 = np.sin(np.radians(30))
    cos30 = np.cos(np.radians(30))
    side_disp = dist_f + np.random.uniform(-pos_range, pos_range)
    world.set_state_local(
            width - dist_s * cos30 + side_disp * sin30,
            dist_s * sin30 + side_disp * cos30,
            cur_heading + np.random.uniform(-heading_range, heading_range)
            )

def generate_data(iter_batch=200):
    car_dist_s = width/2
    car_dist_f = 5 + width
    car_heading = 0

    straight_file = open('straight.csv', mode='x', newline='')
    right90_file = open('right90.csv', mode='x', newline='')
    left90_file = open('left90.csv', mode='x', newline='')
    right120_file = open('right120.csv', mode='x', newline='')
    left120_file = open('left120.csv', mode='x', newline='')
    right90_test_file = open('right90_test.csv', mode='x', newline='')
    left90_test_file = open('left90_test.csv', mode='x', newline='')
    right120_test_file = open('right120_test.csv', mode='x', newline='')
    left120_test_file = open('left120_test.csv', mode='x', newline='')
    right90_test_position_file = open('right90_test_position.csv', mode='x', newline='')
    left90_test_position_file = open('left90_test_position.csv', mode='x', newline='')
    right120_test_position_file = open('right120_test_position.csv', mode='x', newline='')
    left120_test_position_file = open('left120_test_position.csv', mode='x', newline='')
    straight = csv.writer(straight_file)
    right90 = csv.writer(right90_file)
    left90 = csv.writer(left90_file)
    right120 = csv.writer(right120_file)
    left120 = csv.writer(left120_file)
    right90_test = csv.writer(right90_test_file)
    left90_test = csv.writer(left90_test_file)
    right120_test = csv.writer(right120_test_file)
    left120_test = csv.writer(left120_test_file)
    right90_test_position = csv.writer(right90_test_position_file)
    left90_test_position = csv.writer(left90_test_position_file)
    right120_test_position = csv.writer(right120_test_position_file)
    left120_test_position = csv.writer(left120_test_position_file)

    #labels: 1 - straight, 2 - right, 3 - left

    diff = 1
    diff2 = 0.5

    # square right turn
    hallWidths, hallLengths, turns = square_hall_right(width)
    cur_dist_s = width/2
    cur_dist_f = 7
    cur_heading = 0
    w = World(hallWidths, hallLengths, turns,\
            car_dist_s, car_dist_f, car_heading, car_V,\
            episode_length, time_step, lidar_field_of_view,\
            lidar_num_rays, lidar_noise, missing_lidar_rays, state_feedback=state_feedback)

    while cur_dist_f >= 5 - diff:
        for i in range(iter_batch):
            w.set_state_local(
                    cur_dist_s + np.random.uniform(-pos_range, pos_range),
                    cur_dist_f,
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            straight.writerow(obs)

            w.set_state_local(
                    cur_dist_s + np.random.uniform(-pos_range, pos_range),
                    cur_dist_f,
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            right90_test.writerow(obs)
            right90_test_position.writerow([w.car_global_x, w.car_global_y, w.car_global_heading, 0])
        cur_dist_f -= offset

    while cur_dist_f >= width:
        for i in range(iter_batch):
            w.set_state_local(
                    cur_dist_s + np.random.uniform(-pos_range, pos_range),
                    cur_dist_f,
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            right90.writerow(obs)

            w.set_state_local(
                    cur_dist_s + np.random.uniform(-pos_range, pos_range),
                    cur_dist_f,
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            right90_test.writerow(obs)
            right90_test_position.writerow([w.car_global_x, w.car_global_y, w.car_global_heading, 1])
        cur_dist_f -= offset
    cur_heading = -np.pi/4

    while cur_dist_f >= width/2 - pos_range:
        for i in range(iter_batch):
            w.set_state_local(
                    cur_dist_s + np.random.uniform(-pos_range, pos_range),
                    cur_dist_f,
                    cur_heading + np.random.uniform(-middle_square_heading_range, middle_square_heading_range)
                    )
            obs = normalize(w.scan_lidar())
            right90.writerow(obs)

            w.set_state_local(
                    cur_dist_s + np.random.uniform(-pos_range, pos_range),
                    cur_dist_f,
                    cur_heading + np.random.uniform(-middle_square_heading_range, middle_square_heading_range)
                    )
            obs = normalize(w.scan_lidar())
            right90_test.writerow(obs)
            right90_test_position.writerow([w.car_global_x, w.car_global_y, w.car_global_heading, 1])
        cur_dist_f -= offset

    cur_dist_f = width/2

    while cur_dist_s <= width:
        for i in range(iter_batch):
            w.set_state_local(
                    cur_dist_s,
                    cur_dist_f + np.random.uniform(-pos_range, pos_range),
                    cur_heading + np.random.uniform(-middle_square_heading_range, middle_square_heading_range)
                    )
            obs = normalize(w.scan_lidar())
            right90.writerow(obs)

            w.set_state_local(
                    cur_dist_s,
                    cur_dist_f + np.random.uniform(-pos_range, pos_range),
                    cur_heading + np.random.uniform(-middle_square_heading_range, middle_square_heading_range)
                    )
            obs = normalize(w.scan_lidar())
            right90_test.writerow(obs)
            right90_test_position.writerow([w.car_global_x, w.car_global_y, w.car_global_heading, 1])
        cur_dist_s += offset

    cur_heading = -np.pi/2
    cur_dist_s = width/2

    while cur_dist_s <= width + diff2:
        for i in range(iter_batch):
            w.set_state_local(
                    cur_dist_s,
                    cur_dist_f + np.random.uniform(-pos_range, pos_range),
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            right90.writerow(obs)

            w.set_state_local(
                    cur_dist_s,
                    cur_dist_f + np.random.uniform(-pos_range, pos_range),
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            right90_test.writerow(obs)
            right90_test_position.writerow([w.car_global_x, w.car_global_y, w.car_global_heading, 1])
        cur_dist_s += offset

    while cur_dist_s <= 4:
        for i in range(iter_batch):
            w.set_state_local(
                    cur_dist_s,
                    cur_dist_f + np.random.uniform(-pos_range, pos_range),
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            straight.writerow(obs)

            w.set_state_local(
                    cur_dist_s,
                    cur_dist_f + np.random.uniform(-pos_range, pos_range),
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            right90_test.writerow(obs)
            right90_test_position.writerow([w.car_global_x, w.car_global_y, w.car_global_heading, 0])
        cur_dist_s += offset

    # square left turn

    hallWidths, hallLengths, turns = square_hall_left(width)
    cur_dist_f = 7
    cur_heading = 0
    cur_dist_s = width/2
    w = World(hallWidths, hallLengths, turns,\
            car_dist_s, car_dist_f, car_heading, car_V,\
            episode_length, time_step, lidar_field_of_view,\
            lidar_num_rays, lidar_noise, missing_lidar_rays, state_feedback=state_feedback)

    while cur_dist_f >= 5 - diff:
        for i in range(iter_batch):
            w.set_state_local(
                    cur_dist_s + np.random.uniform(-pos_range, pos_range),
                    cur_dist_f,
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            straight.writerow(obs)

            w.set_state_local(
                    cur_dist_s + np.random.uniform(-pos_range, pos_range),
                    cur_dist_f,
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            left90_test.writerow(obs)
            left90_test_position.writerow([w.car_global_x, w.car_global_y, w.car_global_heading, 0])
        cur_dist_f -= offset

    while cur_dist_f >= width:
        for i in range(iter_batch):
            w.set_state_local(
                    cur_dist_s + np.random.uniform(-pos_range, pos_range),
                    cur_dist_f,
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            left90.writerow(obs)

            w.set_state_local(
                    cur_dist_s + np.random.uniform(-pos_range, pos_range),
                    cur_dist_f,
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            left90_test.writerow(obs)
            left90_test_position.writerow([w.car_global_x, w.car_global_y, w.car_global_heading, 2])
        cur_dist_f -= offset
    cur_heading = np.pi/4

    while cur_dist_f >= width/2 - pos_range:
        for i in range(iter_batch):
            w.set_state_local(
                    cur_dist_s + np.random.uniform(-pos_range, pos_range),
                    cur_dist_f,
                    cur_heading + np.random.uniform(-middle_square_heading_range, middle_square_heading_range)
                    )
            obs = normalize(w.scan_lidar())
            left90.writerow(obs)

            w.set_state_local(
                    cur_dist_s + np.random.uniform(-pos_range, pos_range),
                    cur_dist_f,
                    cur_heading + np.random.uniform(-middle_square_heading_range, middle_square_heading_range)
                    )
            obs = normalize(w.scan_lidar())
            left90_test.writerow(obs)
            left90_test_position.writerow([w.car_global_x, w.car_global_y, w.car_global_heading, 2])
        cur_dist_f -= offset

    cur_dist_f = width/2

    while cur_dist_s <= width:
        for i in range(iter_batch):
            w.set_state_local(
                    width - cur_dist_s,
                    cur_dist_f + np.random.uniform(-pos_range, pos_range),
                    cur_heading + np.random.uniform(-middle_square_heading_range, middle_square_heading_range)
                    )
            obs = normalize(w.scan_lidar())
            left90.writerow(obs)

            w.set_state_local(
                    width - cur_dist_s,
                    cur_dist_f + np.random.uniform(-pos_range, pos_range),
                    cur_heading + np.random.uniform(-middle_square_heading_range, middle_square_heading_range)
                    )
            obs = normalize(w.scan_lidar())
            left90_test.writerow(obs)
            left90_test_position.writerow([w.car_global_x, w.car_global_y, w.car_global_heading, 2])
        cur_dist_s += offset

    cur_heading = np.pi/2

    cur_dist_s = width/2

    while cur_dist_s <= width + diff2:
        for i in range(iter_batch):
            w.set_state_local(
                    width - cur_dist_s,
                    cur_dist_f + np.random.uniform(-pos_range, pos_range),
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            left90.writerow(obs)

            w.set_state_local(
                    width - cur_dist_s,
                    cur_dist_f + np.random.uniform(-pos_range, pos_range),
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            left90_test.writerow(obs)
            left90_test_position.writerow([w.car_global_x, w.car_global_y, w.car_global_heading, 2])
        cur_dist_s += offset

    while cur_dist_s <= 4:
        for i in range(iter_batch):
            w.set_state_local(
                    width - cur_dist_s,
                    cur_dist_f + np.random.uniform(-pos_range, pos_range),
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            straight.writerow(obs)

            w.set_state_local(
                    width - cur_dist_s,
                    cur_dist_f + np.random.uniform(-pos_range, pos_range),
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            left90_test.writerow(obs)
            left90_test_position.writerow([w.car_global_x, w.car_global_y, w.car_global_heading, 0])
        cur_dist_s += offset

    # sharp right turn
    hallWidths, hallLengths, turns = triangle_hall_equilateral_right(width)
    cur_dist_s = width/2
    cur_dist_f = 7
    cur_heading = 0
    w = World(hallWidths, hallLengths, turns,\
            car_dist_s, car_dist_f, car_heading, car_V,\
            episode_length, time_step, lidar_field_of_view,\
            lidar_num_rays, lidar_noise, missing_lidar_rays, state_feedback=state_feedback)

    while cur_dist_f >= 5 - diff:
        for i in range(iter_batch):
            w.set_state_local(
                    cur_dist_s + np.random.uniform(-pos_range, pos_range),
                    cur_dist_f,
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            straight.writerow(obs)

            w.set_state_local(
                    cur_dist_s + np.random.uniform(-pos_range, pos_range),
                    cur_dist_f,
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            right120_test.writerow(obs)
            right120_test_position.writerow([w.car_global_x, w.car_global_y, w.car_global_heading, 0])
        cur_dist_f -= offset

    while cur_dist_f >= width:
        for i in range(iter_batch):
            w.set_state_local(
                    cur_dist_s + np.random.uniform(-pos_range, pos_range),
                    cur_dist_f,
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            right120.writerow(obs)

            w.set_state_local(
                    cur_dist_s + np.random.uniform(-pos_range, pos_range),
                    cur_dist_f,
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            right120_test.writerow(obs)
            right120_test_position.writerow([w.car_global_x, w.car_global_y, w.car_global_heading, 3])
        cur_dist_f -= offset
    cur_heading = -np.pi/3

    while cur_dist_f >= width/2 - pos_range:
        for i in range(iter_batch):
            w.set_state_local(
                    cur_dist_s + np.random.uniform(-pos_range, pos_range),
                    cur_dist_f,
                    cur_heading + np.random.uniform(-middle_sharp_heading_range, middle_sharp_heading_range)
                    )
            obs = normalize(w.scan_lidar())
            right120.writerow(obs)

            w.set_state_local(
                    cur_dist_s + np.random.uniform(-pos_range, pos_range),
                    cur_dist_f,
                    cur_heading + np.random.uniform(-middle_sharp_heading_range, middle_sharp_heading_range)
                    )
            obs = normalize(w.scan_lidar())
            right120_test.writerow(obs)
            right120_test_position.writerow([w.car_global_x, w.car_global_y, w.car_global_heading, 3])
        cur_dist_f -= offset

    cur_dist_f = width/2

    while cur_dist_s <= width:
        for i in range(iter_batch):
            right_random120(w, cur_dist_s, cur_dist_f, cur_heading, pos_range, middle_sharp_heading_range)
            obs = normalize(w.scan_lidar())
            right120.writerow(obs)

            right_random120(w, cur_dist_s, cur_dist_f, cur_heading, pos_range, middle_sharp_heading_range)
            obs = normalize(w.scan_lidar())
            right120_test.writerow(obs)
            right120_test_position.writerow([w.car_global_x, w.car_global_y, w.car_global_heading, 3])
        cur_dist_s += offset

    cur_heading = -2 * np.pi/3
    cur_dist_s = width/2

    while cur_dist_s <= width + diff2:
        for i in range(iter_batch):
            right_random120(w, cur_dist_s, cur_dist_f, cur_heading, pos_range, middle_sharp_heading_range)
            obs = normalize(w.scan_lidar())
            right120.writerow(obs)

            right_random120(w, cur_dist_s, cur_dist_f, cur_heading, pos_range, middle_sharp_heading_range)
            obs = normalize(w.scan_lidar())
            right120_test.writerow(obs)
            right120_test_position.writerow([w.car_global_x, w.car_global_y, w.car_global_heading, 3])
        cur_dist_s += offset

    while cur_dist_s <= 4:
        for i in range(iter_batch):
            right_random120(w, cur_dist_s, cur_dist_f, cur_heading, pos_range, middle_sharp_heading_range)
            obs = normalize(w.scan_lidar())
            straight.writerow(obs)

            right_random120(w, cur_dist_s, cur_dist_f, cur_heading, pos_range, middle_sharp_heading_range)
            obs = normalize(w.scan_lidar())
            right120_test.writerow(obs)
            right120_test_position.writerow([w.car_global_x, w.car_global_y, w.car_global_heading, 0])
        cur_dist_s += offset

    # sharp left turn
    hallWidths, hallLengths, turns = triangle_hall_equilateral_left(width)
    cur_dist_s = width/2
    cur_dist_f = 7
    cur_heading = 0
    w = World(hallWidths, hallLengths, turns,\
            car_dist_s, car_dist_f, car_heading, car_V,\
            episode_length, time_step, lidar_field_of_view,\
            lidar_num_rays, lidar_noise, missing_lidar_rays, state_feedback=state_feedback)

    while cur_dist_f >= 5 - diff:
        for i in range(iter_batch):
            w.set_state_local(
                    cur_dist_s + np.random.uniform(-pos_range, pos_range),
                    cur_dist_f,
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            straight.writerow(obs)

            w.set_state_local(
                    cur_dist_s + np.random.uniform(-pos_range, pos_range),
                    cur_dist_f,
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            left120_test.writerow(obs)
            left120_test_position.writerow([w.car_global_x, w.car_global_y, w.car_global_heading, 0])
        cur_dist_f -= offset

    while cur_dist_f >= width:
        for i in range(iter_batch):
            w.set_state_local(
                    cur_dist_s + np.random.uniform(-pos_range, pos_range),
                    cur_dist_f,
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            left120.writerow(obs)

            w.set_state_local(
                    cur_dist_s + np.random.uniform(-pos_range, pos_range),
                    cur_dist_f,
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            left120_test.writerow(obs)
            left120_test_position.writerow([w.car_global_x, w.car_global_y, w.car_global_heading, 4])
        cur_dist_f -= offset
    cur_heading = np.pi/3

    while cur_dist_f >= width/2 - pos_range:
        for i in range(iter_batch):
            w.set_state_local(
                    cur_dist_s + np.random.uniform(-pos_range, pos_range),
                    cur_dist_f,
                    cur_heading + np.random.uniform(-middle_sharp_heading_range, middle_sharp_heading_range)
                    )
            obs = normalize(w.scan_lidar())
            left120.writerow(obs)

            w.set_state_local(
                    cur_dist_s + np.random.uniform(-pos_range, pos_range),
                    cur_dist_f,
                    cur_heading + np.random.uniform(-middle_sharp_heading_range, middle_sharp_heading_range)
                    )
            obs = normalize(w.scan_lidar())
            left120_test.writerow(obs)
            left120_test_position.writerow([w.car_global_x, w.car_global_y, w.car_global_heading, 4])
        cur_dist_f -= offset

    cur_dist_f = width/2

    while cur_dist_s <= width:
        for i in range(iter_batch):
            left_random120(w, cur_dist_s, cur_dist_f, cur_heading, pos_range, middle_sharp_heading_range)
            obs = normalize(w.scan_lidar())
            left120.writerow(obs)

            left_random120(w, cur_dist_s, cur_dist_f, cur_heading, pos_range, middle_sharp_heading_range)
            obs = normalize(w.scan_lidar())
            left120_test.writerow(obs)
            left120_test_position.writerow([w.car_global_x, w.car_global_y, w.car_global_heading, 4])
        cur_dist_s += offset

    cur_heading = 2 * np.pi/3
    cur_dist_s = width/2

    while cur_dist_s <= width + diff2:
        for i in range(iter_batch):
            left_random120(w, cur_dist_s, cur_dist_f, cur_heading, pos_range, middle_sharp_heading_range)
            obs = normalize(w.scan_lidar())
            left120.writerow(obs)

            left_random120(w, cur_dist_s, cur_dist_f, cur_heading, pos_range, middle_sharp_heading_range)
            obs = normalize(w.scan_lidar())
            left120_test.writerow(obs)
            left120_test_position.writerow([w.car_global_x, w.car_global_y, w.car_global_heading, 4])
        cur_dist_s += offset

    while cur_dist_s <= 4:
        for i in range(iter_batch):
            left_random120(w, cur_dist_s, cur_dist_f, cur_heading, pos_range, middle_sharp_heading_range)
            obs = normalize(w.scan_lidar())
            straight.writerow(obs)

            left_random120(w, cur_dist_s, cur_dist_f, cur_heading, pos_range, middle_sharp_heading_range)
            obs = normalize(w.scan_lidar())
            left120_test.writerow(obs)
            left120_test_position.writerow([w.car_global_x, w.car_global_y, w.car_global_heading, 0])
        cur_dist_s += offset

if __name__ == '__main__':
    generate_data(400)
