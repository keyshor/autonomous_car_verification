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

STRAIGHT_BEGIN_LABEL = 0
SQUARE_RIGHT_BEGIN_LABEL = 1
SQUARE_LEFT_BEGIN_LABEL = 2
SHARP_RIGHT_BEGIN_LABEL = 3
SHARP_LEFT_BEGIN_LABEL = 4
STRAIGHT_INTERIOR_LABEL = 5
SQUARE_RIGHT_INTERIOR_LABEL = 6
SQUARE_LEFT_INTERIOR_LABEL = 7
SHARP_RIGHT_INTERIOR_LABEL = 8
SHARP_LEFT_INTERIOR_LABEL = 9

def left_normalize_heading(theta):
    return theta if theta >= 0 else theta + 2 * np.pi

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

    straight_begin_file = open('straight_begin.csv', mode='x', newline='')
    right90_begin_file = open('right90_begin.csv', mode='x', newline='')
    left90_begin_file = open('left90_begin.csv', mode='x', newline='')
    right120_begin_file = open('right120_begin.csv', mode='x', newline='')
    left120_begin_file = open('left120_begin.csv', mode='x', newline='')
    straight_interior_file = open('straight_interior.csv', mode='x', newline='')
    right90_interior_file = open('right90_interior.csv', mode='x', newline='')
    left90_interior_file = open('left90_interior.csv', mode='x', newline='')
    right120_interior_file = open('right120_interior.csv', mode='x', newline='')
    left120_interior_file = open('left120_interior.csv', mode='x', newline='')
    right90_test_file = open('right90_test.csv', mode='x', newline='')
    left90_test_file = open('left90_test.csv', mode='x', newline='')
    right120_test_file = open('right120_test.csv', mode='x', newline='')
    left120_test_file = open('left120_test.csv', mode='x', newline='')
    right90_test_position_file = open('right90_test_position.csv', mode='x', newline='')
    left90_test_position_file = open('left90_test_position.csv', mode='x', newline='')
    right120_test_position_file = open('right120_test_position.csv', mode='x', newline='')
    left120_test_position_file = open('left120_test_position.csv', mode='x', newline='')
    straight_begin = csv.writer(straight_begin_file)
    right90_begin = csv.writer(right90_begin_file)
    left90_begin = csv.writer(left90_begin_file)
    right120_begin = csv.writer(right120_begin_file)
    left120_begin = csv.writer(left120_begin_file)
    straight_interior = csv.writer(straight_interior_file)
    right90_interior = csv.writer(right90_interior_file)
    left90_interior = csv.writer(left90_interior_file)
    right120_interior = csv.writer(right120_interior_file)
    left120_interior = csv.writer(left120_interior_file)
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

    while cur_dist_f >= 5 - diff2:
        for i in range(iter_batch):
            w.set_state_local(
                    cur_dist_s + np.random.uniform(-pos_range, pos_range),
                    cur_dist_f,
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            straight_interior.writerow(obs)

            w.set_state_local(
                    cur_dist_s + np.random.uniform(-pos_range, pos_range),
                    cur_dist_f,
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            right90_test.writerow(obs)
            right90_test_position.writerow([w.car_global_x, w.car_global_y, w.car_global_heading, STRAIGHT_INTERIOR_LABEL])
        cur_dist_f -= offset

    while cur_dist_f >= 5 - diff:
        for i in range(iter_batch):
            w.set_state_local(
                    cur_dist_s + np.random.uniform(-pos_range, pos_range),
                    cur_dist_f,
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            right90_begin.writerow(obs)

            w.set_state_local(
                    cur_dist_s + np.random.uniform(-pos_range, pos_range),
                    cur_dist_f,
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            right90_test.writerow(obs)
            right90_test_position.writerow([w.car_global_x, w.car_global_y, w.car_global_heading, SQUARE_RIGHT_BEGIN_LABEL])
        cur_dist_f -= offset

    while cur_dist_f >= width:
        for i in range(iter_batch):
            w.set_state_local(
                    cur_dist_s + np.random.uniform(-pos_range, pos_range),
                    cur_dist_f,
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            right90_interior.writerow(obs)

            w.set_state_local(
                    cur_dist_s + np.random.uniform(-pos_range, pos_range),
                    cur_dist_f,
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            right90_test.writerow(obs)
            right90_test_position.writerow([w.car_global_x, w.car_global_y, w.car_global_heading, SQUARE_RIGHT_INTERIOR_LABEL])
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
            right90_interior.writerow(obs)

            w.set_state_local(
                    cur_dist_s + np.random.uniform(-pos_range, pos_range),
                    cur_dist_f,
                    cur_heading + np.random.uniform(-middle_square_heading_range, middle_square_heading_range)
                    )
            obs = normalize(w.scan_lidar())
            right90_test.writerow(obs)
            right90_test_position.writerow([w.car_global_x, w.car_global_y, w.car_global_heading, SQUARE_RIGHT_INTERIOR_LABEL])
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
            right90_interior.writerow(obs)

            w.set_state_local(
                    cur_dist_s,
                    cur_dist_f + np.random.uniform(-pos_range, pos_range),
                    cur_heading + np.random.uniform(-middle_square_heading_range, middle_square_heading_range)
                    )
            obs = normalize(w.scan_lidar())
            right90_test.writerow(obs)
            right90_test_position.writerow([w.car_global_x, w.car_global_y, w.car_global_heading, SQUARE_RIGHT_INTERIOR_LABEL])
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
            right90_interior.writerow(obs)

            w.set_state_local(
                    cur_dist_s,
                    cur_dist_f + np.random.uniform(-pos_range, pos_range),
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            right90_test.writerow(obs)
            right90_test_position.writerow([w.car_global_x, w.car_global_y, w.car_global_heading, SQUARE_RIGHT_INTERIOR_LABEL])
        cur_dist_s += offset

    while cur_dist_s <= width + diff:
        for i in range(iter_batch):
            w.set_state_local(
                    cur_dist_s,
                    cur_dist_f + np.random.uniform(-pos_range, pos_range),
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            straight_begin.writerow(obs)

            w.set_state_local(
                    cur_dist_s,
                    cur_dist_f + np.random.uniform(-pos_range, pos_range),
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            right90_test.writerow(obs)
            right90_test_position.writerow([w.car_global_x, w.car_global_y, w.car_global_heading, STRAIGHT_BEGIN_LABEL])
        cur_dist_s += offset

    while cur_dist_s <= 4:
        for i in range(iter_batch):
            w.set_state_local(
                    cur_dist_s,
                    cur_dist_f + np.random.uniform(-pos_range, pos_range),
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            straight_interior.writerow(obs)

            w.set_state_local(
                    cur_dist_s,
                    cur_dist_f + np.random.uniform(-pos_range, pos_range),
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            right90_test.writerow(obs)
            right90_test_position.writerow([w.car_global_x, w.car_global_y, w.car_global_heading, STRAIGHT_INTERIOR_LABEL])
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

    while cur_dist_f >= 5 - diff2:
        for i in range(iter_batch):
            w.set_state_local(
                    cur_dist_s + np.random.uniform(-pos_range, pos_range),
                    cur_dist_f,
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            straight_interior.writerow(obs)

            w.set_state_local(
                    cur_dist_s + np.random.uniform(-pos_range, pos_range),
                    cur_dist_f,
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            left90_test.writerow(obs)
            left90_test_position.writerow([w.car_global_x, w.car_global_y, left_normalize_heading(w.car_global_heading), STRAIGHT_INTERIOR_LABEL])
        cur_dist_f -= offset

    while cur_dist_f >= 5 - diff:
        for i in range(iter_batch):
            w.set_state_local(
                    cur_dist_s + np.random.uniform(-pos_range, pos_range),
                    cur_dist_f,
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            left90_begin.writerow(obs)

            w.set_state_local(
                    cur_dist_s + np.random.uniform(-pos_range, pos_range),
                    cur_dist_f,
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            left90_test.writerow(obs)
            left90_test_position.writerow([w.car_global_x, w.car_global_y, left_normalize_heading(w.car_global_heading), SQUARE_LEFT_BEGIN_LABEL])
        cur_dist_f -= offset

    while cur_dist_f >= width:
        for i in range(iter_batch):
            w.set_state_local(
                    cur_dist_s + np.random.uniform(-pos_range, pos_range),
                    cur_dist_f,
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            left90_interior.writerow(obs)

            w.set_state_local(
                    cur_dist_s + np.random.uniform(-pos_range, pos_range),
                    cur_dist_f,
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            left90_test.writerow(obs)
            left90_test_position.writerow([w.car_global_x, w.car_global_y, left_normalize_heading(w.car_global_heading), SQUARE_LEFT_INTERIOR_LABEL])
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
            left90_interior.writerow(obs)

            w.set_state_local(
                    cur_dist_s + np.random.uniform(-pos_range, pos_range),
                    cur_dist_f,
                    cur_heading + np.random.uniform(-middle_square_heading_range, middle_square_heading_range)
                    )
            obs = normalize(w.scan_lidar())
            left90_test.writerow(obs)
            left90_test_position.writerow([w.car_global_x, w.car_global_y, left_normalize_heading(w.car_global_heading), SQUARE_LEFT_INTERIOR_LABEL])
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
            left90_interior.writerow(obs)

            w.set_state_local(
                    width - cur_dist_s,
                    cur_dist_f + np.random.uniform(-pos_range, pos_range),
                    cur_heading + np.random.uniform(-middle_square_heading_range, middle_square_heading_range)
                    )
            obs = normalize(w.scan_lidar())
            left90_test.writerow(obs)
            left90_test_position.writerow([w.car_global_x, w.car_global_y, left_normalize_heading(w.car_global_heading), SQUARE_LEFT_INTERIOR_LABEL])
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
            left90_interior.writerow(obs)

            w.set_state_local(
                    width - cur_dist_s,
                    cur_dist_f + np.random.uniform(-pos_range, pos_range),
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            left90_test.writerow(obs)
            left90_test_position.writerow([w.car_global_x, w.car_global_y, left_normalize_heading(w.car_global_heading), SQUARE_LEFT_INTERIOR_LABEL])
        cur_dist_s += offset

    while cur_dist_s <= width + diff:
        for i in range(iter_batch):
            w.set_state_local(
                    width - cur_dist_s,
                    cur_dist_f + np.random.uniform(-pos_range, pos_range),
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            straight_begin.writerow(obs)

            w.set_state_local(
                    width - cur_dist_s,
                    cur_dist_f + np.random.uniform(-pos_range, pos_range),
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            left90_test.writerow(obs)
            left90_test_position.writerow([w.car_global_x, w.car_global_y, left_normalize_heading(w.car_global_heading), STRAIGHT_BEGIN_LABEL])
        cur_dist_s += offset

    while cur_dist_s <= 4:
        for i in range(iter_batch):
            w.set_state_local(
                    width - cur_dist_s,
                    cur_dist_f + np.random.uniform(-pos_range, pos_range),
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            straight_interior.writerow(obs)

            w.set_state_local(
                    width - cur_dist_s,
                    cur_dist_f + np.random.uniform(-pos_range, pos_range),
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            left90_test.writerow(obs)
            left90_test_position.writerow([w.car_global_x, w.car_global_y, left_normalize_heading(w.car_global_heading), STRAIGHT_INTERIOR_LABEL])
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

    while cur_dist_f >= 5 - diff2:
        for i in range(iter_batch):
            w.set_state_local(
                    cur_dist_s + np.random.uniform(-pos_range, pos_range),
                    cur_dist_f,
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            straight_interior.writerow(obs)

            w.set_state_local(
                    cur_dist_s + np.random.uniform(-pos_range, pos_range),
                    cur_dist_f,
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            right120_test.writerow(obs)
            right120_test_position.writerow([w.car_global_x, w.car_global_y, w.car_global_heading, STRAIGHT_INTERIOR_LABEL])
        cur_dist_f -= offset

    while cur_dist_f >= 5 - diff:
        for i in range(iter_batch):
            w.set_state_local(
                    cur_dist_s + np.random.uniform(-pos_range, pos_range),
                    cur_dist_f,
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            right120_begin.writerow(obs)

            w.set_state_local(
                    cur_dist_s + np.random.uniform(-pos_range, pos_range),
                    cur_dist_f,
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            right120_test.writerow(obs)
            right120_test_position.writerow([w.car_global_x, w.car_global_y, w.car_global_heading, SHARP_RIGHT_BEGIN_LABEL])
        cur_dist_f -= offset

    while cur_dist_f >= width:
        for i in range(iter_batch):
            w.set_state_local(
                    cur_dist_s + np.random.uniform(-pos_range, pos_range),
                    cur_dist_f,
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            right120_interior.writerow(obs)

            w.set_state_local(
                    cur_dist_s + np.random.uniform(-pos_range, pos_range),
                    cur_dist_f,
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            right120_test.writerow(obs)
            right120_test_position.writerow([w.car_global_x, w.car_global_y, w.car_global_heading, SHARP_RIGHT_INTERIOR_LABEL])
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
            right120_interior.writerow(obs)

            w.set_state_local(
                    cur_dist_s + np.random.uniform(-pos_range, pos_range),
                    cur_dist_f,
                    cur_heading + np.random.uniform(-middle_sharp_heading_range, middle_sharp_heading_range)
                    )
            obs = normalize(w.scan_lidar())
            right120_test.writerow(obs)
            right120_test_position.writerow([w.car_global_x, w.car_global_y, w.car_global_heading, SHARP_RIGHT_INTERIOR_LABEL])
        cur_dist_f -= offset

    cur_dist_f = width/2

    while cur_dist_s <= width:
        for i in range(iter_batch):
            right_random120(w, cur_dist_s, cur_dist_f, cur_heading, pos_range, middle_sharp_heading_range)
            obs = normalize(w.scan_lidar())
            right120_interior.writerow(obs)

            right_random120(w, cur_dist_s, cur_dist_f, cur_heading, pos_range, middle_sharp_heading_range)
            obs = normalize(w.scan_lidar())
            right120_test.writerow(obs)
            right120_test_position.writerow([w.car_global_x, w.car_global_y, w.car_global_heading, SHARP_RIGHT_INTERIOR_LABEL])
        cur_dist_s += offset

    cur_heading = -2 * np.pi/3
    cur_dist_s = width/2

    while cur_dist_s <= width + diff2:
        for i in range(iter_batch):
            right_random120(w, cur_dist_s, cur_dist_f, cur_heading, pos_range, middle_sharp_heading_range)
            obs = normalize(w.scan_lidar())
            right120_interior.writerow(obs)

            right_random120(w, cur_dist_s, cur_dist_f, cur_heading, pos_range, middle_sharp_heading_range)
            obs = normalize(w.scan_lidar())
            right120_test.writerow(obs)
            right120_test_position.writerow([w.car_global_x, w.car_global_y, w.car_global_heading, SHARP_RIGHT_INTERIOR_LABEL])
        cur_dist_s += offset

    while cur_dist_s <= width + diff:
        for i in range(iter_batch):
            right_random120(w, cur_dist_s, cur_dist_f, cur_heading, pos_range, middle_sharp_heading_range)
            obs = normalize(w.scan_lidar())
            straight_begin.writerow(obs)

            right_random120(w, cur_dist_s, cur_dist_f, cur_heading, pos_range, middle_sharp_heading_range)
            obs = normalize(w.scan_lidar())
            right120_test.writerow(obs)
            right120_test_position.writerow([w.car_global_x, w.car_global_y, w.car_global_heading, STRAIGHT_BEGIN_LABEL])
        cur_dist_s += offset

    while cur_dist_s <= 4:
        for i in range(iter_batch):
            right_random120(w, cur_dist_s, cur_dist_f, cur_heading, pos_range, middle_sharp_heading_range)
            obs = normalize(w.scan_lidar())
            straight_interior.writerow(obs)

            right_random120(w, cur_dist_s, cur_dist_f, cur_heading, pos_range, middle_sharp_heading_range)
            obs = normalize(w.scan_lidar())
            right120_test.writerow(obs)
            right120_test_position.writerow([w.car_global_x, w.car_global_y, w.car_global_heading, STRAIGHT_INTERIOR_LABEL])
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

    while cur_dist_f >= 5 - diff2:
        for i in range(iter_batch):
            w.set_state_local(
                    cur_dist_s + np.random.uniform(-pos_range, pos_range),
                    cur_dist_f,
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            straight_interior.writerow(obs)

            w.set_state_local(
                    cur_dist_s + np.random.uniform(-pos_range, pos_range),
                    cur_dist_f,
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            left120_test.writerow(obs)
            left120_test_position.writerow([w.car_global_x, w.car_global_y, left_normalize_heading(w.car_global_heading), STRAIGHT_INTERIOR_LABEL])
        cur_dist_f -= offset

    while cur_dist_f >= 5 - diff:
        for i in range(iter_batch):
            w.set_state_local(
                    cur_dist_s + np.random.uniform(-pos_range, pos_range),
                    cur_dist_f,
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            left120_begin.writerow(obs)

            w.set_state_local(
                    cur_dist_s + np.random.uniform(-pos_range, pos_range),
                    cur_dist_f,
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            left120_test.writerow(obs)
            left120_test_position.writerow([w.car_global_x, w.car_global_y, left_normalize_heading(w.car_global_heading), SHARP_LEFT_BEGIN_LABEL])
        cur_dist_f -= offset

    while cur_dist_f >= width:
        for i in range(iter_batch):
            w.set_state_local(
                    cur_dist_s + np.random.uniform(-pos_range, pos_range),
                    cur_dist_f,
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            left120_interior.writerow(obs)

            w.set_state_local(
                    cur_dist_s + np.random.uniform(-pos_range, pos_range),
                    cur_dist_f,
                    cur_heading + np.random.uniform(-heading_range, heading_range)
                    )
            obs = normalize(w.scan_lidar())
            left120_test.writerow(obs)
            left120_test_position.writerow([w.car_global_x, w.car_global_y, left_normalize_heading(w.car_global_heading), SHARP_LEFT_INTERIOR_LABEL])
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
            left120_interior.writerow(obs)

            w.set_state_local(
                    cur_dist_s + np.random.uniform(-pos_range, pos_range),
                    cur_dist_f,
                    cur_heading + np.random.uniform(-middle_sharp_heading_range, middle_sharp_heading_range)
                    )
            obs = normalize(w.scan_lidar())
            left120_test.writerow(obs)
            left120_test_position.writerow([w.car_global_x, w.car_global_y, left_normalize_heading(w.car_global_heading), SHARP_LEFT_INTERIOR_LABEL])
        cur_dist_f -= offset

    cur_dist_f = width/2

    while cur_dist_s <= width:
        for i in range(iter_batch):
            left_random120(w, cur_dist_s, cur_dist_f, cur_heading, pos_range, middle_sharp_heading_range)
            obs = normalize(w.scan_lidar())
            left120_interior.writerow(obs)

            left_random120(w, cur_dist_s, cur_dist_f, cur_heading, pos_range, middle_sharp_heading_range)
            obs = normalize(w.scan_lidar())
            left120_test.writerow(obs)
            left120_test_position.writerow([w.car_global_x, w.car_global_y, left_normalize_heading(w.car_global_heading), SHARP_LEFT_INTERIOR_LABEL])
        cur_dist_s += offset

    cur_heading = 2 * np.pi/3
    cur_dist_s = width/2

    while cur_dist_s <= width + diff2:
        for i in range(iter_batch):
            left_random120(w, cur_dist_s, cur_dist_f, cur_heading, pos_range, middle_sharp_heading_range)
            obs = normalize(w.scan_lidar())
            left120_interior.writerow(obs)

            left_random120(w, cur_dist_s, cur_dist_f, cur_heading, pos_range, middle_sharp_heading_range)
            obs = normalize(w.scan_lidar())
            left120_test.writerow(obs)
            left120_test_position.writerow([w.car_global_x, w.car_global_y, left_normalize_heading(w.car_global_heading), SHARP_LEFT_INTERIOR_LABEL])
        cur_dist_s += offset

    while cur_dist_s <= width + diff:
        for i in range(iter_batch):
            left_random120(w, cur_dist_s, cur_dist_f, cur_heading, pos_range, middle_sharp_heading_range)
            obs = normalize(w.scan_lidar())
            straight_begin.writerow(obs)

            left_random120(w, cur_dist_s, cur_dist_f, cur_heading, pos_range, middle_sharp_heading_range)
            obs = normalize(w.scan_lidar())
            left120_test.writerow(obs)
            left120_test_position.writerow([w.car_global_x, w.car_global_y, left_normalize_heading(w.car_global_heading), STRAIGHT_BEGIN_LABEL])
        cur_dist_s += offset

    while cur_dist_s <= 4:
        for i in range(iter_batch):
            left_random120(w, cur_dist_s, cur_dist_f, cur_heading, pos_range, middle_sharp_heading_range)
            obs = normalize(w.scan_lidar())
            straight_interior.writerow(obs)

            left_random120(w, cur_dist_s, cur_dist_f, cur_heading, pos_range, middle_sharp_heading_range)
            obs = normalize(w.scan_lidar())
            left120_test.writerow(obs)
            left120_test_position.writerow([w.car_global_x, w.car_global_y, left_normalize_heading(w.car_global_heading), STRAIGHT_INTERIOR_LABEL])
        cur_dist_s += offset

if __name__ == '__main__':
    generate_data(400)
