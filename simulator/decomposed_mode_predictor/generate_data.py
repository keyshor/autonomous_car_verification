import sys
import csv
import numpy as np
sys.path.append('..')
from Car import \
        World, \
        square_hall_right, \
        square_hall_left, \
        trapezoid_hall_sharp_right, \
        triangle_hall_sharp_right, \
        triangle_hall_equilateral_right

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
pos_range = 0.2
goal_heading_range = 0.02
goal_pos_range = 0.01
offset = 0.05
iter_batch = 200
goal_iter_batch = 50

LIDAR_MEAN = 2.5
LIDAR_SPREAD = 5

BIG_LABEL_STRAIGHT = 0
BIG_LABEL_RIGHT90 = 1
BIG_LABEL_LEFT90 = 2

def normalize(scan):
    return (scan - LIDAR_MEAN) / LIDAR_SPREAD

def generate_data():
    with open('straight_little_notgoal.csv', mode='x', newline='') as straight_little_notgoal_file, \
            open('straight_little_goal.csv', mode='x', newline='') as straight_little_goal_file, \
            open('right_little_notgoal.csv', mode='x', newline='') as right_little_notgoal_file, \
            open('right_little_goal.csv', mode='x', newline='') as right_little_goal_file, \
            open('left_little_notgoal.csv', mode='x', newline='') as left_little_notgoal_file, \
            open('left_little_goal.csv', mode='x', newline='') as left_little_goal_file, \
            open('big_straight.csv', mode='x', newline='') as big_straight_file, \
            open('big_right.csv', mode='x', newline='') as big_right_file, \
            open('big_left.csv', mode='x', newline='') as big_left_file:

        straight_little_notgoal = csv.writer(straight_little_notgoal_file)
        straight_little_goal = csv.writer(straight_little_goal_file)
        right_little_notgoal = csv.writer(right_little_notgoal_file)
        right_little_goal = csv.writer(right_little_goal_file)
        left_little_notgoal = csv.writer(left_little_notgoal_file)
        left_little_goal = csv.writer(left_little_goal_file)
        big_straight = csv.writer(big_straight_file)
        big_right = csv.writer(big_right_file)
        big_left = csv.writer(big_left_file)

        # right turn
        hallWidths, hallLengths, turns = square_hall_right(width)
        cur_dist_s = width/2
        cur_dist_f = 12
        cur_heading = 0
        w = World(hallWidths, hallLengths, turns,
                cur_dist_s, cur_dist_f, cur_heading, car_V,
                episode_length, time_step, lidar_field_of_view,
                lidar_num_rays, lidar_noise, missing_lidar_rays, state_feedback=state_feedback)

        while cur_dist_f >= 5 + width - 1.5:
            for _ in range(iter_batch):
                w.set_state_local(
                        cur_dist_s + np.random.uniform(-pos_range, pos_range),
                        cur_dist_f,
                        cur_heading + np.random.uniform(-heading_range, heading_range)
                        )
                obs = normalize(w.scan_lidar())
                straight_little_notgoal.writerow(obs)
                big_straight.writerow(obs)
            cur_dist_f -= offset
        while cur_dist_f >= 5 + width - 2:
            for _ in range(iter_batch):
                w.set_state_local(
                        cur_dist_s + np.random.uniform(-pos_range, pos_range),
                        cur_dist_f,
                        cur_heading + np.random.uniform(-heading_range, heading_range)
                        )
                obs = normalize(w.scan_lidar())
                if abs(w.car_dist_s - cur_dist_s) < 0.01 and abs(w.car_heading - cur_heading) < 0.02:
                    straight_little_goal.writerow(obs)
                else:
                    straight_little_notgoal.writerow(obs)
                right_little_notgoal.writerow(obs)
                big_right.writerow(obs)
            for _ in range(goal_iter_batch):
                w.set_state_local(
                        cur_dist_s + np.random.uniform(-goal_pos_range, goal_pos_range),
                        cur_dist_f,
                        cur_heading + np.random.uniform(-goal_heading_range, goal_heading_range)
                        )
                obs = normalize(w.scan_lidar())
                straight_little_goal.writerow(obs)
                right_little_notgoal.writerow(obs)
                big_right.writerow(obs)
            cur_dist_f -= offset
        while cur_dist_f >= width:
            for _ in range(iter_batch):
                w.set_state_local(
                        cur_dist_s + np.random.uniform(-pos_range, pos_range),
                        cur_dist_f,
                        cur_heading + np.random.uniform(-heading_range, heading_range)
                        )
                obs = normalize(w.scan_lidar())
                right_little_notgoal.writerow(obs)
            cur_dist_f -= offset
        cur_heading = -np.pi/4
        while cur_dist_f >= width/2 - pos_range:
            for _ in range(iter_batch):
                w.set_state_local(
                        cur_dist_s + np.random.uniform(-pos_range, pos_range),
                        cur_dist_f,
                        cur_heading + np.random.uniform(-heading_range, heading_range)
                        )
                obs = normalize(w.scan_lidar())
                right_little_notgoal.writerow(obs)
            cur_dist_f -= offset

        cur_dist_f = width/2
        while cur_dist_s <= width:
            for _ in range(iter_batch):
                w.set_state_local(
                        cur_dist_s,
                        cur_dist_f + np.random.uniform(-pos_range, pos_range),
                        cur_heading + np.random.uniform(-heading_range, heading_range)
                        )
                obs = normalize(w.scan_lidar())
                right_little_notgoal.writerow(obs)
            cur_dist_s += offset
        cur_heading = -np.pi/2
        cur_dist_s = width/2
        while cur_dist_s <= width:
            for _ in range(iter_batch):
                w.set_state_local(
                        cur_dist_s,
                        cur_dist_f + np.random.uniform(-pos_range, pos_range),
                        cur_heading + np.random.uniform(-heading_range, heading_range)
                        )
                obs = normalize(w.scan_lidar())
                right_little_notgoal.writerow(obs)
            cur_dist_s += offset
        while cur_dist_s <= width + 0.5:
            for _ in range(iter_batch):
                w.set_state_local(
                        cur_dist_s,
                        cur_dist_f + np.random.uniform(-pos_range, pos_range),
                        cur_heading + np.random.uniform(-heading_range, heading_range)
                        )
                obs = normalize(w.scan_lidar())
                if abs(w.car_dist_f - cur_dist_f) < 0.01 and abs(w.car_heading - cur_heading) < 0.02:
                    right_little_goal.writerow(obs)
                else:
                    right_little_notgoal.writerow(obs)
                straight_little_notgoal.writerow(obs)
                big_straight.writerow(obs)
            for _ in range(goal_iter_batch):
                w.set_state_local(
                        cur_dist_s,
                        cur_dist_f + np.random.uniform(-goal_pos_range, goal_pos_range),
                        cur_heading + np.random.uniform(-goal_heading_range, goal_heading_range)
                        )
                obs = normalize(w.scan_lidar())
                right_little_goal.writerow(obs)
                straight_little_notgoal.writerow(obs)
                big_straight.writerow(obs)
            cur_dist_s += offset
        while cur_dist_s <= 12:
            for _ in range(iter_batch):
                w.set_state_local(
                        cur_dist_s,
                        cur_dist_f + np.random.uniform(-pos_range, pos_range),
                        cur_heading + np.random.uniform(-heading_range, heading_range)
                        )
                obs = normalize(w.scan_lidar())
                straight_little_notgoal.writerow(obs)
            cur_dist_s += offset

        # left turn
        (hallWidths, hallLengths, turns) = square_hall_left(width)
        cur_dist_f = 5 + width - 1.5 - pos_range
        cur_heading = 0
        cur_dist_s = width/2
        w = World(hallWidths, hallLengths, turns,
                cur_dist_s, cur_dist_f, cur_heading, car_V,
                episode_length, time_step, lidar_field_of_view,
                lidar_num_rays, lidar_noise, missing_lidar_rays, state_feedback=state_feedback)
        while cur_dist_f >= 5 + width - 2:
            for _ in range(iter_batch):
                w.set_state_local(
                        cur_dist_s + np.random.uniform(-pos_range, pos_range),
                        cur_dist_f,
                        cur_heading + np.random.uniform(-heading_range, heading_range)
                        )
                obs = normalize(w.scan_lidar())
                if abs(w.car_dist_s - cur_dist_s) < 0.01 and abs(w.car_heading - cur_heading) < 0.02:
                    straight_little_goal.writerow(obs)
                else:
                    straight_little_notgoal.writerow(obs)
                left_little_notgoal.writerow(obs)
                big_left.writerow(obs)
            for _ in range(goal_iter_batch):
                w.set_state_local(
                        cur_dist_s + np.random.uniform(-goal_pos_range, goal_pos_range),
                        cur_dist_f,
                        cur_heading + np.random.uniform(-goal_heading_range, goal_heading_range)
                        )
                obs = normalize(w.scan_lidar())
                straight_little_goal.writerow(obs)
                left_little_notgoal.writerow(obs)
                big_left.writerow(obs)
            cur_dist_f -= offset
        while cur_dist_f >= width:
            for _ in range(iter_batch):
                w.set_state_local(
                        cur_dist_s + np.random.uniform(-pos_range, pos_range),
                        cur_dist_f,
                        cur_heading + np.random.uniform(-heading_range, heading_range)
                        )
                obs = normalize(w.scan_lidar())
                left_little_notgoal.writerow(obs)
            cur_dist_f -= offset
        cur_heading = np.pi/4
        while cur_dist_f >= width/2 - pos_range:
            for _ in range(iter_batch):
                w.set_state_local(
                        cur_dist_s + np.random.uniform(-pos_range, pos_range),
                        cur_dist_f,
                        cur_heading + np.random.uniform(-heading_range, heading_range)
                        )
                obs = normalize(w.scan_lidar())
                left_little_notgoal.writerow(obs)
            cur_dist_f -= offset
        cur_dist_f = width/2
        while cur_dist_s <= width:
            for _ in range(iter_batch):
                w.set_state_local(
                        cur_dist_s,
                        cur_dist_f + np.random.uniform(-pos_range, pos_range),
                        cur_heading + np.random.uniform(-heading_range, heading_range)
                        )
                obs = normalize(w.scan_lidar())
                left_little_notgoal.writerow(obs)
            cur_dist_s += offset
        cur_heading = np.pi/2
        cur_dist_s = width/2
        while cur_dist_s <= width:
            for _ in range(iter_batch):
                w.set_state_local(
                        cur_dist_s,
                        cur_dist_f + np.random.uniform(-pos_range, pos_range),
                        cur_heading + np.random.uniform(-heading_range, heading_range)
                        )
                obs = normalize(w.scan_lidar())
                left_little_notgoal.writerow(obs)
            cur_dist_s += offset
        while cur_dist_s <= width + 0.5:
            for _ in range(iter_batch):
                w.set_state_local(
                        cur_dist_s,
                        cur_dist_f + np.random.uniform(-pos_range, pos_range),
                        cur_heading + np.random.uniform(-heading_range, heading_range)
                        )
                obs = normalize(w.scan_lidar())
                if abs(w.car_dist_f - cur_dist_f) < 0.01 and abs(w.car_heading - cur_heading) < 0.02:
                    left_little_goal.writerow(obs)
                else:
                    left_little_notgoal.writerow(obs)
                straight_little_notgoal.writerow(obs)
                big_straight.writerow(obs)
            for _ in range(goal_iter_batch):
                w.set_state_local(
                        cur_dist_s,
                        cur_dist_f + np.random.uniform(-goal_pos_range, goal_pos_range),
                        cur_heading + np.random.uniform(-goal_heading_range, goal_heading_range)
                        )
                obs = normalize(w.scan_lidar())
                left_little_goal.writerow(obs)
                straight_little_notgoal.writerow(obs)
                big_straight.writerow(obs)
            cur_dist_s += offset

if __name__ == '__main__':
    generate_data()
