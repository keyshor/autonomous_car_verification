import sys
sys.path.append('..')
from Car import World
from Car import square_hall_right
from Car import square_hall_left
from Car import triangle_hall_equilateral_right
from Car import triangle_hall_equilateral_left
import numpy as np
import matplotlib.pyplot as plt

car_dist_s = 0.75
car_dist_f = 6.5
car_heading = 0
car_V = 2.4
episode_length = 100
time_step = 0.1
lidar_field_of_view = 115
lidar_num_rays = 21
lidar_noise = 0
missing_lidar_rays = 0

def square_right_data():
    hallWidths, hallLengths, turns = square_hall_right()
    return (World(hallWidths, hallLengths, turns,
        car_dist_s, car_dist_f, car_heading, car_V,
        episode_length, time_step, lidar_field_of_view,
        lidar_num_rays, lidar_noise, missing_lidar_rays, True),
        np.loadtxt('right90_test_position.csv', delimiter=','))

def sharp_right_data():
    hallWidths, hallLengths, turns = triangle_hall_equilateral_right()
    return (World(hallWidths, hallLengths, turns,
        car_dist_s, car_dist_f, car_heading, car_V,
        episode_length, time_step, lidar_field_of_view,
        lidar_num_rays, lidar_noise, missing_lidar_rays, True),
        np.loadtxt('right120_test_position.csv', delimiter=','))

def square_left_data():
    hallWidths, hallLengths, turns = square_hall_left()
    return (World(hallWidths, hallLengths, turns,
        car_dist_s, car_dist_f, car_heading, car_V,
        episode_length, time_step, lidar_field_of_view,
        lidar_num_rays, lidar_noise, missing_lidar_rays, True),
        np.loadtxt('left90_test_position.csv', delimiter=','))

def sharp_left_data():
    hallWidths, hallLengths, turns = triangle_hall_equilateral_left()
    return (World(hallWidths, hallLengths, turns,
        car_dist_s, car_dist_f, car_heading, car_V,
        episode_length, time_step, lidar_field_of_view,
        lidar_num_rays, lidar_noise, missing_lidar_rays, True),
        np.loadtxt('left120_test_position.csv', delimiter=','))

def plot_data(world, data, xy_file, xh_file, yh_file):
    plt.clf()
    plt.figure(figsize=(12, 10))
    world.plotHalls()
    labels = data[:,3].ravel()
    for label, array, color in [
            ('straight', data[np.equal(labels, 0)], 'g'),
            ('square_right', data[np.equal(labels, 1)], 'b'),
            ('square_left', data[np.equal(labels, 2)], 'm'),
            ('sharp_right', data[np.equal(labels, 3)], 'c'),
            ('sharp_left', data[np.equal(labels, 4)], 'y')
            ]:
        plt.plot(
                array[:,0], array[:,1],
                f'{color}.', label=label,
                markersize=1
                )
    plt.legend(markerscale=10)
    plt.savefig(xy_file)
    plt.clf()

    xh_fig, xh_ax = plt.subplots()
    yh_fig, yh_ax = plt.subplots()
    for label, array, color in [
            ('straight', data[np.equal(labels, 0)], 'g'),
            ('square_right', data[np.equal(labels, 1)], 'b'),
            ('square_left', data[np.equal(labels, 2)], 'm'),
            ('sharp_right', data[np.equal(labels, 3)], 'c'),
            ('sharp_left', data[np.equal(labels, 4)], 'y')
            ]:
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
    xh_ax.legend(markerscale=10)
    yh_ax.legend(markerscale=10)
    xh_ax.set_xlabel('x coordinate')
    xh_ax.set_ylabel('heading')
    yh_ax.set_xlabel('y coordinate')
    yh_ax.set_ylabel('heading')
    xh_fig.savefig(xh_file)
    yh_fig.savefig(yh_file)

if __name__ == '__main__':
    square_right_world, square_right_points = square_right_data()
    square_left_world, square_left_points = square_left_data()
    sharp_right_world, sharp_right_points = sharp_right_data()
    sharp_left_world, sharp_left_points = sharp_left_data()
    plot_data(square_right_world, square_right_points,
            'square_right_xy.png', 'square_right_xh.png', 'square_right_yh.png')
    plot_data(square_left_world, square_left_points,
            'square_left_xy.png', 'square_left_xh.png', 'square_left_yh.png')
    plot_data(sharp_right_world, sharp_right_points,
            'sharp_right_xy.png', 'sharp_right_xh.png', 'sharp_right_yh.png')
    plot_data(sharp_left_world, sharp_left_points,
            'sharp_left_xy.png', 'sharp_left_xh.png', 'sharp_left_yh.png')
