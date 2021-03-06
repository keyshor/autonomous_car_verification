from Car import World
from Car import complex_track
import numpy as np
import sys
import matplotlib.pyplot as plt
import yaml

def main():

    (hallWidths, hallLengths, turns) = complex_track(1.5)
    
    car_dist_s = 0.65
    car_dist_f = 2
    car_V = 2.4
    car_heading = -np.pi/4
    episode_length = 80
    time_step = 0.1
    time = 0

    state_feedback = False

    lidar_field_of_view = 135
    lidar_num_rays = 1081

    lidar_noise = 0
    missing_lidar_rays = 0
    
    w = World(hallWidths, hallLengths, turns,\
              car_dist_s, car_dist_f, car_heading, car_V,\
              episode_length, time_step, lidar_field_of_view,\
              lidar_num_rays, lidar_noise, missing_lidar_rays, state_feedback=state_feedback)
        
    w.plot_lidar(False, True, 'lidar_scan.png')
    
if __name__ == '__main__':
    main()
