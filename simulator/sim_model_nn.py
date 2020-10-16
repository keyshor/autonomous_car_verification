from Car import World
from Car import square_hall_right
from Car import trapezoid_hall_sharp_right
from Car import triangle_hall_sharp_right
from Car import triangle_hall_equilateral_right
import numpy as np
import random
from keras import models
import sys
import matplotlib.pyplot as plt

def normalize(s):
    mean = [2.5]
    spread = [5.0]
    return (s - mean) / spread

def main(argv):

    input_filename = argv[0]
    
    model = models.load_model(input_filename)

    #(hallWidths, hallLengths, turns) = square_hall_right(2)
    #(hallWidths, hallLengths, turns) = trapezoid_hall_sharp_right(2)
    (hallWidths, hallLengths, turns) = triangle_hall_equilateral_right(2)
    
    car_dist_s = 0.65
    car_dist_f = 10
    car_V = 2.4
    car_heading = 0.02
    episode_length = 65
    time_step = 0.1
    time = 0

    state_feedback = True

    lidar_field_of_view = 115
    lidar_num_rays = 21

    lidar_noise = 0
    missing_lidar_rays = 0
    
    w = World(hallWidths, hallLengths, turns,\
              car_dist_s, car_dist_f, car_heading, car_V,\
              episode_length, time_step, lidar_field_of_view,\
              lidar_num_rays, lidar_noise, missing_lidar_rays, state_feedback=state_feedback)

    throttle = 16

    rew = 0

    observation = w.scan_lidar()

    if state_feedback:
        observation =  w.reset(side_pos = car_dist_s, pos_noise = 0, heading_noise = 0)


    prev_err = 0
    
    for e in range(episode_length):

        if not state_feedback:
            observation = normalize(observation)
            
        delta = 15 * model.predict(observation.reshape(1,len(observation)))

        delta = np.clip(delta, -15, 15)
        
        observation, reward, done, info = w.step(delta, throttle)

        time += time_step
        
        rew += reward

        if done:
            break
        
    print('velocity: ' + str(w.car_V))
    print('distance from side wall: ' + str(w.car_dist_s))
    print('distance from front wall: ' + str(w.car_dist_f))
    print('heading: ' + str(w.car_heading))
    print('last control input: ' + str(delta))
    print('steps: ' + str(e))
    print('final reward: ' + str(rew))
    w.plot_trajectory()
    #w.plot_lidar()
    
if __name__ == '__main__':
    main(sys.argv[1:])
