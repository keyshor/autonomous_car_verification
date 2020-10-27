from Car import World
from Car import square_hall_right
from Car import square_hall_left
from Car import trapezoid_hall_sharp_right
from Car import triangle_hall_sharp_right
from Car import triangle_hall_equilateral_right
from Car import triangle_hall_equilateral_left
import numpy as np
import random
from keras import models
import sys
import matplotlib.pyplot as plt
import yaml

def predict(model, inputs):
    weights = {}
    offsets = {}

    layerCount = 0
    activations = []

    for layer in range(1, len(model['weights']) + 1):
        
        weights[layer] = np.array(model['weights'][layer])
        offsets[layer] = np.array(model['offsets'][layer])

        layerCount += 1
        activations.append(model['activations'][layer])

    curNeurons = inputs

    for layer in range(layerCount):

        curNeurons = curNeurons.dot(weights[layer + 1].T) + offsets[layer + 1]

        if 'Sigmoid' in activations[layer]:
            curNeurons = sigmoid(curNeurons)
        elif 'Tanh' in activations[layer]:
            curNeurons = np.tanh(curNeurons)

    return curNeurons    

def normalize(s):
    mean = [2.5]
    spread = [5.0]
    return (s - mean) / spread

def main(argv):

    input_filename = argv[0]

    if 'yml' in input_filename:
        with open(input_filename, 'rb') as f:
            
            model = yaml.load(f)
    else:
    
        model = models.load_model(input_filename)

    #(hallWidths, hallLengths, turns) = square_hall_right(1.5)
    #(hallWidths, hallLengths, turns) = square_hall_left(1.5)
    (hallWidths, hallLengths, turns) = triangle_hall_equilateral_right(1.5)
    #(hallWidths, hallLengths, turns) = triangle_hall_equilateral_left(1.5)

    #temp = [0.5829178211060824, 8, -0.09540465144355266]
    
    car_dist_s = 0.65
    car_dist_f = 7
    car_V = 2.4
    car_heading = 0
    episode_length = 80
    time_step = 0.1
    time = 0

    state_feedback = False

    lidar_field_of_view = 115
    lidar_num_rays = 21

    lidar_noise = 0
    missing_lidar_rays = 0
    
    w = World(hallWidths, hallLengths, turns,\
              car_dist_s, car_dist_f, car_heading, car_V,\
              episode_length, time_step, lidar_field_of_view,\
              lidar_num_rays, lidar_noise, missing_lidar_rays, state_feedback=state_feedback)

    action_scale = float(w.action_space.high[0])

    throttle = 16

    rew = 0

    observation = w.scan_lidar()

    if state_feedback:
        observation =  w.reset(side_pos = car_dist_s, pos_noise = 0, heading_noise = 0)

    prev_err = 0
    
    for e in range(episode_length):

        if not state_feedback:
            observation = normalize(observation)
            
        if 'yml' in input_filename:
            delta = action_scale * predict(model, observation.reshape(1,len(observation)))
        else:
            delta = action_scale * model.predict(observation.reshape(1,len(observation)))

        delta = np.clip(delta, -action_scale, action_scale)
        
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
