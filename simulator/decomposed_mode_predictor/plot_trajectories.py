import os
import sys
sys.path.append('..')
from Car import World
from Car import square_hall_right
from Car import square_hall_left
from Car import trapezoid_hall_sharp_right
from Car import triangle_hall_sharp_right
from Car import triangle_hall_equilateral_right
from Car import T_hall_right
from Car import complex_track
import numpy as np
import random
from tensorflow.keras import models
import matplotlib.pyplot as plt
import yaml
from enum import Enum

# direction parameters
UP = 0
DOWN = 1
RIGHT = 2
LEFT = 3

class Modes(Enum):
    STRAIGHT = 'STRAIGHT'
    SQUARE_RIGHT = 'SQUARE_RIGHT'
    SQUARE_LEFT = 'SQUARE_LEFT'
    SHARP_RIGHT = 'SHARP_RIGHT'
    SHARP_LEFT = 'SHARP_LEFT'

def int2mode(i):
    if i == 0:
        return Modes.STRAIGHT
    elif i == 1:
        return Modes.SQUARE_RIGHT
    elif i == 2:
        return Modes.SQUARE_LEFT
    elif i == 3:
        return Modes.SHARP_RIGHT
    elif i == 4:
        return Modes.SHARP_LEFT
    else:
        raise ValueError

class ComposedModePredictor:
    def __init__(self, big_file,
                 straight_file, square_right_file, square_left_file,
                 sharp_right_file, sharp_left_file, yml=False):

        self.yml = yml

        if yml:
            with open(big_file, 'rb') as f:
                print(big_file)
                self.big = yaml.load(f, Loader=yaml.CLoader)

            self.little = {}
            with open(straight_file, 'rb') as f:
                self.little[Modes.STRAIGHT] = yaml.load(f, Loader=yaml.CLoader)
            with open(square_right_file, 'rb') as f:
                self.little[Modes.SQUARE_RIGHT] = yaml.load(f, Loader=yaml.CLoader)
            with open(square_left_file, 'rb') as f:
                self.little[Modes.SQUARE_LEFT] = yaml.load(f, Loader=yaml.CLoader)
            with open(sharp_right_file, 'rb') as f:
                self.little[Modes.SHARP_RIGHT] = yaml.load(f, Loader=yaml.CLoader)
            with open(sharp_left_file, 'rb') as f:
                self.little[Modes.SHARP_LEFT] = yaml.load(f, Loader=yaml.CLoader)                
        else:
        
            self.big = models.load_model(big_file)
            self.little = {
                    Modes.STRAIGHT: models.load_model(straight_file),
                    Modes.SQUARE_RIGHT: models.load_model(square_right_file),
                    Modes.SQUARE_LEFT: models.load_model(square_left_file),
                    Modes.SHARP_RIGHT: models.load_model(sharp_right_file),
                    Modes.SHARP_LEFT: models.load_model(sharp_left_file)
                    }
        self.current_mode = Modes.STRAIGHT

    def predict(self, observation):
        obs = observation.reshape(1, -1)

        if self.yml:
            if predict(self.little[self.current_mode], obs).round()[0] > 0.5:
                self.current_mode = int2mode(np.argmax(predict(self.big, obs)))
        else:
            if self.little[self.current_mode].predict(obs).round()[0] > 0.5:
                self.current_mode = int2mode(np.argmax(self.big.predict(obs)))
        return self.current_mode

class ComposedSteeringPredictor:
    def __init__(self, square_file, sharp_file, action_scale):
        with open(square_file, 'rb') as f:
            self.square_ctrl = yaml.load(f, Loader=yaml.CLoader)
        with open(sharp_file, 'rb') as f:
            self.sharp_ctrl = yaml.load(f, Loader=yaml.CLoader)
        self.action_scale = action_scale

    def predict(self, observation, mode):
        if mode == Modes.STRAIGHT or mode == Modes.SQUARE_RIGHT or mode == Modes.SQUARE_LEFT:
            delta = self.action_scale * predict(self.square_ctrl, observation.reshape(1, -1))
        else:
            delta = self.action_scale * predict(self.sharp_ctrl, observation.reshape(1, -1))
        if mode == Modes.SQUARE_LEFT or mode == Modes.SHARP_LEFT:
            delta = -delta
        return delta

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

def reverse_lidar(data):
    new_data = np.zeros((data.shape))

    for i in range(len(data)):
        new_data[i] = data[len(data) - i - 1]

    return new_data

def main(argv):
    
    numTrajectories = 100

    # mode_predictor = ComposedModePredictor(
    #         'big.h5', 'straight_little.h5',
    #         'square_right_little.h5', 'square_left_little.h5',
    #         'sharp_right_little.h5', 'sharp_left_little.h5'
    #         )

    mode_predictor = ComposedModePredictor(
            'big.yml', 'straight_little.yml',
            'square_right_little.yml', 'square_left_little.yml',
            'sharp_right_little.yml', 'sharp_left_little.yml', True)    

    #(hallWidths, hallLengths, turns) = T_hall_right(1.5)
    #(hallWidths, hallLengths, turns) = square_hall_left(1.5)
    #(hallWidths, hallLengths, turns) = triangle_hall_equilateral_right(1.5)
    (hallWidths, hallLengths, turns) = complex_track(1.5)
    
    car_dist_s = hallWidths[0]/2.0
    car_dist_f = 6.5
    car_heading = 0
    car_V = 2.4
    episode_length = 400
    time_step = 0.1

    state_feedback = False

    lidar_field_of_view = 115
    lidar_num_rays = 21

    # Change this to 0.1 or 0.2 to generate Figure 3 or 5 in the paper, respectively
    lidar_noise = 0

    # Change this to 0 or 5 to generate Figure 3 or 5 in the paper, respectively
    missing_lidar_rays = 0

    num_unsafe = 0

    w = World(hallWidths, hallLengths, turns,\
              car_dist_s, car_dist_f, car_heading, car_V,\
              episode_length, time_step, lidar_field_of_view,\
              lidar_num_rays, lidar_noise, missing_lidar_rays, True, state_feedback=state_feedback)

    throttle = 16
    action_scale = float(w.action_space.high[0])

    steering_ctrl = ComposedSteeringPredictor(
            os.path.join('..', '..', 'verisig', 'tanh64x64_right_turn_lidar.yml'),
            os.path.join('..', '..', 'verisig', 'tanh64x64_sharp_turn_lidar.yml'),
            action_scale
            )
    
    allX = []
    allY = []
    allR = []

    #posX = []
    #posY = []
    #negX = []
    #negY = []
    straight_pred_x = []
    square_right_pred_x = []
    square_left_pred_x = []
    sharp_right_pred_x = []
    sharp_left_pred_x = []
    straight_pred_y = []
    square_right_pred_y = []
    square_left_pred_y = []
    sharp_right_pred_y = []
    sharp_left_pred_y = []

    # initial uncertainty
    init_pos_noise = 0.1
    init_heading_noise = 0.2
    it = 0

    for step in range(numTrajectories):

        observation = w.reset(pos_noise = init_pos_noise, heading_noise = init_heading_noise)

        init_cond = [w.car_dist_f, w.car_dist_s, w.car_heading, w.car_V]

        #observation = w.scan_lidar()

        rew = 0

        for e in range(episode_length):
            
            if not state_feedback:
                observation = normalize(observation)
            
            #mode = np.argmax(mode_predictor.predict(observation.reshape(1,len(observation)))[0])
            mode = mode_predictor.predict(observation)
            
            if mode == Modes.SQUARE_LEFT or mode == Modes.SHARP_LEFT:
                observation = reverse_lidar(observation)
            
            allX.append(w.car_global_x)
            allY.append(w.car_global_y)
            
            delta = steering_ctrl.predict(observation, mode)

            if mode == Modes.STRAIGHT:
                straight_pred_x.append(w.car_global_x)
                straight_pred_y.append(w.car_global_y)
            elif mode == Modes.SQUARE_RIGHT:
                square_right_pred_x.append(w.car_global_x)
                square_right_pred_y.append(w.car_global_y)
            elif mode == Modes.SQUARE_LEFT:
                square_left_pred_x.append(w.car_global_x)
                square_left_pred_y.append(w.car_global_y)
            elif mode == Modes.SHARP_RIGHT:
                sharp_right_pred_x.append(w.car_global_x)
                sharp_right_pred_y.append(w.car_global_y)
            elif mode == Modes.SHARP_LEFT:
                sharp_left_pred_x.append(w.car_global_x)
                sharp_left_pred_y.append(w.car_global_y)
            
            observation, reward, done, info = w.step(delta, throttle)

            if done:
                
                if e < episode_length - 1:
                    num_unsafe += 1
                
                break

            rew += reward

    print('number of crashes: ' + str(num_unsafe))
    #print(it)
    fig = plt.figure(figsize=(12,10))
    w.plotHalls()
    
    #plt.ylim((-1,11))
    # plt.xlim((-1.75,15.25))
    # plt.tick_params(labelsize=20)

    #plt.scatter(posX, posY, s = 1, c = 'r')
    #plt.scatter(negX, negY, s = 1, c = 'b')
    # plt.plot(negX, negY, 'b')
    plt.scatter(straight_pred_x, straight_pred_y, s=1, c='r', label='straight')
    plt.scatter(square_right_pred_x, square_right_pred_y, s=1, c='g', label='square right')
    plt.scatter(square_left_pred_x, square_left_pred_y, s=1, c='b', label='square left')
    plt.scatter(sharp_right_pred_x, sharp_right_pred_y, s=1, c='m', label='sharp right')
    plt.scatter(sharp_left_pred_x, sharp_left_pred_y, s=1, c='c', label='sharp left')

    # for i in range(numTrajectories):
    #     plt.plot(allX[i], allY[i], 'r-')

    #plt.show()
    plt.legend(markerscale=10)
    plt.savefig('trajectories.png')
    
if __name__ == '__main__':
    main(sys.argv[1:])
