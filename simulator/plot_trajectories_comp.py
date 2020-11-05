from Car import World
from Car import square_hall_right
from Car import square_hall_left
from Car import trapezoid_hall_sharp_right
from Car import triangle_hall_sharp_right
from Car import triangle_hall_equilateral_right
from Car import T_hall_right
import numpy as np
import random
from tensorflow.keras import models
import matplotlib.pyplot as plt
import sys
import yaml

# direction parameters
UP = 0
DOWN = 1
RIGHT = 2
LEFT = 3

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
    input_right = argv[0]
    
    if 'yml' in input_right:
        with open(input_right, 'rb') as f:
            
            right_ctrl = yaml.load(f)
    else:
    
        right_ctrl = models.load_model(input_right)
    
    numTrajectories = 100
    mode_predictor = models.load_model("modepredictor.h5")

    (hallWidths, hallLengths, turns) = T_hall_right(1.5)
    #(hallWidths, hallLengths, turns) = square_hall_left(1.5)
    #(hallWidths, hallLengths, turns) = triangle_hall_equilateral_right(1.5)
    
    car_dist_s = hallWidths[0]/2.0
    car_dist_f = 6.5
    car_heading = 0
    car_V = 2.4
    episode_length = 100
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
    
    allX = []
    allY = []
    allR = []

    posX = []
    posY = []
    negX = []
    negY = []

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
            
            mode = np.argmax(mode_predictor.predict(observation.reshape(1,len(observation)))[0])
            
            if mode == 2:
                observation = reverse_lidar(observation)
            
            allX.append(w.car_global_x)
            allY.append(w.car_global_y)
            
            if 'yml' in input_right:
                delta = action_scale * predict(right_ctrl, observation.reshape(1,len(observation)))
            else:
                delta = action_scale * right_ctrl.predict(observation.reshape(1,len(observation)))
            
            if mode == 2:
                delta = -delta

            # verifying mode predictor
            if w.car_dist_f >= 4 and w.direction == UP:
                if mode == 0:
                    posX.append(w.car_global_x)
                    posY.append(w.car_global_y)
                else:
                    it += 1
                    negX.append(w.car_global_x)
                    negY.append(w.car_global_y)

            if w.car_dist_f < 4 and w.car_dist_s < 1.5 and w.direction == UP:
                if mode == 1:
                    posX.append(w.car_global_x)
                    posY.append(w.car_global_y)
                else:
                    it += 1
                    negX.append(w.car_global_x)
                    negY.append(w.car_global_y)

            if w.car_dist_s >= 1.5 and w.car_dist_f < 1.5 and w.direction == UP:
                if mode == 0:
                    posX.append(w.car_global_x)
                    posY.append(w.car_global_y)
                else:
                    it += 1
                    negX.append(w.car_global_x)
                    negY.append(w.car_global_y)

            if w.car_dist_f >= 4 and w.direction == RIGHT:
                if mode == 0:
                    posX.append(w.car_global_x)
                    posY.append(w.car_global_y)
                else:
                    it += 1
                    negX.append(w.car_global_x)
                    negY.append(w.car_global_y)

            if w.car_dist_f < 4 and w.car_dist_s < 1.5 and w.direction == RIGHT:
                if mode == 2:
                    posX.append(w.car_global_x)
                    posY.append(w.car_global_y)
                else:
                    it += 1
                    negX.append(w.car_global_x)
                    negY.append(w.car_global_y)

            if w.car_dist_s >= 1.5 and w.car_dist_f < 1.5 and w.direction == RIGHT:
                if mode == 0:
                    posX.append(w.car_global_x)
                    posY.append(w.car_global_y)
                else:
                    it += 1
                    negX.append(w.car_global_x)
                    negY.append(w.car_global_y) 
            
            observation, reward, done, info = w.step(delta, throttle)

            if done:
                
                if e < episode_length - 1:
                    num_unsafe += 1
                
                break

            rew += reward

    print('number of crashes: ' + str(num_unsafe))
    print(it)
    fig = plt.figure(figsize=(12,10))
    w.plotHalls()
    
    #plt.ylim((-1,11))
    # plt.xlim((-1.75,15.25))
    # plt.tick_params(labelsize=20)

    plt.scatter(posX, posY, s = 1, c = 'r')
    plt.scatter(negX, negY, s = 1, c = 'b')
    # plt.plot(negX, negY, 'b')

    # for i in range(numTrajectories):
    #     plt.plot(allX[i], allY[i], 'r-')

    plt.show()
    
if __name__ == '__main__':
    main(sys.argv[1:])
