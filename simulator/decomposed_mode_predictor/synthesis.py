import numpy as np
import warnings
import yaml
import time
import os

from enum import Enum
from tensorflow.keras import models

from Car import triangle_hall_equilateral_right
from Car import square_hall_right
from Car import World

warnings.filterwarnings("ignore")


START_STATE = [0.825, 0., 0.]
SQUARE_START_Y = 6.5
SQUARE_EXIT_Y = 12.
SHARP_START_Y = 8.
SHARP_EXIT_Y = 10.

X_SAMPLE_MEAN = 0.8
H_SAMPLE_MEAN = 0.
X_NOISE = 0.05
H_NOISE = 0.006
Y_NOISE = 0.25


class Modes(Enum):
    STRAIGHT = 'STRAIGHT'
    SQUARE_RIGHT = 'SQUARE_RIGHT'
    SQUARE_LEFT = 'SQUARE_LEFT'
    SHARP_RIGHT = 'SHARP_RIGHT'
    SHARP_LEFT = 'SHARP_LEFT'


SYNTH_MODES = [Modes.SQUARE_RIGHT, Modes.SQUARE_LEFT, Modes.SHARP_RIGHT, Modes.SHARP_LEFT]


def sigmoid(x):

    sigm = 1. / (1. + np.exp(-x))

    return sigm


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
                self.big = yaml.load(f)

            self.little = {}
            with open(straight_file, 'rb') as f:
                self.little[Modes.STRAIGHT] = yaml.load(f)
            with open(square_right_file, 'rb') as f:
                self.little[Modes.SQUARE_RIGHT] = yaml.load(f)
            with open(square_left_file, 'rb') as f:
                self.little[Modes.SQUARE_LEFT] = yaml.load(f)
            with open(sharp_right_file, 'rb') as f:
                self.little[Modes.SHARP_RIGHT] = yaml.load(f)
            with open(sharp_left_file, 'rb') as f:
                self.little[Modes.SHARP_LEFT] = yaml.load(f)
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
            self.square_ctrl = yaml.load(f)
        with open(sharp_file, 'rb') as f:
            self.sharp_ctrl = yaml.load(f)
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


def get_dist_f(mode):
    if mode == Modes.SQUARE_RIGHT or mode == Modes.SQUARE_LEFT:
        car_dist_f = SQUARE_START_Y
        car_exit_f = SQUARE_EXIT_Y
    if mode == Modes.SHARP_RIGHT or mode == Modes.SHARP_LEFT:
        car_dist_f = SHARP_START_Y
        car_exit_f = SHARP_EXIT_Y
    return car_dist_f, car_exit_f


def generate_implications(num_trajectories, mode, mode_predictor, num_modes=4,
                          car_dist_s=X_SAMPLE_MEAN, car_heading=H_SAMPLE_MEAN,
                          s_noise=X_NOISE, f_noise=Y_NOISE,
                          h_noise=H_NOISE, crash_print=True, seed_sample=True):

    if mode == Modes.SQUARE_RIGHT or mode == Modes.SQUARE_LEFT:
        (hallWidths, hallLengths, turns) = square_hall_right(1.5)
    if mode == Modes.SHARP_RIGHT or mode == Modes.SHARP_LEFT:
        (hallWidths, hallLengths, turns) = triangle_hall_equilateral_right(1.5)

    car_V = 2.4
    car_dist_f, car_exit_f = get_dist_f(mode)

    if mode == Modes.SQUARE_LEFT or mode == Modes.SHARP_LEFT:
        car_dist_s = 1.5 - car_dist_s
        car_heading = -car_heading

    episode_length = 100
    time_step = 0.1

    state_feedback = False
    lidar_field_of_view = 115
    lidar_num_rays = 21

    # Change this to 0.1 or 0.2 to generate Figure 3 or 5 in the paper, respectively
    lidar_noise = 0

    # Change this to 0 or 5 to generate Figure 3 or 5 in the paper, respectively
    missing_lidar_rays = 0

    w = World(hallWidths, hallLengths, turns,
              car_dist_s, car_dist_f, car_heading, car_V,
              episode_length, time_step, lidar_field_of_view,
              lidar_num_rays, lidar_noise, missing_lidar_rays, True, state_feedback=state_feedback)

    action_scale = float(w.action_space.high[0])
    steering_ctrl = ComposedSteeringPredictor(
        os.path.join('..', '..', 'verisig', 'tanh64x64_right_turn_lidar.yml'),
        os.path.join('..', '..', 'verisig', 'tanh64x64_sharp_turn_lidar.yml'),
        action_scale
    )
    throttle = 16

    entry_set = []
    exit_set = []

    # initial uncertainty
    init_pos_noise = s_noise
    init_heading_noise = h_noise
    init_y_noise = f_noise

    # init state
    init_point = [START_STATE[0], car_dist_f-START_STATE[1], START_STATE[2]]
    if mode == Modes.SHARP_LEFT or mode == Modes.SQUARE_LEFT:
        init_point[0] = 1.5 - init_point[0]
        init_point[2] = -init_point[2]

    for step in range(num_trajectories):

        if step == 0 and seed_sample:
            observation = w.reset(side_pos=init_point)
        else:
            observation = w.reset(pos_noise=init_pos_noise,
                                  heading_noise=init_heading_noise, y_noise=init_y_noise)

        init_cond = np.array([w.car_dist_s, car_dist_f - w.car_dist_f, w.car_heading])
        if mode == Modes.SQUARE_LEFT or mode == Modes.SHARP_LEFT:
            init_cond[0] = 1.5 - init_cond[0]
            init_cond[2] = -init_cond[2]

        mode_predictor.current_mode = Modes.STRAIGHT

        for e in range(episode_length):

            if not state_feedback:
                observation = normalize(observation)
            if mode == Modes.SQUARE_LEFT or mode == Modes.SHARP_LEFT:
                observation = reverse_lidar(observation)

            predicted_mode = mode_predictor.predict(observation)

            if predicted_mode == Modes.SQUARE_LEFT or predicted_mode == Modes.SHARP_LEFT:
                observation = reverse_lidar(observation)

            delta = steering_ctrl.predict(observation, predicted_mode)
            if mode == Modes.SQUARE_LEFT or mode == Modes.SHARP_LEFT:
                delta = -delta

            observation, _, done, _ = w.step(delta, throttle)

            if done:
                if e < episode_length - 1 and crash_print:
                    print('Crash starting at: {}'.format(init_cond.tolist()))
                break

            elif w.ax_changed and w.car_dist_f <= car_exit_f:
                entry_set.append(init_cond)
                exit_cond = np.array([w.car_dist_s, car_exit_f - w.car_dist_f, w.car_heading])
                if mode == Modes.SQUARE_LEFT or mode == Modes.SHARP_LEFT:
                    exit_cond[0] = 1.5 - exit_cond[0]
                    exit_cond[2] = -exit_cond[2]
                exit_set.append(exit_cond)
                break

    implications = []
    for i in range(len(entry_set)):
        implications.append(((mode, entry_set[i]), (mode, exit_set[i]), 'dynamics'))
        # Only one entry set and hence one transition is enough.
        implications.append(((mode, exit_set[i]), (None, exit_set[i]), 'transition'))

    return implications


def mode2int(mode):
    if mode == Modes.STRAIGHT:
        return 0
    if mode == Modes.SQUARE_RIGHT:
        return 1
    if mode == Modes.SQUARE_LEFT:
        return 2
    if mode == Modes.SHARP_RIGHT:
        return 3
    if mode == Modes.SHARP_LEFT:
        return 4
    else:
        raise ValueError


def extend(box, point):
    for i in range(len(box)):
        if box[i] is None:
            box[i] = [point[i], point[i]]
        elif point[i] < box[i][0]:
            box[i][0] = point[i]
        elif point[i] > box[i][1]:
            box[i][1] = point[i]


def contains(box, point):
    retval = True
    for i in range(len(box)):
        if box[i] is None:
            retval = False
        elif point[i] < box[i][0] or point[i] > box[i][1]:
            retval = False
    return retval


def print_conditions(boxes):
    for i in range(len(boxes)):
        box = boxes[i]
        mode = int2mode(i)
        if i == 0:
            print('\nEntry')
        else:
            print('\nExit in {}'.format(mode))

        print('x in [{}, {}]'.format(box[0][0], box[0][1]))
        print('y in [{}, {}]'.format(box[1][0], box[1][1]))
        print('theta in [{}, {}]'.format(box[2][0], box[2][1]))
    print('\n')


def synthesize(boxes, implications):
    i = 0
    while i < len(implications):
        left_b = boxes[implications[i][0][0]]
        right_b = boxes[implications[i][1][0]]
        left_p = implications[i][0][1]
        right_p = implications[i][1][1]

        if contains(left_b, left_p):
            if not contains(right_b, right_p):
                extend(right_b, right_p)
                i = 0
            else:
                i += 1
        else:
            i += 1


def make_synthesis_instance(implications, num_modes=4):

    # Create boxes
    entry_box = []
    for s in START_STATE:
        entry_box.append([s, s])
    boxes = [entry_box]
    for m in range(num_modes):
        boxes.append([None, None, None])

    # Create implications
    box_implications = []
    for imp in implications:
        if imp[2] == 'dynamics':
            left_b = 0
            right_b = mode2int(imp[1][0])
        elif imp[2] == 'transition':
            left_b = mode2int(imp[0][0])
            right_b = 0
        left_p = imp[0][1]
        right_p = imp[1][1]
        box_implications.append([(left_b, left_p), (right_b, right_p)])

    return boxes, box_implications


def synthesis_with_sampling(num_iter, num_trajectories, mode_predictor):
    boxes = None
    implications = []
    cur_time = time.time()
    synthesis_time = 0.
    simulation_time = 0.

    for j in range(num_iter):
        print('\nIteration {}'.format(j))

        # Compute sample ranges.
        if boxes is not None:
            box = boxes[0]
            x_mean = (box[0][0] + box[0][1]) / 2
            x_noise = (box[0][1] - box[0][0]) / 2
            h_mean = (box[2][0] + box[2][1]) / 2
            h_noise = (box[2][1] - box[2][0]) / 2
            y_noise = box[1][1]
            seed_sample = False
        else:
            x_mean = X_SAMPLE_MEAN
            x_noise = X_NOISE
            h_mean = H_SAMPLE_MEAN
            h_noise = H_NOISE
            y_noise = Y_NOISE
            seed_sample = True

        # Simulation
        raw_implications = []
        for i in range(4):
            print('Starting simulation in mode {}'.format(int2mode(i+1)))
            new_imps = generate_implications(num_trajectories, int2mode(i+1), mode_predictor,
                                             car_dist_s=x_mean, car_heading=h_mean, s_noise=x_noise,
                                             f_noise=y_noise, h_noise=h_noise,
                                             seed_sample=seed_sample)
            raw_implications.extend(new_imps)

        # Record simulation time
        simulation_time += (time.time() - cur_time)
        cur_time = time.time()

        # Convert to box synthesis
        new_boxes, new_implications = make_synthesis_instance(raw_implications)
        if boxes is None:
            boxes = new_boxes
        implications.extend(new_implications)

        # Synthesis
        print('Synthesizing all boxes...')
        synthesize(boxes, implications)

        # Record sythesis time
        synthesis_time += (time.time() - cur_time)
        cur_time = time.time()

    print('\nTotal simulation time: {}'.format(simulation_time))
    print('Total synthesis time: {}'.format(synthesis_time))

    return boxes


if __name__ == '__main__':

    mode_predictor = ComposedModePredictor(
        'big.yml', 'straight_little.yml',
        'square_right_little.yml', 'square_left_little.yml',
        'sharp_right_little.yml', 'sharp_left_little.yml', True)

    boxes = synthesis_with_sampling(10, 100, mode_predictor)

    print_conditions(boxes)
