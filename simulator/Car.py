import gym
from gym import spaces
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# car parameters
CAR_LENGTH = .45 # in m
CAR_CENTER_OF_MASS = .225 # from rear of car (m)
CAR_DECEL_CONST = .4
CAR_ACCEL_CONST = 1.633 # estimated from data
CAR_MOTOR_CONST = 0.2 # estimated from data
HYSTERESIS_CONSTANT = 4
MAX_TURNING_INPUT = 15 # in degrees

# lidar parameter
LIDAR_RANGE = 5 # in m

# safety parameter
SAFE_DISTANCE = 0.1 # in m

# default throttle if left unspecified
CONST_THROTTLE = 16

# training parameters
STEP_REWARD_GAIN = 10
INPUT_REWARD_GAIN = -0.05
CRASH_REWARD = -100
MIDDLE_REWARD_GAIN = -5

# direction parameters
UP = 0
DOWN = 1
RIGHT = 2
LEFT = 3

class World:

    def __init__(self, hallWidths, hallLengths, turns,\
                 car_dist_s, car_dist_f, car_heading, car_V,\
                 episode_length, time_step, lidar_field_of_view,\
                 lidar_num_rays, lidar_noise = 0, lidar_missing_rays = 0, lidar_missing_in_turn_only = False):

        # hallway parameters
        self.numHalls = len(hallWidths)
        self.hallWidths = hallWidths
        self.hallLengths = hallLengths
        self.turns = turns
        self.curHall = 0

        # car relative states
        self.car_dist_s = car_dist_s
        self.car_dist_f = car_dist_f
        self.car_V = car_V
        self.car_heading = car_heading

        # car global states
        self.car_global_x = -self.hallWidths[0] / 2.0 + self.car_dist_s
        self.car_global_y = self.hallLengths[0] / 2.0 - car_dist_f
        self.car_global_heading = self.car_heading + np.pi / 2 #first hall goes "up" by default
        self.direction = UP

        # car initial conditions (used in reset)
        self.init_car_dist_s = self.car_dist_s
        self.init_car_dist_f = self.car_dist_f
        self.init_car_heading = self.car_heading
        self.init_car_V = self.car_V

        # step parameters
        self.time_step = time_step
        self.cur_step = 0
        self.episode_length = episode_length

        # storage
        self.allX = []
        self.allY = []
        self.allX.append(self.car_global_x)
        self.allY.append(self.car_global_y)

        # lidar setup
        self.lidar_field_of_view = lidar_field_of_view
        self.lidar_num_rays = lidar_num_rays

        self.lidar_noise = lidar_noise
        self.total_lidar_missing_rays = lidar_missing_rays

        self.lidar_missing_in_turn_only = lidar_missing_in_turn_only
        
        self.cur_num_missing_rays = lidar_missing_rays
        self.missing_indices = np.random.choice(self.lidar_num_rays, self.cur_num_missing_rays)

        # parameters needed for consistency with gym environments
        self.obs_low = np.zeros(self.lidar_num_rays, )
        self.obs_high = LIDAR_RANGE * np.ones(self.lidar_num_rays, )

        self.action_space = spaces.Box(low=-MAX_TURNING_INPUT, high=MAX_TURNING_INPUT, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=self.obs_low, high=self.obs_high, dtype=np.float32)

        self._max_episode_steps = episode_length

    # this is a limited-support function for setting the car state in the first hallway
    def set_state_local(self, x, y, theta):
        
        self.car_dist_s = x
        self.car_dist_f = y
        self.car_heading = theta

        self.car_global_x = x - self.hallWidths[0]/2
        self.car_global_y = -y + self.hallLengths[0]/2
        self.car_global_heading = theta + np.pi / 2
        self.direction = UP

        #test if in Region 3
        if y > self.hallLengths[0] - LIDAR_RANGE:

            self.direction = RIGHT
            
            temp = x
            self.car_dist_s = self.hallLengths[0] - y
            self.car_dist_f = temp
            self.car_heading = theta - np.pi / 2

            if self.car_heading < - np.pi:
                self.car_heading = self.car_heading + 2 * np.pi

        if self.car_global_heading > np.pi:
            self.car_global_heading = self.car_global_heading - 2 * np.pi

    # this is a limited-support function for setting the car state in the first hallway
    def set_state_global(self, x, y, theta):

        self.car_dist_s = x + self.hallWidths[0]/2
        self.car_dist_f = -y + self.hallLengths[0]/2
        self.car_heading = theta - np.pi / 2

        self.car_global_x = x
        self.car_global_y = y
        self.car_global_heading = theta
        
        self.direction = UP

        #test if in Region 3
        if y > self.hallLengths[0] - LIDAR_RANGE:

            self.direction = RIGHT
            
            temp = x
            self.car_dist_s = self.hallLengths[0] - y
            self.car_dist_f = temp
            self.car_heading = theta - np.pi / 2

            if self.car_heading < - np.pi:
                self.car_heading = self.car_heading + 2 * np.pi

        if self.car_global_heading > np.pi:
            self.car_global_heading = self.car_global_heading - 2 * np.pi        

    def reset(self, side_pos = None, pos_noise = 0.2, heading_noise = 0.1):
        self.curHall = 0

        self.car_dist_s = self.init_car_dist_s + np.random.uniform(-pos_noise, pos_noise)

        if not side_pos == None:
            self.car_dist_s = side_pos
        
        self.car_dist_f = self.init_car_dist_f
        self.car_V = self.init_car_V
        self.car_heading = self.init_car_heading + np.random.uniform(-heading_noise, heading_noise)
        
        self.car_global_x = -self.hallWidths[0] / 2.0 + self.car_dist_s
        if 'left' in self.turns[0]:
            self.car_global_x = -self.car_global_x
        
        self.car_global_y = self.hallLengths[0] / 2.0 - self.car_dist_f
        
        self.car_global_heading = self.car_heading + np.pi / 2 #first hall goes "up" by default
        self.direction = UP

        self.missing_indices = np.random.choice(self.lidar_num_rays, self.cur_num_missing_rays)
        
        self.cur_step = 0

        self.allX = []
        self.allY = []
        self.allX.append(self.car_global_x)
        self.allY.append(self.car_global_y)

        return self.scan_lidar()

    #NB: Mode switches are handled in the step function
    # x := [s, f, V, theta_local, x, y, theta_global]
    def bicycle_dynamics(self, x, t, u, delta, turn):

        if 'right' in turn:
            # -V * sin(theta_local + beta)
            dsdt = -x[2] * np.sin(x[3] + np.arctan(CAR_CENTER_OF_MASS * np.tan(delta) / CAR_LENGTH))
        else:
            # V * sin(theta_local + beta)
            dsdt = x[2] * np.sin(x[3] + np.arctan(CAR_CENTER_OF_MASS * np.tan(delta) / CAR_LENGTH))

        # -V * cos(theta_local + beta)
        dfdt = -x[2] * np.cos(x[3] + np.arctan(CAR_CENTER_OF_MASS * np.tan(delta) / CAR_LENGTH)) 

        if u > HYSTERESIS_CONSTANT:
            # a * u - V
            dVdt = CAR_ACCEL_CONST * CAR_MOTOR_CONST * (u - HYSTERESIS_CONSTANT) - CAR_ACCEL_CONST * x[2]
        else:
            dVdt = - CAR_ACCEL_CONST * x[2]
            

        dtheta_ldt = x[2] * np.cos(np.arctan(CAR_CENTER_OF_MASS * np.tan(delta) / CAR_LENGTH)) * np.tan(delta) / CAR_LENGTH 

        # V * cos(theta_global + beta)
        dxdt = x[2] * np.cos(x[6] + np.arctan(CAR_CENTER_OF_MASS * np.tan(delta) / CAR_LENGTH)) 

        # V * sin(theta_global + beta)
        dydt = x[2] * np.sin(x[6] + np.arctan(CAR_CENTER_OF_MASS * np.tan(delta) / CAR_LENGTH))

        # V * cos(beta) * tan(delta) / l
        dtheta_gdt = x[2] * np.cos(np.arctan(CAR_CENTER_OF_MASS * np.tan(delta) / CAR_LENGTH)) * np.tan(delta) / CAR_LENGTH

        dXdt = [dsdt, dfdt, dVdt, dtheta_ldt, dxdt, dydt, dtheta_gdt]
        
        return dXdt

    #NB: Mode switches are handled in the step function
    # x := [s, f, V, theta_local, x, y, theta_global]
    def bicycle_dynamics_no_beta(self, x, t, u, delta, turn):

        if 'right' in turn:
            # -V * sin(theta_local)
            dsdt = -x[2] * np.sin(x[3])
        else:
            # V * sin(theta_local)
            dsdt = x[2] * np.sin(x[3])

        # -V * cos(theta_local)
        dfdt = -x[2] * np.cos(x[3]) 

        if u > HYSTERESIS_CONSTANT:
            # a * u - V
            dVdt = CAR_ACCEL_CONST * CAR_MOTOR_CONST * (u - HYSTERESIS_CONSTANT) - CAR_ACCEL_CONST * x[2]
        else:
            dVdt = - CAR_ACCEL_CONST * x[2]
            
        # V * tan(delta) / l
        dtheta_ldt = x[2] * np.tan(delta) / CAR_LENGTH 

        # V * cos(theta_global)
        dxdt = x[2] * np.cos(x[6]) 

        # V * sin(theta_global)
        dydt = x[2] * np.sin(x[6])

        # V * tan(delta) / l
        dtheta_gdt = x[2] * np.tan(delta) / CAR_LENGTH

        dXdt = [dsdt, dfdt, dVdt, dtheta_ldt, dxdt, dydt, dtheta_gdt]
        
        return dXdt
    
    def step(self, delta, throttle = CONST_THROTTLE, x_noise = 0, y_noise = 0, v_noise = 0, theta_noise = 0):
        self.cur_step += 1

        # Constrain turning input
        if delta > MAX_TURNING_INPUT:
            delta = MAX_TURNING_INPUT

        if delta < -MAX_TURNING_INPUT:
            delta = -MAX_TURNING_INPUT

        # simulate dynamics
        x0 = [self.car_dist_s, self.car_dist_f, self.car_V, self.car_heading, self.car_global_x, self.car_global_y, self.car_global_heading]
        t = [0, self.time_step]
        
        #new_x = odeint(self.bicycle_dynamics, x0, t, args=(throttle, delta * np.pi / 180, self.turns[self.curHall],))
        new_x = odeint(self.bicycle_dynamics_no_beta, x0, t, args=(throttle, delta * np.pi / 180, self.turns[self.curHall],))

        new_x = new_x[1]

        # add noise
        x_added_noise = x_noise * (2 * np.random.random() - 1)
        y_added_noise = y_noise * (2 * np.random.random() - 1)
        v_added_noise = v_noise * (2 * np.random.random() - 1)
        #theta_added_noise = theta_noise * (2 * np.random.random() - 1)
        theta_added_noise = theta_noise * (np.random.random())


        new_x[0] = new_x[0] + x_added_noise

        if self.direction == UP and 'right' in self.turns[self.curHall]\
           or self.direction == DOWN and 'left' in self.turns[self.curHall]:
            new_x[4] = new_x[4] + x_added_noise
            
        elif self.direction == DOWN and 'right' in self.turns[self.curHall]\
             or self.direction == UP and 'left' in self.turns[self.curHall]:
            new_x[4] = new_x[4] - x_added_noise
            
        elif self.direction == RIGHT and 'right' in self.turns[self.curHall]\
             or self.direction == LEFT and 'left' in self.turns[self.curHall]:
            new_x[4] = new_x[4] - y_added_noise
            
        elif self.direction == LEFT and 'right' in self.turns[self.curHall]\
             or self.direction == RIGHT and 'left' in self.turns[self.curHall]:
            new_x[4] = new_x[4] + y_added_noise
        
        new_x[1] = new_x[1] + y_added_noise

        if self.direction == UP and 'right' in self.turns[self.curHall]\
           or self.direction == DOWN and 'left' in self.turns[self.curHall]:
            new_x[5] = new_x[5] - y_added_noise
            
        elif self.direction == DOWN and 'right' in self.turns[self.curHall]\
             or self.direction == UP and 'left' in self.turns[self.curHall]:
            new_x[5] = new_x[5] + y_added_noise
            
        elif self.direction == RIGHT and 'right' in self.turns[self.curHall]\
             or self.direction == LEFT and 'left' in self.turns[self.curHall]:
            new_x[5] = new_x[5] - x_added_noise
            
        elif self.direction == LEFT and 'right' in self.turns[self.curHall]\
             or self.direction == RIGHT and 'left' in self.turns[self.curHall]:
            new_x[5] = new_x[5] + x_added_noise
        
        new_x[2] = new_x[2] + v_added_noise
        
        # new_x[3] = new_x[3] + theta_added_noise
        # new_x[6] = new_x[6] + theta_added_noise

        # NB: The heading noise only affects heading in the direction
        # of less change
        if new_x[3] < x0[3]:
            new_x[3] = new_x[3] + theta_added_noise
            new_x[6] = new_x[6] + theta_added_noise
        else:
            new_x[3] = new_x[3] - theta_added_noise
            new_x[6] = new_x[6] - theta_added_noise
        # end of adding noise

        self.car_dist_s, self.car_dist_f, self.car_V, self.car_heading, self.car_global_x, self.car_global_y, self.car_global_heading =\
                    new_x[0], new_x[1], new_x[2], new_x[3], new_x[4], new_x[5], new_x[6]

        terminal = False

        # Compute reward
        reward = STEP_REWARD_GAIN

        # Region 1
        if self.car_dist_s > 0 and self.car_dist_s < self.hallWidths[self.curHall] and\
           self.car_dist_f > self.hallWidths[self.curHall]:

            reward += INPUT_REWARD_GAIN * delta * delta
            reward += MIDDLE_REWARD_GAIN * abs(self.car_dist_s - self.hallWidths[self.curHall] / 2.0)
            #pass

        # Region 2
        elif self.car_dist_s > 0 and self.car_dist_s < self.hallWidths[self.curHall] and\
           self.car_dist_f <= self.hallWidths[self.curHall]:

            #reward += INPUT_REWARD_GAIN * delta * delta
            pass

        # Region 3
        elif self.car_dist_s >  self.hallWidths[self.curHall] and\
             self.car_dist_f <= self.hallWidths[self.curHall]:

            pass

        # Set reward to maximum negative value if too close to a wall
        if self.car_dist_s < SAFE_DISTANCE or self.car_dist_f < SAFE_DISTANCE or\
           (self.car_dist_s > self.hallWidths[self.curHall] - SAFE_DISTANCE and self.car_dist_f > self.hallWidths[self.curHall] - SAFE_DISTANCE):
            terminal = True
            reward = CRASH_REWARD

        if self.cur_step == self.episode_length:
            terminal = True

        # Test if a mode switch in the world has changed
        if 'right' in self.turns[self.curHall]:
    
            if self.car_dist_s > LIDAR_RANGE:
                temp = self.car_dist_s
                self.car_heading = self.car_heading + np.pi / 2
                self.car_dist_s = self.car_dist_f # front wall is now the left wall
                self.curHall = self.curHall + 1 # next hallway

                #NB: this case deals with loops in the environment
                if self.curHall >= self.numHalls:
                    self.curHall = 0

                self.car_dist_f = self.hallLengths[self.curHall] - temp

                if self.direction == UP:
                    self.direction = RIGHT
                elif self.direction == RIGHT:
                    self.direction = DOWN
                elif self.direction == DOWN:
                    self.direction = LEFT
                elif self.direction == LEFT:
                    self.direction = UP
    
        else: # left turn 
    
            if self.car_dist_s > LIDAR_RANGE:
                temp = self.car_dist_s
                self.car_heading = self.car_heading - np.pi / 2
                self.car_dist_s = self.car_dist_f # front wall is now the left wall
                self.curHall = self.curHall + 1 # next hallway

                #NB: this case deals with loops in the environment
                if self.curHall >= self.numHalls:
                    self.curHall = 0                

                self.car_dist_f = self.hallLengths[self.curHall] - temp

                if self.direction == UP:
                    self.direction = LEFT
                elif self.direction == RIGHT:
                    self.direction = UP
                elif self.direction == DOWN:
                    self.direction = RIGHT
                elif self.direction == LEFT:
                    self.direction = DOWN                

        self.allX.append(self.car_global_x)
        self.allY.append(self.car_global_y)
        
        return self.scan_lidar(), reward, terminal, -1

    def scan_lidar(self):

        car_heading_deg = self.car_heading * 180 / np.pi

        alpha = int(np.floor(4 * car_heading_deg))

        theta_t = np.linspace(-self.lidar_field_of_view, self.lidar_field_of_view, self.lidar_num_rays)

        # lidar measurements
        data = np.zeros(len(theta_t))

        if 'right' in self.turns[self.curHall]:
            dist_l = self.car_dist_s
            dist_r = self.hallWidths[self.curHall] - self.car_dist_s
        else:
            dist_l = self.hallWidths[self.curHall] - self.car_dist_s
            dist_r = self.car_dist_s

        # Region 1 (before turn)
        if self.car_dist_s > 0 and self.car_dist_s < self.hallWidths[self.curHall] and\
           self.car_dist_f > self.hallWidths[(self.curHall + 1) % self.numHalls]:

            if 'right' in self.turns[self.curHall]:
            
                theta_l = np.arctan(self.car_dist_s / self.car_dist_f) * 180 / np.pi
                theta_r = -np.arctan((self.hallWidths[self.curHall] - self.car_dist_s) /\
                            (self.car_dist_f - self.hallWidths[(self.curHall + 1) % self.numHalls])) * 180 / np.pi

            else:
                theta_l = np.arctan((self.hallWidths[self.curHall] - self.car_dist_s) /\
                                    (self.car_dist_f - self.hallWidths[(self.curHall + 1) % self.numHalls])) * 180 / np.pi
                theta_r = -np.arctan(self.car_dist_s / self.car_dist_f) * 180 / np.pi

            index = 0

            for angle in theta_t:

                angle = angle + car_heading_deg
                if angle > 180:
                    angle = angle - 360
                elif angle < -180:
                    angle = angle + 360

                if angle <= theta_r:
                    data[index] = (dist_r) /\
                            (np.cos( (90 + angle) * np.pi / 180))

                elif angle > theta_r and angle <= theta_l:
                    data[index] = (self.car_dist_f) /\
                            (np.cos( (angle) * np.pi / 180))

                else:
                    data[index] = (dist_l) /\
                            (np.cos( (90 - angle) * np.pi / 180))

                #add noise
                data[index] += np.random.uniform(0, self.lidar_noise)

                if data[index] > LIDAR_RANGE or data[index] < 0:
                    data[index] = LIDAR_RANGE
                    
                index += 1

        # Region 2 (during turn)
        elif self.car_dist_s > 0 and self.car_dist_s < self.hallWidths[self.curHall] and\
           self.car_dist_f <= self.hallWidths[(self.curHall+1) % self.numHalls]:

            if 'right' in self.turns[self.curHall]:
                theta_l = np.arctan(self.car_dist_s / self.car_dist_f) * 180 / np.pi
                theta_r = -np.arctan((self.hallWidths[self.curHall] - self.car_dist_s) /\
                            (self.car_dist_f - self.hallWidths[(self.curHall + 1) % self.numHalls])) * 180 / np.pi - 180

            else:
                theta_l = 180 - np.arctan((self.hallWidths[self.curHall] - self.car_dist_s) /\
                            (self.hallWidths[(self.curHall + 1) % self.numHalls] - self.car_dist_f)) * 180 / np.pi
                theta_r = -np.arctan(self.car_dist_s / self.car_dist_f) * 180 / np.pi
                
            index = 0
            
            for angle in theta_t:

                angle = angle + car_heading_deg
                if angle > 180:
                    angle = angle - 360
                elif angle < -180:
                    angle = angle + 360

                if 'right' in self.turns[self.curHall]:
                    if angle <= theta_r:
                        data[index] = (dist_r) /\
                                (np.cos( (90 + angle) * np.pi / 180))

                    elif angle > theta_r and angle < -90:
                        data[index] = (self.hallWidths[(self.curHall + 1) % self.numHalls] - self.car_dist_f) /\
                                (np.cos( (180 + angle) * np.pi / 180))

                    elif angle > -90 and angle <= theta_l:
                        data[index] = (self.car_dist_f) /\
                                (np.cos( (angle) * np.pi / 180))
                    else:
                        data[index] = (dist_l) /\
                                (np.cos( (90 - angle) * np.pi / 180))

                else:
                    if angle <= theta_r:
                        data[index] = (dist_r) /\
                                (np.cos( (90 + angle) * np.pi / 180))

                    elif angle > theta_r and angle <= 90:
                        data[index] = (self.car_dist_f) /\
                                (np.cos( (angle) * np.pi / 180))

                    elif angle > 90 and angle <= theta_l:
                        data[index] = (self.hallWidths[(self.curHall + 1) % self.numHalls] - self.car_dist_f) /\
                                (np.cos( (180 + angle) * np.pi / 180))
                    else:
                        data[index] = (dist_l) /\
                                (np.cos( (90 - angle) * np.pi / 180))

                #add noise
                data[index] += np.random.uniform(0, self.lidar_noise)

                if data[index] > LIDAR_RANGE or data[index] < 0:
                    data[index] = LIDAR_RANGE

                index += 1

        # Region 3 (after turn)
        elif self.car_dist_s > self.hallWidths[self.curHall] and\
             self.car_dist_f <= self.hallWidths[(self.curHall + 1) % self.numHalls]:

            if 'right' in self.turns[self.curHall]:            
            
                theta_l =  np.arctan(self.car_dist_s / self.car_dist_f) * 180 / np.pi
                theta_r =  180 - np.arctan(- (self.hallWidths[self.curHall] - self.car_dist_s) /\
                            (self.hallWidths[(self.curHall + 1) % self.numHalls] - self.car_dist_f)) * 180 / np.pi

            else:
                theta_l =  np.arctan(- (self.hallWidths[self.curHall] - self.car_dist_s) /\
                            (self.hallWidths[(self.curHall + 1) % self.numHalls] - self.car_dist_f)) * 180 / np.pi - 180
                theta_r =  -np.arctan(self.car_dist_s / self.car_dist_f) * 180 / np.pi

            index = 0

            for angle in theta_t:

                angle = angle + car_heading_deg
                if angle > 180:
                    angle = angle - 360
                elif angle < -180:
                    angle = angle + 360

                if 'right' in self.turns[self.curHall]:
                    if angle < -90:
                        data[index] = (self.hallWidths[(self.curHall + 1) % self.numHalls] - self.car_dist_f) /\
                                (np.cos( (180 + angle) * np.pi / 180))

                    elif angle == -90:
                          data[index] = LIDAR_RANGE

                    elif angle > -90 and angle <= theta_l:
                        data[index] = (self.car_dist_f) /\
                                (np.cos( (angle) * np.pi / 180))

                    elif angle > theta_l and angle <= theta_r:
                        data[index] = (dist_l) /\
                                (np.cos( (90 - angle) * np.pi / 180))

                    else:
                        data[index] = (self.hallWidths[(self.curHall + 1) % self.numHalls] - self.car_dist_f) /\
                                (np.cos( (180 - angle) * np.pi / 180))
                else:
                    if angle > 90:
                        data[index] = (self.hallWidths[(self.curHall + 1) % self.numHalls] - self.car_dist_f) /\
                                (np.cos( (180 + angle) * np.pi / 180))

                    elif angle < 90 and angle >= theta_r:
                        data[index] = (self.car_dist_f) /\
                                (np.cos( (angle) * np.pi / 180))

                    elif angle < theta_r and angle >= theta_l:
                        data[index] = (dist_r) /\
                                (np.cos( (90 - angle) * np.pi / 180))

                    else:
                        data[index] = (self.hallWidths[(self.curHall + 1) % self.numHalls] - self.car_dist_f) /\
                                (np.cos( (180 - angle) * np.pi / 180))

                # add noise
                data[index] += np.random.uniform(0, self.lidar_noise)


                if data[index] > LIDAR_RANGE or data[index] < 0:
                    data[index] = LIDAR_RANGE

                index += 1

        # add missing rays
        if self.lidar_missing_in_turn_only:
            
            # add missing rays only in Region 2 (plus an extra 1m before it)
            if self.car_dist_s > 0 and self.car_dist_s < self.hallWidths[self.curHall] and\
               self.car_dist_f <= self.hallWidths[(self.curHall + 1) % self.numHalls] + 1:

                for ray in self.missing_indices:
                    data[ray] = LIDAR_RANGE                
        else:
            # add missing rays in all regions
            for ray in self.missing_indices:
                data[ray] = LIDAR_RANGE
                
        return data

    def plot_trajectory(self):
        fig = plt.figure()

        self.plotHalls()

        plt.plot(self.allX, self.allY, 'r--')

        plt.show()

    def plot_lidar(self):
        fig = plt.figure()

        self.plotHalls()

        plt.scatter([self.car_global_x], [self.car_global_y], c = 'red')

        data = self.scan_lidar()

        lidX = []
        lidY = []

        theta_t = np.linspace(-self.lidar_field_of_view, self.lidar_field_of_view, self.lidar_num_rays)

        index = 0

        for curAngle in theta_t:    

            lidX.append(self.car_global_x + data[index] * np.cos(curAngle * np.pi / 180 + self.car_global_heading))
            lidY.append(self.car_global_y + data[index] * np.sin(curAngle * np.pi / 180 + self.car_global_heading))
                          
            index += 1

        plt.scatter(lidX, lidY, c = 'green')

        plt.show()

    def plot_real_lidar(self, data, newfig = True):


        if newfig:
            fig = plt.figure()

            self.plotHalls()

        plt.scatter([self.car_global_x], [self.car_global_y], c = 'red')

        lidX = []
        lidY = []

        theta_t = np.linspace(-self.lidar_field_of_view, self.lidar_field_of_view, self.lidar_num_rays)

        index = 0

        for curAngle in theta_t:    

            lidX.append(self.car_global_x + data[index] * np.cos(curAngle * np.pi / 180 + self.car_global_heading))
            lidY.append(self.car_global_y + data[index] * np.sin(curAngle * np.pi / 180 + self.car_global_heading))
                          
            index += 1

        plt.scatter(lidX, lidY, c = 'green')

        if newfig: 
            plt.show()

    def plotHalls(self, wallwidth = 3):

        # 1st hall going up by default and centralized around origin
        midX = 0
        midY = 0
        going_up = True
        left = True
        
        for i in range(self.numHalls):

            # vertical hallway
            if i % 2 == 0:
                x1 = midX - self.hallWidths[i]/2.0
                x2 = midX - self.hallWidths[i]/2.0
                x3 = midX + self.hallWidths[i]/2.0
                x4 = midX + self.hallWidths[i]/2.0

                # L shape of bottom corner
                
                # Case 1: going down and about to turn left
                if 'left' in self.turns[i] and not going_up:
                    y1 = midY - self.hallLengths[i]/2.0 
                    y3 = midY - self.hallLengths[i]/2.0 + self.hallWidths[(i + 1) % self.numHalls]
                    
                # Case 2: going up and previous turn was right
                elif 'right' in self.turns[i-1] and going_up:
                    y1 = midY - self.hallLengths[i]/2.0 
                    y3 = midY - self.hallLengths[i]/2.0 + self.hallWidths[i-1]

                # _| shape of bottom corner
                # Case 1: going down and about to turn right
                elif 'right' in self.turns[i] and not going_up:
                    y1 = midY - self.hallLengths[i]/2.0 + self.hallWidths[(i + 1) % self.numHalls]
                    y3 = midY - self.hallLengths[i]/2.0

                # Case 2: going up and previous turn was left
                else:
                    y1 = midY - self.hallLengths[i]/2.0 + self.hallWidths[i-1]
                    y3 = midY - self.hallLengths[i]/2.0

                # Gamma shape of top corner
                # Case 1: going up and about to turn right
                if 'right' in self.turns[i] and going_up:
                    y2 = midY + self.hallLengths[i]/2.0 
                    y4 = midY + self.hallLengths[i]/2.0 - self.hallWidths[(i + 1) % self.numHalls]
                # Case 2: going down and previous turn was left 
                elif 'left' in self.turns[i-1] and not going_up:
                    y2 = midY + self.hallLengths[i]/2.0 
                    y4 = midY + self.hallLengths[i]/2.0 - self.hallWidths[i-1]

                # Reverse Gamma shape of top corner
                # Case 1: going up and about to turn left
                elif 'left' in self.turns[i] and going_up:
                    y2 = midY + self.hallLengths[i]/2.0 - self.hallWidths[(i + 1) % self.numHalls]
                    y4 = midY + self.hallLengths[i]/2.0
                # Case 2: going down and previous turn was right
                else:
                    y2 = midY + self.hallLengths[i]/2.0 - self.hallWidths[(i-1)]
                    y4 = midY + self.hallLengths[i]/2.0

                # update coordinates and directions
                if going_up:
                    if 'left' in self.turns[i]:
                        midX = midX - self.hallLengths[(i + 1) % self.numHalls]/2.0 + self.hallWidths[i]/2.0
                        left = True

                    else:
                        midX = midX + self.hallLengths[(i + 1) % self.numHalls]/2.0 - self.hallWidths[i]/2.0
                        left = False
                        
                    midY = midY + self.hallLengths[i]/2.0 - self.hallWidths[(i + 1) % self.numHalls]/2.0
                    

                else:
                    if 'left' in self.turns[i]:
                        midX = midX + self.hallLengths[(i + 1) % self.numHalls]/2.0 - self.hallWidths[i]/2.0
                        left = False

                    else:
                        midX = midX - self.hallLengths[(i + 1) % self.numHalls]/2.0 + self.hallWidths[i]/2.0
                        left = True
                        
                    midY = midY - self.hallLengths[i]/2.0 + self.hallWidths[(i + 1) % self.numHalls]/2.0
                    
            # horizontal hallway    
            else:

                # Gamma shape of left corner
                # Case 1: going right and previous turn was right
                if 'right' in self.turns[i-1] and not left:
                    x1 = midX - self.hallLengths[i]/2.0 
                    x3 = midX - self.hallLengths[i]/2.0 + self.hallWidths[i-1]
                # Case 2: going left and about to turn left
                elif left and 'left' in self.turns[i]:
                    x1 = midX - self.hallLengths[i]/2.0 
                    x3 = midX - self.hallLengths[i]/2.0 + self.hallWidths[(i + 1) % self.numHalls]
                    
                # L shape of left corner
                # Case 1: going right and previous turn was left
                elif 'left' in self.turns[i-1] and not left:
                    x1 = midX - self.hallLengths[i]/2.0 + self.hallWidths[i-1]
                    x3 = midX - self.hallLengths[i]/2.0
                # Case 2: going left and about to turn right
                else:
                    x1 = midX - self.hallLengths[i]/2.0 + self.hallWidths[(i + 1) % self.numHalls]
                    x3 = midX - self.hallLengths[i]/2.0

                    
                # Reverse Gamma shape of right corner
                # Case 1: going right and about to turn right
                if 'right' in self.turns[i] and not left:
                    x2 = midX + self.hallLengths[i]/2.0 
                    x4 = midX + self.hallLengths[i]/2.0 - self.hallWidths[(i + 1) % self.numHalls]
                # Case 2: going left and previous turn was left
                elif 'left' in self.turns[i-1] and left:
                    x2 = midX + self.hallLengths[i]/2.0 
                    x4 = midX + self.hallLengths[i]/2.0 - self.hallWidths[i-1]

                # _| shape of right corner
                # Case 1: going right and about to turn left
                elif 'left' in self.turns[i] and not left:
                    x2 = midX + self.hallLengths[i]/2.0 - self.hallWidths[(i + 1) % self.numHalls]
                    x4 = midX + self.hallLengths[i]/2.0
                # Case 2: going left and previous turn was right
                else:
                    x2 = midX + self.hallLengths[i]/2.0 - self.hallWidths[i-1]
                    x4 = midX + self.hallLengths[i]/2.0

                
                y1 = midY + self.hallWidths[i]/2.0
                y2 = midY + self.hallWidths[i]/2.0
                y3 = midY - self.hallWidths[i]/2.0
                y4 = midY - self.hallWidths[i]/2.0

                # update coordinates and directions
                if left:
                    if 'left' in self.turns[i]:
                        midY = midY - self.hallLengths[(i + 1) % self.numHalls]/2.0 + self.hallWidths[i]/2.0
                        going_up = False
                    else:
                        midY = midY + self.hallLengths[(i + 1) % self.numHalls]/2.0 - self.hallWidths[i]/2.0
                        going_up = True
                        
                    midX = midX - self.hallLengths[i]/2.0 + self.hallWidths[(i + 1) % self.numHalls]/2.0

                else:
                    if 'left' in self.turns[i]:
                        midY = midY + self.hallLengths[(i + 1) % self.numHalls]/2.0 - self.hallWidths[i]/2.0
                        going_up = True

                    else:
                        midY = midY - self.hallLengths[(i + 1) % self.numHalls]/2.0 + self.hallWidths[i]/2.0
                        going_up = False
                    midX = midX + self.hallLengths[i]/2.0 - self.hallWidths[(i + 1) % self.numHalls]/2.0

                    

            l1x = np.array([x1, x2])
            l1y = np.array([y1, y2])
            l2x = np.array([x3, x4])
            l2y = np.array([y3, y4])
            plt.plot(l1x, l1y, 'b', linewidth=wallwidth)
            plt.plot(l2x, l2y, 'b', linewidth=wallwidth)
