import gym
from gym import spaces
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math

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
MAX_THROTTLE = 50 # just used to compute maximum possible velocity

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
                 lidar_num_rays, lidar_noise = 0, lidar_missing_rays = 0,\
                 lidar_missing_in_turn_only = False, state_feedback=False):

        # hallway parameters
        self.numHalls = len(hallWidths)
        self.hallWidths = hallWidths
        self.hallLengths = hallLengths
        self.turns = turns
        self.curHall = 0

        # observation parameter
        self.state_feedback = state_feedback

        # car relative states
        self.car_dist_s = car_dist_s
        self.car_dist_f = car_dist_f
        self.car_V = car_V
        self.car_heading = car_heading

        # car global states
        self.car_global_x = -self.hallWidths[0] / 2.0 + self.car_dist_s
        if self.turns[0] > 0:
            self.car_global_x = -self.car_global_x
        
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

        # coordinates of two corners in turn
        self.cur_hall_heading = np.pi/2
        next_heading = self.cur_hall_heading + self.turns[0]
        if next_heading > np.pi:
            next_heading -= 2 * np.pi
        elif next_heading < -np.pi:
            next_heading += 2 * np.pi
            
        reverse_cur_heading = self.cur_hall_heading - np.pi
        
        if self.turns[0] < 0:
            self.outer_x = -self.hallWidths[0]/2.0
            self.outer_y = self.hallLengths[0]/2.0

        else:
            self.outer_x = self.hallWidths[0]/2.0
            self.outer_y = self.hallLengths[0]/2.0

        out_wall_proj_length = np.abs(self.hallWidths[0] / np.sin(self.turns[0]))
        proj_point_x = self.outer_x + np.cos(next_heading) * out_wall_proj_length
        proj_point_y = self.outer_y + np.sin(next_heading) * out_wall_proj_length

        in_wall_proj_length = np.abs(self.hallWidths[1] / np.sin(self.turns[0]))
        self.inner_x = proj_point_x + np.cos(reverse_cur_heading) * in_wall_proj_length
        self.inner_y = proj_point_y + np.sin(reverse_cur_heading) * in_wall_proj_length            

        # parameters needed for consistency with gym environments
        if state_feedback:
            #self.obs_low = np.array([0, 0, 0, -np.pi])
            #self.obs_high = np.array([max(hallLengths), max(hallLengths), CAR_MOTOR_CONST * (MAX_THROTTLE - HYSTERESIS_CONSTANT), np.pi])
            self.obs_low = np.array([0, 0, -np.pi])
            self.obs_high = np.array([max(hallLengths), max(hallLengths), np.pi])

        else:
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
        if self.turns[0] == np.pi/2:
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

        if self.state_feedback:
            #return np.array([self.car_dist_s, self.car_dist_f, self.car_V, self.car_heading])
            return np.array([self.car_dist_s, self.car_dist_f, self.car_heading])
        else:
            return self.scan_lidar()

    #NB: Mode switches are handled in the step function
    # x := [s, f, V, theta_local, x, y, theta_global]
    def bicycle_dynamics(self, x, t, u, delta, turn):

        if turn < 0: #right turn
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

        if turn < 0: #right turn
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

        if self.direction == UP and self.turns[self.curHall] == -np.pi/2\
           or self.direction == DOWN and self.turns[self.curHall] == np.pi/2:
            new_x[4] = new_x[4] + x_added_noise
            
        elif self.direction == DOWN and self.turns[self.curHall] == -np.pi/2\
             or self.direction == UP and self.turns[self.curHall] == np.pi/2:
            new_x[4] = new_x[4] - x_added_noise
            
        elif self.direction == RIGHT and self.turns[self.curHall] == -np.pi/2\
             or self.direction == LEFT and self.turns[self.curHall] == np.pi/2:
            new_x[4] = new_x[4] - y_added_noise
            
        elif self.direction == LEFT and self.turns[self.curHall] == -np.pi/2\
             or self.direction == RIGHT and self.turns[self.curHall] == np.pi/2:
            new_x[4] = new_x[4] + y_added_noise
        
        new_x[1] = new_x[1] + y_added_noise

        if self.direction == UP and self.turns[self.curHall] == -np.pi/2\
           or self.direction == DOWN and self.turns[self.curHall] == np.pi/2:
            new_x[5] = new_x[5] - y_added_noise
            
        elif self.direction == DOWN and self.turns[self.curHall] == -np.pi/2\
             or self.direction == UP and self.turns[self.curHall] == np.pi/2:
            new_x[5] = new_x[5] + y_added_noise
            
        elif self.direction == RIGHT and self.turns[self.curHall] == -np.pi/2\
             or self.direction == LEFT and self.turns[self.curHall] == np.pi/2:
            new_x[5] = new_x[5] - x_added_noise
            
        elif self.direction == LEFT and self.turns[self.curHall] == -np.pi/2\
             or self.direction == RIGHT and self.turns[self.curHall] == np.pi/2:
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

        # Check for a crash
        corner_angle = np.pi - np.abs(self.turns[self.curHall])
        normal_to_top_wall = [np.sin(corner_angle), -np.cos(corner_angle)]

        # note that dist_f is the x coordinate, and dist_s is the y coordinate
        dot_prod_top = normal_to_top_wall[0] * self.car_dist_f + normal_to_top_wall[1] * self.car_dist_s
        
        if dot_prod_top <= SAFE_DISTANCE or\
           (dot_prod_top >= (self.hallWidths[(self.curHall+1) % self.numHalls] - SAFE_DISTANCE)\
           and self.car_dist_s >= self.hallWidths[(self.curHall) % self.numHalls] - SAFE_DISTANCE) or\
           self.car_dist_s <= SAFE_DISTANCE:
            terminal = True
            reward = CRASH_REWARD

        if self.cur_step == self.episode_length:
            terminal = True

        # Check if a mode switch in the world has changed
        if self.car_dist_s > LIDAR_RANGE:

            # update global hall heading            
            self.cur_hall_heading += self.turns[self.curHall]
            if self.cur_hall_heading > np.pi:
                self.cur_hall_heading -= 2 * np.pi
            elif self.cur_hall_heading < -np.pi:
                self.cur_hall_heading += 2 * np.pi

            flip_sides = False

            # rightish turn
            if self.turns[self.curHall] < 0:

                # update corner coordinates
                if self.turns[(self.curHall+1)%self.numHalls] < 0:
                    flip_sides = False
                    
                else:
                    flip_sides = True

                if self.direction == UP:
                    self.direction = RIGHT
                elif self.direction == RIGHT:
                    self.direction = DOWN
                elif self.direction == DOWN:
                    self.direction = LEFT
                elif self.direction == LEFT:
                    self.direction = UP
    
            else: # left turn 
                # update corner coordinates
                if self.turns[(self.curHall+1)%self.numHalls] > 0:
                    flip_sides = True
                        
                else:
                    flip_sides = False

                if self.direction == UP:
                    self.direction = LEFT
                elif self.direction == RIGHT:
                    self.direction = UP
                elif self.direction == DOWN:
                    self.direction = RIGHT
                elif self.direction == LEFT:
                    self.direction = DOWN

            # update local car states
            dist_to_outer = np.sqrt(self.car_dist_s ** 2 + self.car_dist_f ** 2)
            corner_angle = np.pi - np.abs(self.turns[self.curHall])
            inner_angle = corner_angle - math.atan(self.car_dist_s / np.abs(self.car_dist_f))

            if corner_angle > np.pi/2:
                inner_angle = corner_angle - math.atan(np.abs(self.car_dist_f) / self.car_dist_s) - np.pi/2

            self.car_dist_s = np.sin(inner_angle) * dist_to_outer
            if flip_sides:
                self.car_dist_s = self.hallWidths[(self.curHall+1)%self.numHalls] - self.car_dist_s

            self.car_dist_f = self.hallLengths[(self.curHall+1)%self.numHalls] - np.cos(inner_angle) * dist_to_outer
            
            self.car_heading = self.car_heading - self.turns[self.curHall]

            # update corner coordinates        
            # add the length minus the distance from starting outer to inner corner
            if flip_sides:
                starting_corner_dist = np.sqrt((self.outer_x - self.inner_x) ** 2 + (self.outer_y - self.inner_y) ** 2)
                wall_dist = np.sqrt(starting_corner_dist ** 2 - self.hallWidths[(self.curHall+1)%self.numHalls] ** 2)
                        
                self.outer_x = self.inner_x + np.cos(self.cur_hall_heading) * (self.hallLengths[(self.curHall+1)%self.numHalls] - wall_dist)
                self.outer_y = self.inner_y + np.sin(self.cur_hall_heading) * (self.hallLengths[(self.curHall+1)%self.numHalls] - wall_dist)
            else:
                self.outer_x = self.outer_x + np.cos(self.cur_hall_heading) * self.hallLengths[(self.curHall+1)%self.numHalls]
                self.outer_y = self.outer_y + np.sin(self.cur_hall_heading) * self.hallLengths[(self.curHall+1)%self.numHalls]

            reverse_cur_heading = self.cur_hall_heading - np.pi
            if reverse_cur_heading > np.pi:
                reverse_cur_heading -= 2 * np.pi
            elif reverse_cur_heading < -np.pi:
                reverse_cur_heading += 2 * np.pi

            next_heading = self.cur_hall_heading + self.turns[(self.curHall+1)%self.numHalls]
            if next_heading > np.pi:
                next_heading -= 2 * np.pi
            elif next_heading < -np.pi:
                next_heading += 2 * np.pi
                
            out_wall_proj_length = np.abs(self.hallWidths[self.curHall] / np.sin(self.turns[(self.curHall+1)%self.numHalls]))
            proj_point_x = self.outer_x + np.cos(next_heading) * out_wall_proj_length
            proj_point_y = self.outer_y + np.sin(next_heading) * out_wall_proj_length

            in_wall_proj_length = np.abs(self.hallWidths[(self.curHall+1)%self.numHalls] / np.sin(self.turns[(self.curHall+1)%self.numHalls]))
            self.inner_x = proj_point_x + np.cos(reverse_cur_heading) * in_wall_proj_length
            self.inner_y = proj_point_y + np.sin(reverse_cur_heading) * in_wall_proj_length

            # update hall index
            self.curHall = self.curHall + 1 # next hallway
            #NB: this case deals with loops in the environment
            if self.curHall >= self.numHalls:
                self.curHall = 0                    


        self.allX.append(self.car_global_x)
        self.allY.append(self.car_global_y)

        if self.state_feedback:
            #return np.array([self.car_dist_s, self.car_dist_f, self.car_V, self.car_heading]), reward, terminal, -1
            return np.array([self.car_dist_s, self.car_dist_f, self.car_heading]), reward, terminal, -1
        else:
            return self.scan_lidar(), reward, terminal, -1

    def scan_lidar(self):

        car_heading_deg = self.car_heading * 180 / np.pi

        theta_t = np.linspace(-self.lidar_field_of_view, self.lidar_field_of_view, self.lidar_num_rays)

        # lidar measurements
        data = np.zeros(len(theta_t))

        corner_dist = np.sqrt((self.outer_x - self.inner_x) ** 2 + (self.outer_y - self.inner_y) ** 2)
        wall_dist = np.sqrt(corner_dist ** 2 - self.hallWidths[self.curHall] ** 2)

        car_dist_s_inner = self.hallWidths[self.curHall] - self.car_dist_s
        car_dist_f_inner = self.car_dist_f - wall_dist

        region1 = False
        region2 = False
        region3 = False

        if car_dist_f_inner >= 0:

            theta_outer = np.arctan(float(self.car_dist_s) / self.car_dist_f) * 180 / np.pi

            if self.car_dist_s <= self.hallWidths[self.curHall]:
                
                theta_inner = -np.arctan(float(car_dist_s_inner) / car_dist_f_inner) * 180 / np.pi

                if np.abs(theta_inner) <= np.abs(self.turns[self.curHall]) * 180 / np.pi:
                    region1 = True
                else:
                    region2 = True
            else:

                car_dist_s_inner = self.car_dist_s - self.hallWidths[self.curHall]
                theta_inner = np.arctan(float(car_dist_s_inner) / car_dist_f_inner) * 180 / np.pi
                region3 = True
        else:

            car_dist_f_inner = np.abs(car_dist_f_inner)
            
            if car_dist_s_inner >= 0:
                theta_inner = -90 - np.arctan(float(car_dist_f_inner) / car_dist_s_inner) * 180 / np.pi
                region2 = True
            else:
                car_dist_s_inner = np.abs(car_dist_s_inner)
                theta_inner = 90 + np.arctan(float(car_dist_f_inner) / car_dist_s_inner) * 180 / np.pi
                region3 = True

            if self.car_dist_f >= 0:
                car_dist_f_outer = self.car_dist_f
                theta_outer = np.arctan(float(self.car_dist_s) / self.car_dist_f) * 180 / np.pi
            else:
                car_dist_f_outer = np.abs(self.car_dist_f)
                theta_outer = 90 + np.arctan(np.abs(self.car_dist_f) / float(self.car_dist_s)) * 180 / np.pi

        car_dist_s_outer = self.car_dist_s

        if self.turns[self.curHall] < 0:
            dist_l = car_dist_s_outer
            dist_r = car_dist_s_inner
            theta_l = theta_outer
            theta_r = theta_inner
            
        else:
            dist_l = car_dist_s_inner
            dist_r = car_dist_s_outer
            theta_l = -theta_inner
            theta_r = -theta_outer

        # Region 1 (before turn)
        if region1:

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
                    dist_to_outer = np.sqrt(self.car_dist_s ** 2 + self.car_dist_f ** 2)
                    if self.turns[self.curHall] < 0:
                        outer_angle = -self.turns[self.curHall] + theta_l * np.pi / 180
                    else:
                        outer_angle = -self.turns[self.curHall] + theta_r * np.pi / 180
                    dist_to_top_wall = dist_to_outer * np.sin(outer_angle)
                    
                    data[index] = dist_to_top_wall /\
                                  np.sin(-self.turns[self.curHall] + angle * np.pi / 180)

                else:
                    
                    data[index] = (dist_l) /\
                            (np.cos( (90 - angle) * np.pi / 180))

                #add noise
                data[index] += np.random.uniform(0, self.lidar_noise)

                if data[index] > LIDAR_RANGE or data[index] < 0:
                    data[index] = LIDAR_RANGE
                    
                index += 1

        # Region 2 (during turn)
        elif region2:
                
            index = 0
            
            for angle in theta_t:

                angle = angle + car_heading_deg
                if angle > 180:
                    angle = angle - 360
                elif angle < -180:
                    angle = angle + 360
                    
                if self.turns[self.curHall] < 0:
                    if angle <= theta_r:
                        data[index] = (dist_r) /\
                                (np.cos( (90 + angle) * np.pi / 180))

                    elif angle > theta_r and angle < self.turns[self.curHall] * 180 / np.pi:
                        dist_to_outer = np.sqrt(self.car_dist_s ** 2 + self.car_dist_f ** 2)                    
                        outer_angle = -self.turns[self.curHall] + theta_outer * np.pi / 180
                        dist_to_bottom_wall = self.hallWidths[(self.curHall+1) % self.numHalls] - dist_to_outer * np.sin(outer_angle)
                        
                        data[index] = dist_to_bottom_wall /\
                                  np.cos(np.pi/2 - self.turns[self.curHall] + angle * np.pi / 180)

                    elif angle > self.turns[self.curHall] * 180 / np.pi and angle <= theta_l:
                        dist_to_outer = np.sqrt(self.car_dist_s ** 2 + self.car_dist_f ** 2)
                        outer_angle = -self.turns[self.curHall] + theta_outer * np.pi / 180
                        dist_to_top_wall = dist_to_outer * np.sin(outer_angle)
                    
                        data[index] = dist_to_top_wall /\
                                  np.cos(np.pi/2 + self.turns[self.curHall] - angle * np.pi / 180)                        
                    else:
                        data[index] = (dist_l) /\
                                (np.cos( (90 - angle) * np.pi / 180))

                else:
                    if angle <= theta_r:
                        data[index] = (dist_r) /\
                                (np.cos( (90 + angle) * np.pi / 180))

                    elif angle > theta_r and angle <= self.turns[self.curHall] * 180 / np.pi:
                        dist_to_outer = np.sqrt(self.car_dist_s ** 2 + self.car_dist_f ** 2)
                        outer_angle = self.turns[self.curHall] - theta_r * np.pi / 180
                        dist_to_top_wall = dist_to_outer * np.sin(outer_angle)
                    
                        data[index] = dist_to_top_wall /\
                                  np.sin(np.pi - self.turns[self.curHall] + angle * np.pi / 180)   

                    elif angle > self.turns[self.curHall] * 180 / np.pi and angle <= theta_l:
                        dist_to_outer = np.sqrt(self.car_dist_s ** 2 + self.car_dist_f ** 2)                    
                        outer_angle = self.turns[self.curHall] - theta_r * np.pi / 180
                        dist_to_bottom_wall = self.hallWidths[(self.curHall+1) % self.numHalls] - dist_to_outer * np.sin(outer_angle)
                        
                        data[index] = dist_to_bottom_wall /\
                                  np.cos(np.pi/2 + self.turns[self.curHall] - angle * np.pi / 180)
                    else:
                        data[index] = (dist_l) /\
                                (np.cos( (90 - angle) * np.pi / 180))

                #add noise
                data[index] += np.random.uniform(0, self.lidar_noise)

                if data[index] > LIDAR_RANGE or data[index] < 0:
                    data[index] = LIDAR_RANGE

                index += 1

        # Region 3 (after turn)
        elif region3:

            index = 0

            for angle in theta_t:

                angle = angle + car_heading_deg
                if angle > 180:
                    angle = angle - 360
                elif angle < -180:
                    angle = angle + 360
                    
                if self.turns[self.curHall] < 0:
                    if angle < self.turns[self.curHall] * 180 / np.pi:
                        dist_to_outer = np.sqrt(self.car_dist_s ** 2 + self.car_dist_f ** 2)                    
                        outer_angle = -self.turns[self.curHall] + theta_outer * np.pi / 180
                        dist_to_bottom_wall = self.hallWidths[(self.curHall+1) % self.numHalls] - dist_to_outer * np.sin(outer_angle)
                        
                        data[index] = dist_to_bottom_wall /\
                                  np.cos(np.pi/2 - self.turns[self.curHall] + angle * np.pi / 180)

                    elif angle == self.turns[self.curHall] * 180 / np.pi:
                          data[index] = LIDAR_RANGE

                    elif angle >= self.turns[self.curHall] * 180 / np.pi and angle <= theta_l:
                        dist_to_outer = np.sqrt(self.car_dist_s ** 2 + self.car_dist_f ** 2)
                        outer_angle = -self.turns[self.curHall] + theta_outer * np.pi / 180
                        dist_to_top_wall = dist_to_outer * np.sin(outer_angle)
                    
                        data[index] = dist_to_top_wall /\
                                  np.cos(np.pi/2 + self.turns[self.curHall] - angle * np.pi / 180)       

                    elif angle > theta_l and angle <= theta_r:
                        data[index] = (dist_l) /\
                                (np.cos( (90 - angle) * np.pi / 180))

                    else:
                        dist_to_outer = np.sqrt(self.car_dist_s ** 2 + self.car_dist_f ** 2)                    
                        outer_angle = -self.turns[self.curHall] + theta_outer * np.pi / 180
                        dist_to_bottom_wall = self.hallWidths[(self.curHall+1) % self.numHalls] - dist_to_outer * np.sin(outer_angle)
                        
                        data[index] = dist_to_bottom_wall /\
                                  np.cos(np.pi/2 - self.turns[self.curHall] + angle * np.pi / 180)
                else:
                    if angle >= self.turns[self.curHall] * 180 / np.pi:
                        dist_to_outer = np.sqrt(self.car_dist_s ** 2 + self.car_dist_f ** 2)                    
                        outer_angle = -self.turns[self.curHall] + theta_r * np.pi / 180
                        dist_to_bottom_wall = self.hallWidths[(self.curHall+1) % self.numHalls] - dist_to_outer * np.sin(outer_angle)
                        
                        data[index] = dist_to_bottom_wall /\
                                  np.cos(np.pi/2 + self.turns[self.curHall] - angle * np.pi / 180)

                    elif angle < self.turns[self.curHall] * 180 / np.pi and angle >= theta_r:
                        dist_to_outer = np.sqrt(self.car_dist_s ** 2 + self.car_dist_f ** 2)
                        outer_angle = -self.turns[self.curHall] + theta_r * np.pi / 180
                        dist_to_top_wall = dist_to_outer * np.sin(outer_angle)
                    
                        data[index] = dist_to_top_wall /\
                                  np.sin(np.pi - self.turns[self.curHall] + angle * np.pi / 180)  

                    elif angle < theta_r and angle >= theta_l:
                        data[index] = (dist_r) /\
                                (np.cos( (90 + angle) * np.pi / 180))

                    else:

                        dist_to_outer = np.sqrt(self.car_dist_s ** 2 + self.car_dist_f ** 2)                    
                        outer_angle = -self.turns[self.curHall] + theta_outer * np.pi / 180
                        dist_to_bottom_wall = self.hallWidths[(self.curHall+1) % self.numHalls] - dist_to_outer * np.sin(outer_angle)
                        
                        data[index] = dist_to_bottom_wall /\
                                  np.cos(np.pi/2 + self.turns[self.curHall] - angle * np.pi / 180)

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

        #plt.ylim((-1,11))
        #plt.xlim((-2, np.max(self.hallLengths) + np.max(self.hallWidths)))

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

        cur_heading = np.pi / 2
            
        for i in range(self.numHalls):

            # set up starting and ending outer corners
            
            # the starting shape of the first hallway will assume a
            # loop (if the hallways do not form a loop, it will just
            # look non-symmetrical)
            if i == 0:
                if self.turns[-1] < 0:
                    l1x1 = -self.hallWidths[0]/2.0
                    l1y1 = -self.hallLengths[0]/2.0

                else:
                    l2x1 = self.hallWidths[0]/2.0
                    l2y1 = -self.hallLengths[0]/2.0

                if self.turns[0] < 0:
                    l1x2 = -self.hallWidths[0]/2.0
                    l1y2 = self.hallLengths[0]/2.0

                else:
                    l2x2 = self.hallWidths[0]/2.0
                    l2y2 = self.hallLengths[0]/2.0
            else:
                if self.turns[i-1] < 0:

                    l1x1 = prev_outer_x
                    l1y1 = prev_outer_y

                    if self.turns[i] < 0:
                        
                        l1x2 = l1x1 + np.cos(cur_heading) * self.hallLengths[i]
                        l1y2 = l1y1 + np.sin(cur_heading) * self.hallLengths[i]
                    # add the length minus the distance from starting outer to inner corner
                    else:
                        l2x2 = prev_inner_x + np.cos(cur_heading) * (self.hallLengths[i] - wall_dist)
                        l2y2 = prev_inner_y + np.sin(cur_heading) * (self.hallLengths[i] - wall_dist)

                else:
                    l2x1 = prev_outer_x
                    l2y1 = prev_outer_y

                    # add the length minus the distance from starting outer to inner corner
                    if self.turns[i] < 0:
                        l1x2 = prev_inner_x + np.cos(cur_heading) * (self.hallLengths[i] - wall_dist)
                        l1y2 = prev_inner_y + np.sin(cur_heading) * (self.hallLengths[i] - wall_dist)

                    else:

                        l2x2 = l2x1 + np.cos(cur_heading) * self.hallLengths[i]
                        l2y2 = l2y1 + np.sin(cur_heading) * self.hallLengths[i]

            prev_heading = cur_heading - self.turns[i-1]
            reverse_prev_heading = prev_heading - np.pi

            if reverse_prev_heading > np.pi:
                reverse_prev_heading -= 2 * np.pi
            elif reverse_prev_heading < -np.pi:
                reverse_prev_heading += 2 * np.pi

            next_heading = cur_heading + self.turns[i]
            if next_heading > np.pi:
                next_heading -= 2 * np.pi
            elif next_heading < -np.pi:
                next_heading += 2 * np.pi
            
            reverse_cur_heading = cur_heading - np.pi

            if reverse_cur_heading > np.pi:
                reverse_cur_heading -= 2 * np.pi
            elif reverse_cur_heading < -np.pi:
                reverse_cur_heading += 2 * np.pi

            # rightish turn coming into the current turn (L shape)
            if self.turns[i-1] < 0:

                in_wall_proj_length = np.abs(self.hallWidths[i] / np.sin(self.turns[i-1]))
                proj_point_x = l1x1 + np.cos(reverse_prev_heading) * in_wall_proj_length
                proj_point_y = l1y1 + np.sin(reverse_prev_heading) * in_wall_proj_length

                out_wall_proj_length = np.abs(self.hallWidths[i-1] / np.sin(self.turns[i-1]))
                l2x1 = proj_point_x + np.cos(cur_heading) * out_wall_proj_length
                l2y1 = proj_point_y + np.sin(cur_heading) * out_wall_proj_length

            # _| shape
            else:

                in_wall_proj_length = np.abs(self.hallWidths[i] / np.sin(self.turns[i-1]))
                proj_point_x = l2x1 + np.cos(reverse_prev_heading) * in_wall_proj_length
                proj_point_y = l2y1 + np.sin(reverse_prev_heading) * in_wall_proj_length

                out_wall_proj_length = np.abs(self.hallWidths[i-1] / np.sin(self.turns[i-1]))
                l1x1 = proj_point_x + np.cos(cur_heading) * out_wall_proj_length
                l1y1 = proj_point_y + np.sin(cur_heading) * out_wall_proj_length

            # rightish turn going out of the current turn (Gamma shape)
            next_ind = i+1
            if next_ind >= self.numHalls:
                next_ind = 0
            if self.turns[i] < 0:

                out_wall_proj_length = np.abs(self.hallWidths[i] / np.sin(self.turns[next_ind]))
                proj_point_x = l1x2 + np.cos(next_heading) * out_wall_proj_length
                proj_point_y = l1y2 + np.sin(next_heading) * out_wall_proj_length

                in_wall_proj_length = np.abs(self.hallWidths[next_ind] / np.sin(self.turns[next_ind]))
                l2x2 = proj_point_x + np.cos(reverse_cur_heading) * in_wall_proj_length
                l2y2 = proj_point_y + np.sin(reverse_cur_heading) * in_wall_proj_length

                # update next outer corner
                prev_outer_x = l1x2
                prev_outer_y = l1y2

                prev_inner_x = l2x2
                prev_inner_y = l2y2

            else:

                out_wall_proj_length = np.abs(self.hallWidths[i] / np.sin(self.turns[next_ind]))
                proj_point_x = l2x2 + np.cos(next_heading) * out_wall_proj_length
                proj_point_y = l2y2 + np.sin(next_heading) * out_wall_proj_length

                in_wall_proj_length = np.abs(self.hallWidths[next_ind] / np.sin(self.turns[next_ind]))
                l1x2 = proj_point_x + np.cos(reverse_cur_heading) * in_wall_proj_length
                l1y2 = proj_point_y + np.sin(reverse_cur_heading) * in_wall_proj_length

                # update next outer corner
                prev_outer_x = l2x2
                prev_outer_y = l2y2

                prev_inner_x = l1x2
                prev_inner_y = l1y2

            starting_corner_dist = np.sqrt((l1x1 - l2x1) ** 2 + (l1y1 - l2y1) ** 2)
            wall_dist = np.sqrt(starting_corner_dist ** 2 - self.hallWidths[i] ** 2)

            cur_heading = next_heading

            l1x = np.array([l1x1, l1x2])
            l1y = np.array([l1y1, l1y2])
            l2x = np.array([l2x1, l2x2])
            l2y = np.array([l2y1, l2y2])
            plt.plot(l1x, l1y, 'b', linewidth=wallwidth)
            plt.plot(l2x, l2y, 'b', linewidth=wallwidth)



def square_hall_right():

    hallWidths = [1.5, 1.5, 1.5, 1.5]
    hallLengths = [20, 20, 20, 20]
    turns = [-np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2]

    return (hallWidths, hallLengths, turns)

def square_hall_left():

    hallWidths = [1.5, 1.5, 1.5, 1.5]
    hallLengths = [20, 20, 20, 20]
    turns = [np.pi/2, np.pi/2, np.pi/2, np.pi/2]

    return (hallWidths, hallLengths, turns)

def trapezoid_hall_sharp_right(width=1.5):

    hallWidths = [width, width, width, width]
    hallLengths = [20 + 2 * np.sqrt(200), 20, 20, 20]
    turns = [(-3 * np.pi) / 4, -np.pi/4, -np.pi/4, (-3 * np.pi)/4]

    return (hallWidths, hallLengths, turns)

def trapezoid_hall_sharp_left():

    hallWidths = [1.5, 1.5, 1.5, 1.5]
    hallLengths = [20 + 2 * np.sqrt(200), 20, 20, 20]
    turns = [(3 * np.pi) / 4, np.pi/4, np.pi/4, (3 * np.pi)/4]

    return (hallWidths, hallLengths, turns)

def trapezoid_hall_slight_right():

    hallWidths = [1.5, 1.5, 1.5, 1.5]
    hallLengths = [20, 20, 20 + 2 * np.sqrt(200), 20]
    turns = [-np.pi/4, (-3 * np.pi) / 4,  (-3 * np.pi)/4, -np.pi/4]    

    return (hallWidths, hallLengths, turns)            
