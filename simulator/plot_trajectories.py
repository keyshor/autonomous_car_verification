from Car import World
from Car import square_hall_right
import numpy as np
import matplotlib.pyplot as plt
from controller2 import Controller

def normalize(s):
    mean = [2.5]
    spread = [5.0]
    return (s - mean) / spread

def main():

    params = [14, 0.0, 3]  # pee, eye, dee, thresh
    model = Controller(params)

    numTrajectories = 100

    (hallWidths, hallLengths, turns) = square_hall_right()
    
    car_dist_s = hallWidths[0]/2.0
    car_dist_f = 6.5
    car_heading = 0
    car_V = 2.4
    episode_length = 100
    time_step = 0.1

    lidar_field_of_view = 115
    lidar_num_rays = 21

    # Change this to 0.1 or 0.2 to generate Figure 3 or 5 in the paper, respectively
    lidar_noise = 0

    # Change this to 0 or 5 to generate Figure 3 or 5 in the paper, respectively
    missing_lidar_rays = 0

    num_unsafe = 0
    prev_err = 0

    w = World(hallWidths, hallLengths, turns,
              car_dist_s, car_dist_f, car_heading, car_V,
              episode_length, time_step, lidar_field_of_view,
              lidar_num_rays, lidar_noise, missing_lidar_rays, True)

    throttle = 16

    allX = []
    allY = []
    allR = []

    np.random.seed(500)

    # dynamics noise parameters
    x_dynamics_noise = 0
    y_dynamics_noise = 0
    v_dynamics_noise = 0
    theta_dynamics_noise = 0

    # initial uncertainty
    init_pos_noise = 0.2
    init_heading_noise = 0.02

    for step in range(numTrajectories):

        w.reset(pos_noise = init_pos_noise, heading_noise = init_heading_noise)

        init_cond_s = w.car_dist_s
        init_cond_h = w.car_heading

        observation = w.scan_lidar()

        rew = 0

        for e in range(episode_length):

            observation = normalize(observation)

            #delta, prev_err = predict(observation, prev_err)
            # 15 * model.predict(observation.reshape(1, len(observation)))

            delta = model.predict(observation.reshape(1, len(observation)))

            delta = np.clip(delta, -15, 15)

            observation, reward, done, info = w.step(delta, throttle)

            if done:

                if e < episode_length - 1:
                    num_unsafe += 1

                break

            rew += reward

        allX.append(w.allX)
        allY.append(w.allY)
        allR.append(rew)

    print(np.mean(allR))
    print('number of crashes: ' + str(num_unsafe))

    plt.figure(figsize=(12, 10))
    w.plotHalls()

    plt.ylim((-1, 11))
    plt.xlim((-1.75, 15))
    plt.tick_params(labelsize=20)

    for i in range(numTrajectories):
        plt.plot(allX[i], allY[i], 'r-')

    plt.show()


if __name__ == '__main__':
    main()
