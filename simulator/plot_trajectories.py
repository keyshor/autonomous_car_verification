from Car import World
import numpy as np
import matplotlib.pyplot as plt


def normalize(s):
    mean = [2.5]
    spread = [5.0]
    return (s - mean) / spread


def predict(observation, prevErr):
    thresh = 0.005
    pee = 5
    dee = 0.6
    err = errorFunc(observation, thresh)
    result = pee*err + dee*(err - prevErr)
    return result, err


def errorFunc(observation, thresh):
    mid = len(observation)//2
    rightView = observation[0:mid]
    leftView = observation[mid+1:]
    numRight = sum([int(a > thresh) for a in rightView])
    numLeft = sum([int(a > thresh) for a in leftView])
    err = numLeft - numRight
    return err


def main():

    # input_filename = argv[0]

    # model = models.load_model(input_filename)

    numTrajectories = 100

    hallWidths = [1.5, 1.5, 1.5, 1.5]
    hallLengths = [20, 20, 20, 20]
    turns = ['right', 'right', 'right', 'right']
    car_dist_s = hallWidths[0]/2.0
    car_dist_f = 9.9
    car_heading = 0
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
              car_dist_s, car_dist_f, car_heading,
              episode_length, time_step, lidar_field_of_view,
              lidar_num_rays, lidar_noise, missing_lidar_rays, True)

    throttle = 16

    allX = []
    allY = []
    allR = []

    np.random.seed(50)

    # dynamics noise parameters
    x_dynamics_noise = 0
    y_dynamics_noise = 0
    v_dynamics_noise = 0
    theta_dynamics_noise = 0

    # initial uncertainty
    init_pos_noise = 0.2
    init_heading_noise = 0.1

    for step in range(numTrajectories):

        w.reset(pos_noise = init_pos_noise, heading_noise = init_heading_noise)

        init_cond_s = w.car_dist_s
        init_cond_h = w.car_heading

        observation = w.scan_lidar()

        rew = 0

        for e in range(episode_length):

            observation = normalize(observation)

            delta, prev_err = predict(observation, prev_err)
            # 15 * model.predict(observation.reshape(1, len(observation)))

            observation, reward, done, info = w.step(delta, throttle)

            if done:

                if e < episode_length - 1:
                    print(init_cond_s)
                    print(init_cond_h)
                    exit()
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
