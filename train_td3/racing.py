import sys
sys.path.append('../simulator')

from Car import World
from Car import CONST_THROTTLE
from Car import CAR_MOTOR_CONST
from Car import HYSTERESIS_CONSTANT
from Car import trapezoid_hall_sharp_right
from Car import square_hall_right
from Car import complex_track
from Car import square_hall_left
from Car import triangle_hall_sharp_right
from Car import triangle_hall_equilateral_right
from Car import triangle_hall_equilateral_left
import numpy as np
import torch
import gym
import argparse
import os

import utils
import TD3

CONST_THROTTLE = 10

SHARP_RIGHT_TURN = 0
SHARP_LEFT_TURN = 1
RIGHT_TURN = 2
LEFT_TURN = 3
COMPLEX_TRACK = 4
ALL_TURNS = 5

def normalize(s, state_feedback=False):
    mean = [2.5]
    spread = [5.0]

    if state_feedback:
        return s
    
    return (s - mean) / spread

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env, seed, eval_episodes=10, state_feedback=False):
    eval_env = env
    #eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False

        while not done:

            state = normalize(state, state_feedback)
            
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    #print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("Evaluation over " + str(eval_episodes) + " episodes: " + str(avg_reward))
    print("---------------------------------------")
    return avg_reward


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="Pendulum-v0")             # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e4, type=int) # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=2e5, type=int)   # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()

    #file_name = f"{args.policy}_{args.env}_{args.seed}"
    file_name = str(args.policy) + '_racing_' + str(args.seed)
    
    print("---------------------------------------")
    #print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    state_feedback = True
    turn = COMPLEX_TRACK

    episode_length = 80
        
    if turn == RIGHT_TURN:
        (hallWidths, hallLengths, turns) = square_hall_right(1.5)
        name = 'right_turn_'
        car_dist_f = 7
    elif turn == LEFT_TURN:
        (hallWidths, hallLengths, turns) = square_hall_left(1.5)
        name = 'left_turn_'
        car_dist_f = 7        
    elif turn == SHARP_RIGHT_TURN:
        (hallWidths, hallLengths, turns) = triangle_hall_equilateral_right(1.5)
        name = 'sharp_right_turn_'
        car_dist_f = 8
    elif turn == SHARP_LEFT_TURN:
        (hallWidths, hallLengths, turns) = triangle_hall_equilateral_left(1.5)
        name = 'sharp_left_turn_'
        car_dist_f = 8
    elif turn == COMPLEX_TRACK:
        (hallWidths, hallLengths, turns) = complex_track(1.5)
        name = 'complex_track_'
        car_dist_f = 8
        episode_length = 400
        
    elif turn == ALL_TURNS:
        (hallWidths, hallLengths, turns) = triangle_hall_equilateral_right(1.5)
        (hallWidths2, hallLengths2, turns2) = square_hall_right(1.5)
        (hallWidths3, hallLengths3, turns3) = square_hall_left(1.5)
        (hallWidths4, hallLengths4, turns4) = triangle_hall_equilateral_left(1.5)
        name = 'both_turns_'
        car_dist_f = 8
    
    car_dist_s = hallWidths[0]/2.0
    car_heading = 0
    car_V = 2.4
    time_step = 0.1
    time = 0



    lidar_field_of_view = 115

    lidar_num_rays = 21
    lidar_noise = 0#0.1 #m
    missing_lidar_rays = 0
    
    env = World(hallWidths, hallLengths, turns,\
                car_dist_s, car_dist_f, car_heading, car_V,\
                episode_length, time_step, lidar_field_of_view,\
                lidar_num_rays, lidar_noise, missing_lidar_rays, state_feedback=state_feedback)

    if turn == ALL_TURNS:
        env2 = World(hallWidths2, hallLengths2, turns2,\
                    car_dist_s, car_dist_f, car_heading, car_V,\
                    episode_length, time_step, lidar_field_of_view,\
                    lidar_num_rays, lidar_noise, missing_lidar_rays, state_feedback=state_feedback)

        env3 = World(hallWidths3, hallLengths3, turns3,\
                     car_dist_s, car_dist_f, car_heading, car_V,\
                     episode_length, time_step, lidar_field_of_view,\
                     lidar_num_rays, lidar_noise, missing_lidar_rays, state_feedback=state_feedback)

        env4 = World(hallWidths4, hallLengths4, turns4,\
                     car_dist_s, car_dist_f, car_heading, car_V,\
                     episode_length, time_step, lidar_field_of_view,\
                     lidar_num_rays, lidar_noise, missing_lidar_rays, state_feedback=state_feedback)

        env2.action_space.seed(args.seed)
        env3.action_space.seed(args.seed)
        env4.action_space.seed(args.seed)

    test_env = World(hallWidths, hallLengths, turns,\
                     car_dist_s, car_dist_f, car_heading, car_V,\
                     episode_length, time_step, lidar_field_of_view,\
                     lidar_num_rays, lidar_noise, missing_lidar_rays, state_feedback=state_feedback)    
        

    #env = gym.make(args.env)

    # Set seeds
    env.action_space.seed(args.seed)
    test_env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    # Initialize policy
    if args.policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3.TD3(**kwargs)
    elif args.policy == "OurDDPG":
        policy = OurDDPG.DDPG(**kwargs)
    elif args.policy == "DDPG":
        policy = DDPG.DDPG(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        #policy.load(f"./models/{policy_file}")
        policy.load("./models/'" + str(policy_file))

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    
    # Evaluate untrained policy
    evaluations = [eval_policy(policy, test_env, args.seed, state_feedback=state_feedback)]

    state, done = env.reset(), False
    state = normalize(state, state_feedback)
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    env1 = 0
    cur_env = env

    for t in range(int(args.max_timesteps)):
        
        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = cur_env.action_space.sample()

            # this is a hack to improve exploration
            # if state_feedback and state[2] < 3:
            #    action = -15
        else:
            action = (
                policy.select_action(np.array(state))
                + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = cur_env.step(action)
        next_state = normalize(next_state, state_feedback)
        done_bool = float(done) if episode_timesteps < cur_env._max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            #print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            print("Total T: " + str(t+1) + " Episode Num: " + str(episode_num+1) +
                  " Episode T: " + str(episode_timesteps) +" Reward: " + str(episode_reward))

            if turn == ALL_TURNS:
                if env1 == 0:
                    cur_env = env
                elif env1 == 1:
                    cur_env = env2
                elif env1 == 2:
                    cur_env = env3
                else:
                    cur_env = env4

                env1 = (env1 + 1) % 4

            # Reset environment
            state, done = cur_env.reset(), False
            state = normalize(state, state_feedback)
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1 

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, test_env, args.seed, state_feedback=state_feedback))
            np.save("./results/" + str(file_name), evaluations)
            if state_feedback:
                policy.save("./models_state_feedback/tanh_" + name + \
                        str(env.observation_space.shape[0]) + "_m" + str(missing_lidar_rays) + '_')
            else:
                policy.save("./models/tanh_" + name + \
                            str(env.observation_space.shape[0]) + "_m" + str(missing_lidar_rays) + '_')
                
