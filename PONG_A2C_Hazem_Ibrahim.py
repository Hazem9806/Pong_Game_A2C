# %% [code]
import time
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym # openAi gym
#from gym import envs
from IPython.display import Image
import os
from IPython import display
import time
from matplotlib import pyplot as plt
import cv2
import gym
import gym.spaces
from torch.distributions import Categorical
from itertools import count

import argparse
import csv

from datetime import datetime

# %% [markdown]
#     Pong require a user to press the FIRE button to start the game.
#     The following code corresponds to the wrapper FireResetEnvthat presses the FIRE button in 
#     environments that require that for the game to start.
#     
#     In addition to pressing FIRE, this wrapper checks for several corner cases that are present in some games.

# %% [code]
class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

# %% [markdown]
# The next wrapper that we will require is MaxAndSkipEnv that codes a couple of important transformations for Pong

# %% [code]
class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip
    def step(self, action):
            total_reward = 0.0
            done = None
            for _ in range(self._skip):
              obs, reward, done, info = self.env.step(action)
              self._obs_buffer.append(obs)
              total_reward += reward
              if done:
                  break
            max_frame = np.max(np.stack(self._obs_buffer), axis=0)
            return max_frame, total_reward, done, info
    def reset(self):
          self._obs_buffer.clear()
          obs = self.env.reset()
          self._obs_buffer.append(obs)
          return obs

# %% [code]
class ScaledFloatFrame(gym.ObservationWrapper):
    """Normalize pixel values in frame --> 0 to 1"""
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0

class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
                                                old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer

# %% [code]
class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.float32)

    def observation(self, observation):
      ## moveaxis --> Move axes of an array to new positions
      ## as the Conv2d takes the image argument as (channels,height, width)
        return np.moveaxis(observation, 2, 0)

# %% [code]
def make_env(env_name):
    env = gym.make(env_name)
    env = MaxAndSkipEnv(env)
    env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    return ScaledFloatFrame(env)

# %% [code]
class PolicyNet(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(PolicyNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.policy = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
#        print(',-- torch.zeros(1, *shape): ',torch.zeros(1, *shape), ', -- shape: ', shape, ', \nreturn data: ',int(np.prod(o.size())))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return Categorical(torch.softmax(self.policy(conv_out), dim=1))

# %% [code]
class CriticNet(nn.Module):
    def __init__(self, input_shape):
        super(CriticNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.critic = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.critic(conv_out)

# %% [markdown]
#     Before feeding the frames to the neural network every frame is scaled down from 210x160, 
#     with three color frames (RGB color channels), 
#     to a single-color 84 x84 image using a colorimetric grayscale conversion. 
#     Different approaches are possible. 
#     One of them is cropping non-relevant parts of the image and then scaling down

# %% [code]
class ProcessFrame84(gym.ObservationWrapper):
    """
    Downsamples image to 84x84
    Greyscales image
    Returns numpy array
    """
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            rgb_img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            rgb_img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        
#        cv2.imwrite('./rgb_image2.png', rgb_img)
        ## Image Resizing to crop the not needed space
        rgb_img = cv2.resize(rgb_img, (84, 110), interpolation=cv2.INTER_AREA)
        rgb_img = rgb_img[18:102, :]
        
        ## Conversion from RGB to Gray Scale --> b as [0.2989, 0.5870, 0.1140]
        grayscale_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)

#        Gray_Scale_Parameters = [0.2989, 0.5870, 0.1140]
#        grayscale_img = rgb_img[:, :, 0] * Gray_Scale_Parameters[0] + \
#              rgb_img[:, :, 1] * Gray_Scale_Parameters[1] + \
#              rgb_img[:, :, 2] * Gray_Scale_Parameters[2]
        
        ## Conversion from Gray Scale to Binary --> thershold as 127
        thresh = 127
        binary_img = cv2.threshold(grayscale_img, thresh, 255, cv2.THRESH_BINARY)[1]

        ## Image Resizing to crop the not needed space
#        binary_img = cv2.resize(binary_img, (84, 110), interpolation=cv2.INTER_AREA)
#        binary_img = binary_img[18:102, :]
        binary_img = np.reshape(binary_img, [84, 84, 1])
        
        
        ## Saving some sample image of each conversinon
        save_sample_images = False
        if(save_sample_images):
            cv2.imwrite('./rgb_image.png', rgb_img)
            cv2.imwrite('./grayscale_image.png', grayscale_img)
            cv2.imwrite('./binary_image.png', binary_img)
        
        return binary_img.astype(np.uint8)

# %% [code]
## function that takes a list of rewards and reutrn the list of returns for each step
def discounted_returns(rewards, gamma=0.9):
    ## Init R
    R = 0
    returns = list()
    for reward in reversed(rewards):
        R = reward + gamma * R
        #print(R)
        returns.insert(0, R)
        #returns.append(R)

    returns = torch.tensor(returns)
    
    ## normalize the returns
    returns = (returns - returns.mean()) / (returns.std() + 1e-6)
    return returns

# %% [code]
ENVIROMENT_NAME = 'PongNoFrameskip-v4'

##################################################################
##                   HYPERPARAMETERS                            ##
##################################################################
GAMMA = 0.99

# %% [code]
env = make_env(ENVIROMENT_NAME)

#observations = env.reset()

GPU = True
LOAD_MODEL = True 

## RUN MODE is Either to Train or To Play with Trained Model
PLAY = 1
TRAIN = 0
while(True):
    user_input = input("Please Select the Mode for the Model:\n"\
                       "-------------------------------------\n"\
                       "  Train = 0 \n"\
                       "  Play  = 1\n\n")
    if(int(user_input) == TRAIN):
        RUN_MODE = TRAIN
        break
    elif(int(user_input) == PLAY):
        RUN_MODE = PLAY
        break
    else:
        print("Please Enter Valid Choice")
    
device = ("cuda" if GPU else "cpu")
POLICY_LOAD_PATH = 'policy_17.pt'
CRITIC_LOAD_PATH = 'critic_17.pt'


#print('env.observation_space.shape: ',env.observation_space.shape)

##################################################################
##        Actor & Critic Neural Networks Model                  ##
##################################################################
policy_nn = PolicyNet(env.observation_space.shape, env.action_space.n).to(device)
critic_nn = CriticNet(env.observation_space.shape).to(device)

if(LOAD_MODEL):
    policy_nn.load_state_dict(torch.load(POLICY_LOAD_PATH))
    critic_nn.load_state_dict(torch.load(CRITIC_LOAD_PATH))
    print("Previous Model loaded successfully")

#print(policy_nn)
#print(critic_nn)
LEARNING_RATE = 0.00001
_MOMENTUM = 0.9
## initialize an optimizer
policy_optimizer = torch.optim.Adam(policy_nn.parameters()
                    , lr=LEARNING_RATE
                    #, momentum = _MOMENTUM
)
critic_optimizer = torch.optim.Adam(critic_nn.parameters()
                    , lr=LEARNING_RATE
                    #, momentum = _MOMENTUM
                    )

#print(state.shape[0])
running_reward = 0

with open('Run_info.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["episode no", "Episode Reward", "Accumulated Reward", "Time Stamp"])

previous_reward = -21.0
for e in count(1):

    action_log_probs = list()
    rewards = list()
    values = list()
    next_states = list()
    state = env.reset()
    action_counter = 0
    last_action = 0
    action_list = list()
    single_episode_rewards = list()
    action_losses = list()
    critic_losses = list()
    
    while True:
        env.render()
        
        ## First Step:
        ##  Get an action from our policy (agent) neural network
        ##  Take an action sampled from a categorical distribution given the state
        if(GPU):
            action_prob = policy_nn(torch.cuda.FloatTensor(state).unsqueeze(0))
        else:
            action_prob = policy_nn(torch.FloatTensor(state).unsqueeze(0))
        
        ## Sampling the action from the action probability list
        action = action_prob.sample()
        
        action_log_prob = action_prob.log_prob(action)
        action_log_probs.append(action_prob.log_prob(action))
        
        if(GPU):
            value = critic_nn(torch.cuda.FloatTensor(state).unsqueeze(0))
        else:
            value = critic_nn(torch.FloatTensor(state).unsqueeze(0))
        values.append(value[0])
        next_state, reward, is_done, _ = env.step(action.item())
         
        ## current state is next state now
        state = next_state

        rewards.append(reward)
        single_episode_rewards.append(reward)
        
        if is_done:
            print('Episode Ends with reward: {}'.format(sum(single_episode_rewards)))
            break

    if(RUN_MODE == TRAIN):
        ## Now we have the discounted reward + log_probs of the actions
        returns = discounted_returns(rewards)
        #print(returns)
        
        action_losses = list()
        critic_losses = list()
        ## collect the action losses to a list
        for ret, l_prob, v in zip(returns, action_log_probs, values):
            advantage = ret - v
            #print(advantage)
            #print(-l_prob * ret)
            action_losses.append(-l_prob * advantage.detach())
            critic_losses.append(advantage.pow(2))
    
        policy_optimizer.zero_grad()
        
        ## accumulate the action losses
        action_loss = torch.cat(action_losses).sum()
        action_loss.backward()
        
        ## step the optimizer
        policy_optimizer.step()
    
        critic_optimizer.zero_grad()
        critic_loss = torch.cat(critic_losses).mean()
        critic_loss.backward()
        critic_optimizer.step()
        
        ## get stats
        ep_reward = sum(rewards)
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        
        if e % 10 == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(e, ep_reward, running_reward))
    
        if e % 10 == 0:
            now = datetime.now()
            _time = now.strftime("%Y-%m-%d_%H:%M:%S")
            
            with open('Run_info.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([e, ep_reward, running_reward, str(_time)])
                
        if int(running_reward) > int(previous_reward):
            previous_reward = running_reward
            now = datetime.now()
            _time = now.strftime("%Y-%m-%d_%H:%M:%S")
            torch.save(policy_nn.state_dict(),'policy_' + str(int(running_reward)) + '.pt')
            torch.save(critic_nn.state_dict(),'critic_' + str(int(running_reward)) + '.pt')
            #torch.save(policy_nn.state_dict(),'policy_' + str(_time) + '_r_' + str(int(running_reward)) + '.pt')
            #torch.save(critic_nn.state_dict(),'critic_' + str(_time) + '_r_' + str(int(running_reward)) + '.pt')
            
            print('Model Saved Successfully at : '+ _time+', with reward: '+ str(int(running_reward)))
            
            
        if (env.spec.reward_threshold is not None):
            if (running_reward > env.spec.reward_threshold):
                print("Solved! Running reward is now {} and the last episode runs to {} time steps!".format(running_reward))
                break
