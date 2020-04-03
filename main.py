import gym, random, math
import numpy as np
from collections import deque

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
import tensorflow as tf

import matplotlib.pyplot as plt

class Memory:
    ''' Experience memory, used to store experience and sample random experiences for training '''
    def __init__(self, max_len):
        self.buffer = deque(maxlen=max_len)
        
    def add(self, experience):
        self.buffer.append(experience)
        
    def sample(self):
        index = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[i] for i in index]
    
    def populate_memory(self, episodes):
        ''' used to initialize the experience memory buffer '''
        for i in range(episodes):
            # first episode
            if i == 0:
                state = env.reset()            
            # make a random action
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            
            # if episode is done (player is killed)
            if done:
                state = np.zeros(state.shape)
                self.add((state, action, reward, next_state, done))
                state = env.reset()
            else:
                self.add((state, action, reward, next_state, done))
            # update the state
            state = next_state

# simple MLP model with linear activation at the output
def dqn(input_shape):
    In = Input(shape=input_shape)
    x = Dense(128, activation='relu')(In)
    x = Dense(64, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    Out = Dense(env.action_space.n)(x)
    
    model = Model(In, Out)
    model.compile(loss='mse', optimizer=RMSprop(learning_rate=alpha))
    
    return model

# decaying epsilon
def get_epsilon(e):
    ''' decrease the exploration rate at each episode '''
    return max(min_epsilon, min(1, 1.0 - math.log10((e + 1) / 30)))

# chooses between exploration and exploitation based on epsilon value
def get_action(state, epsilon):
    if random.uniform(0,1) < epsilon:
        action = env.action_space.sample()
    else:
        Qs = model.predict(np.array([state],))
        action = np.argmax(Qs)
    return action

def train_model():
    ''' trains the model using Q-learning algorithm '''
    batch = memory.sample()
    
    states = np.array([x[0] for x in batch])
    actions = np.array([x[1] for x in batch])
    rewards = np.array([x[2] for x in batch]) 
    next_states = np.array([x[3] for x in batch])
    dones = np.array([x[4] for x in batch])
    
    target = model.predict(states)
    Qs = target_model.predict(next_states)
    
    # create x_train, y_train for the DQN
    for i in range(batch_size):
        terminal = dones[i]
        # if terminal state, target equals to current reward
        if terminal:
            target[i][actions[i]] = rewards[i]
        else:
            target[i][actions[i]] = rewards[i] + gamma*np.max(Qs[i])
    loss = model.train_on_batch(states, target)
    
    return loss

# environment for agent
env = gym.make('MountainCar-v0')
state = env.reset()

# hyperparameters
alpha = 0.001 # learning rate
gamma = 0.99 # discount factor
min_epsilon = 0.1 # exploration factor
batch_size = 64
tau = 0.9
episodes = 500
max_iter = 5000

# experience memory
memory = Memory(max_len=10_000)
# dqn model
model = dqn(input_shape=state.shape)
target_model = dqn(input_shape=state.shape)

# initialize the experience memory
memory.populate_memory(episodes=batch_size)

total_rewards = list()

rewards = 0
loss = np.nan
episode = 1

while episode < episodes:
    state = env.reset()
    done = False
    step = 0
    
    episode_rewards = 0
    epsilon = get_epsilon(episode)
    
    while step < max_iter:
        step += 1
        env.render()
        action = get_action(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        
        episode_rewards += reward
        
        if done or step == max_iter:
            # update target model weights
            W = np.array(model.get_weights())
            W_ = np.array(target_model.get_weights())
            target_model.set_weights(tau*W + (1-tau)*W_)
            
            success = True if step < 200 else False
            print(f'Episode: {episode}/{episodes} total reward: {rewards}, loss: {loss}, success: {success}')
            break
        # Add experience to memory
        memory.add((state, action, reward, next_state, done))

        # s(t+1) is now the current state
        state = next_state
        
        # train the model
        loss = train_model()
            
    total_rewards.append(rewards)
    episode += 1

env.close()

# plot the distribution of rewards during training
plt.plot(total_rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Reward per episode")
plt.show()
