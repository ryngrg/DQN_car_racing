import gym
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib as plt

env_name = "CarRacing-v0"
env = gym.make(env_name)
print(env.observation_space)
print(env.action_space)

class replayBuffer():
    def __init__(self, input_dim, action_size):
        self.mem_size = 1000
        self.mem_counter = 0
        self.state_memory = np.zeros((self.mem_size, input_dim[0], input_dim[1], input_dim[2]))
        self.new_state_memory = np.zeros((self.mem_size, input_dim[0], input_dim[1], input_dim[2]))

        self.action_memory = np.zeros((self.mem_size, action_size), dtype = np.int8)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype = np.float32)

    def store_transition(self, state, action, reward, new_state, done):
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state

        actions = np.zeros(self.action_memory.shape[1])
        actions[action] = 1.0
        self.action_memory[index] = actions

        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done)
        self.mem_counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

class QNetwork():
    def __init__(self, state_dim, action_size):
        # print(state_dim)
        # print(action_size)

        self.model = keras.models.Sequential()

        self.model.add(layers.Conv2D(32, (3, 3), strides = (1, 1), padding = "valid", activation = "relu", input_shape = state_dim))
        self.model.add(layers.MaxPool2D(2, 2))
        self.model.add(layers.Conv2D(32, (3, 3), activation = "relu"))
        self.model.add(layers.MaxPool2D(2, 2))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation = "relu"))
        self.model.add(layers.Dense(action_size))

        print(self.model.summary())
        
        self.optim = keras.optimizers.Adam(lr = 0.001)
        self.model.compile(optimizer = self.optim, loss = 'mse')

    def update_model(self, state, q_target):
        _ = self.model.fit(state, q_target, verbose = 0)

    def get_q_state(self, state):
        q_state = self.model.predict(state)
        return q_state

class DQNAgent():
    def __init__(self, env, batch_size):
        self.state_dim = env.observation_space.shape
        self.action_size = 8
        self.q_network = QNetwork(self.state_dim, self.action_size)
        self.gamma = 0.99
        self.eps = 1.0
        self.batch_size = batch_size

        self.memory = replayBuffer(self.state_dim, self.action_size)

    def get_action(self, state):
        state = state[np.newaxis, :]
        if random.random() < self.eps:
            action = np.random.randint(self.action_size)
        else:
            actions = self.q_network.get_q_state(state)
            action = np.argmax(actions)

        if action == 1:
            return [0, 0, 0.5] # brake
        elif action == 2:
            return [0, 1, 0] # accelerate
        elif action == 3:
            return [1, 0, 0] # steer right
        elif action == 4:
            return [-1, 0, 0] # steer left
        elif action == 5:
            return [1, 0.7, 0] # steer right while accelerating
        elif action == 6:
            return [-1, 0.7, 0] # steer left while accelerating
        elif action == 7:
            return [1, 0, 0.3] # steer right while braking
        elif action == 0:
            return [-1, 0, 0.3] # steer left while braking

    def train(self):
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

       # print(np.shape(state))
       # print(np.shape(new_state))
        
        action_values = np.array([i for i in range(self.action_size)])
        action_indices = np.dot(action, action_values)

       # print(action_indices)

        q_eval = self.q_network.get_q_state(state)
        q_next = self.q_network.get_q_state(new_state)
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype = np.int32)

       # print(action_indices)

        q_target[batch_index, action_indices] = reward * 5 + self.gamma*np.max(q_next, axis = 1)

        self.q_network.update_model(state, q_target)

        self.eps = max(0.99 * self.eps, 0.2)

    def remember(self, state, action, reward, next_state, done):
        if action == [0, 0, 0.5]:
            action = 1 # brake
        elif action == [0, 1, 0]:
            action = 2 # accelerate
        elif action == [1, 0, 0]:
            action = 3 # steer right
        elif action == [-1, 0, 0]:
            action = 4 # steer left
        elif action == [1, 0.7, 0]:
            action = 5 # steer right while accelerating
        elif action == [-1, 0.7, 0]:
            action = 6 # steer left while accelerating
        elif action == [1, 0, 0.3]:
            action = 7 # steer right while braking
        elif action == [-1, 0, 0.3]:
            action = 0 # steer left while braking
        self.memory.store_transition(state, action, reward, next_state, done)

    def save_model(self):
        self.q_network.model.save('carRacingModel.h5')

    def load_model(self):
        self.q_network.model = keras.models.load_model('carRacingModel.h5')

agent = DQNAgent(env, 32)
# agent.load_model()
num_episodes = 500

for ep in range(num_episodes):
    state = env.reset()
    total_reward = 0
    num = 0
    done = False
    while (not done) and (num < 2000):
        if num < 40:
            action = [0, 1, 0]
        else:
            action = agent.get_action(state)
        for i in range(3):
            num += 1
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            # print(action)
            agent.train()
            env.render()
            total_reward += reward
            state = next_state
    print("Episode: {}, total_reward: {:.2f}".format(ep, total_reward))
    agent.save_model()

env.close()
