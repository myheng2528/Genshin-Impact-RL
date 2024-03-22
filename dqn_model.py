"""
Deep Q Network off-policy
"""
from collections import deque

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from parsers import args
from parsers import args
np.random.seed(42)
torch.manual_seed(2)


class Network(nn.Module):
    """
    Network Structure
    """

    def __init__(self,
                 n_features,
                 n_actions,
                 ):
        super(Network, self).__init__()

        self.conv_0 = nn.Conv3d(3, 32, kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))

        # P3D-B inspired layers
        self.conv_1 = nn.Conv3d(32, 32, kernel_size=(1, 1, 1), padding=(0, 0, 0))
        self.conv_2_0 = nn.Conv3d(32, 32, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.conv_2_1 = nn.Conv3d(32, 32, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.conv_3 = nn.Conv3d(32, 32, kernel_size=(1, 1, 1), padding=(0, 0, 0))

        self.batch_norm = nn.BatchNorm1d(n_features)

        self.dense = nn.Linear(32 * 3 * args.capture_height * args.capture_width, n_features)

        self.net = nn.Sequential(
            nn.Linear(in_features=n_features, out_features=16, bias=True),
            nn.Linear(in_features=16, out_features=n_actions, bias=True),
            nn.ReLU()
        )

    def forward(self, x):
        """

        :param s: s
        :return: q
        """
        conv0_out = F.relu(self.conv_0(x))

        residual = conv0_out
        conv1_out = F.relu(self.conv_1(conv0_out))
        conv2_0_out = F.relu(self.conv_2_0(conv1_out))
        conv2_1_out = F.relu(self.conv_2_1(conv1_out))
        # conv2_out = conv2_0_out + conv2_1_out
        conv2_out = torch.add(conv2_0_out,conv2_1_out)
        conv3_out = F.relu(self.conv_3(conv2_out))
        # conv3_out += residual
        conv3_out = torch.add(conv3_out,residual)
        flat = torch.flatten(F.relu(conv3_out),start_dim=1)
        flat1 = F.relu(self.dense(flat))
        batch_norm_out = self.batch_norm(flat1)
        output = self.net(batch_norm_out)

        return output


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, observation, action, reward, next_observation):
        self.buffer.append((observation, action, reward, next_observation))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        observations, actions, rewards, next_observations = zip(*batch)
        return np.array(observations), np.array(actions), np.array(rewards), np.array(next_observations)


class DeepQNetwork(nn.Module):
    """
    Q Learning Algorithm
    """

    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy=0.9,
                 replace_target_iter=300,
                 memory_size=5000,
                 batch_size=32,
                 e_greedy_increment=None):
        super(DeepQNetwork, self).__init__()

        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        # 这里用pd.DataFrame创建的表格作为memory
        # 表格的行数是memory的大小，也就是transition的个数
        # 表格的列数是transition的长度，一个transition包含[s, a, r, s_]，其中a和r分别是一个数字，s和s_的长度分别是n_features
        self.memory = ReplayBuffer(self.memory_size)

        # build two network: eval_net and target_net
        self.eval_net = Network(n_features=self.n_features, n_actions=self.n_actions)
        self.target_net = Network(n_features=self.n_features, n_actions=self.n_actions)
        # 将两个网络放到cuda上
        self.eval_net.cuda()
        self.target_net.cuda()
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)

        # 记录每一步的误差
        self.cost_his = []

    def store_transition(self, s, a, r, s_):
        self.memory.add(s, a, r, s_)

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            s = torch.FloatTensor(observation).cuda()
            self.eval_net.eval()
            actions_value = self.eval_net(s)
            self.eval_net.train()
            action = [np.argmax(actions_value.detach().cpu().numpy())][0]
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def _replace_target_params(self):
        # 复制网络参数
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replace_target_params()
            print('\ntarget params replaced\n')

        # sample batch memory from all memory
        # batch_memory = self.memory.sample(self.batch_size)
        s, a, r, s_ = self.memory.sample(self.batch_size)


        # run the nextwork
        s = torch.FloatTensor(s).cuda()
        s_ = torch.FloatTensor(s_).cuda()

        q_eval = self.eval_net(s)
        q_next = self.target_net(s_)

        # change q_target w.r.t q_eval's action
        q_target = q_eval.clone()

        # 更新值
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = a.astype(int)
        reward = r

        q_target[batch_index, eval_act_index] = torch.FloatTensor(reward).cuda() + self.gamma * q_next.max(dim=1).values

        # train eval network
        loss = self.loss_function(q_target, q_eval)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.cost_his.append(loss.detach().cpu().numpy())

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        plt.figure()
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.show()
