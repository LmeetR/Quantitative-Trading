import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Q_Net(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, device) -> None:
        """

        :param state_dim: state大小 = window窗口大小 x 股票所有特征
        :param hidden_dim: 隐藏层的大小维度
        :param action_dim: 最终输出的是action的数组

        """
        super(Q_Net, self).__init__()
        self.linear1 = nn.Linear(state_dim, hidden_dim).to(device)
        self.linear2 = nn.Linear(hidden_dim, action_dim).to(device)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class DQN_Agent(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, lr, device, gamma, epsilon, target_update) -> None:
        super(DQN_Agent, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # action的状态个数
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.gamma = gamma
        self.device = device

        # 需要2个网络：https://blog.csdn.net/xz15873139854/article/details/108032932
        # 预测网络：利用神经网络计算Q值
        self.Q_Net = Q_Net(state_dim, hidden_dim, action_dim, device)
        # 目标网络
        # 利用预测网络计算预测值，利用目标网络计算目标值，两者之间的误差，用来更新预测网络的参数。
        # 目标网络就是用来检验预测网络的Q值是否预测的正确，如果不正确，就更新预测网络参数。
        self.Target_Q_Net = Q_Net(state_dim, hidden_dim, action_dim, device)
        self.Q_optimizer = torch.optim.Adam(self.Q_Net.parameters(), lr=lr)
        self.Loss = []
        self.count = 0
        self.target_update = target_update  # 目标网络更新频率

    def take_action(self, x, random):
        """The policy of agent"""
        if random:
            # 如果满足随机概率，则随机选择动作，epsilon算法
            if np.random.random() < self.epsilon:
                # 随机生成-1， 0， 1 3个状态
                action = np.random.randint(self.action_dim) - 1
            else:
                # 超过随机概率，则选择Q值最大的动作
                action = self.Q_Net(x).argmax().item() - 1
            return action
        else:
            action = self.Q_Net(x).argmax().item() - 1
            return action

    def update(self, transition_dict):
        # transition_dict: [ 1xwindows 数组 ] ，假设数组长度是m
        # torch.stack(0) : m x 1 x windows
        # squeeze(dim=1) : m x windows,
        states = torch.stack(transition_dict['states'], 0).squeeze(dim=1).to(self.device)
        # print('states:', states.size(), states.dtype)
        # 把经验池中的action数组变成int类型，然后变成m x 1 维度
        actions = torch.from_numpy(np.array(transition_dict['actions']).astype(np.int64)).view(-1, 1).to(self.device)  # .unsqueeze(dim=2)
        # print('action:', actions.size(), actions.dtype)
        # 把经验池中的reward数组变成float类型，然后变成m x 1 维度
        reward = torch.from_numpy(np.array(transition_dict['rewards'])).view(-1, 1).to(self.device).type(torch.float32)
        # print('Reward: ', reward.size(), reward.dtype)
        # 把经验池中的next_state数组 用states的处理方式，变成m x windows 大小的批次数据
        next_states = torch.stack(transition_dict['next_states'], 0).squeeze(dim=1).to(self.device)
        # print('next_state:', next_states.size(), next_states.dtype)
        # 把经验池中的done数组变成float类型，然后变成m x 1 维度
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float64).view(-1, 1).to(self.device).type(torch.float32)
        # print('done:', dones.size(), dones.dtype)

        # 准备开始训练

        # DQN 智能体优化R
        # 1. 首先给定批次的states数据m个，得到预测的Q值
        # 预测Q数据应该是 3 x m ， 然后用gather得到每一个批次的预测的Q值
        Q_values = self.Q_Net(states).gather(1, actions)
        # print("Q_Values:", Q_values.size(), Q_values.dtype)

        # 2. 然后给定批次的next_states数据m个，得到目标网络的Q值
        q_targets = self.Target_Q_Net(next_states).max(1)[0].view(-1, 1)
        # print('q_targets:', q_targets.size())

        # 3. 计算TD误差目标
        Q_targets = reward + self.gamma * q_targets * (1 - dones)  # TD误差目标
        # print("Q_target:", Q_targets.size(), Q_targets.dtype)
        DQN_Loss = torch.mean(F.mse_loss(Q_values, Q_targets))

        # 计算误差，
        self.Q_optimizer.zero_grad()
        DQN_Loss.backward()

        # 记录误差，开始训练预测网络
        self.Loss.append(DQN_Loss)
        self.Q_optimizer.step()
        # 目标网络参数更新，多少次就复制一次预测网络参数到目标网络
        if self.count % self.target_update == 0:
            self.Target_Q_Net.load_state_dict(self.Q_Net.state_dict())
        self.count += 1
