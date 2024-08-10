import torch
import matplotlib.pyplot as plt
from Data.Stock_data import data
from Model.Deep_Q_Network import Q_Net, DQN_Agent
from tradeEnv import portfolio_tradeEnv


def Normalize(state):
    # 状态标准化
    state = (state - state.mean()) / (state.std())
    return state


def DQN_train(episode, stock_Name, minimum):
    """

    :param episode: 训练次数
    :param stock_Name: 训练股票名
    :param minimum: 经验回放池最低size
    :return:
    """
    # 智能体
    # state大小 = 每个股票有10个特征 x 窗口大小（15）
    agent = DQN_Agent(state_dim=150, hidden_dim=30, action_dim=3, lr=0.001, device="cuda:0", gamma=0.95,
                      epsilon=0.01, target_update=10)
    # 训练数据
    link = f'../Data/{stock_Name}.csv'
    train_df = data(link, window_length=15, t=2000).train_data()
    # 训练环境
    Env = portfolio_tradeEnv(day=0, balance=1, stock=train_df, cost=0.003)
    return_List = []
    # 智能体经验池
    transition_dict = {
        'states': [],
        'actions': [],
        'next_states': [],
        'rewards': [],
        'dones': [],
    }

    # 遍历训练次数
    for i in range(episode):
        print('Episode:', i)
        done = False
        episode_return = 0
        # 返回初始状态
        state = Env.reset()
        # 标准化state， 通过reshape压平成一维数据，输入给神经网络
        state = torch.tensor(Normalize(state).values, dtype=torch.float32).reshape(1, -1).to(device="cuda:0")
        while not done:
            action = agent.take_action(state, random=True)

            # reward 就是这一步股票的收益
            next_state, reward, done, _ = Env.step(action)
            # 经验回放池
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action + 1)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)

            # 标准化next state， 通过reshape压平成一维数据
            next_state = torch.tensor(Normalize(next_state).values, dtype=torch.float32).reshape(1, -1).to(device="cuda:0")
            transition_dict['next_states'].append(next_state)
            state = next_state

            # 累计奖励
            episode_return += reward

            # 经验回放池满了，开始学习
            if len(transition_dict['states']) >= minimum:
                # 智能体学习
                # print('---Learning---')
                agent.update(transition_dict)

        return_List.append(episode_return)
    # 模型保存
    PATH = r'..\Result\agent_dqn_' + stock_Name + '.pt'
    torch.save(agent.state_dict(), PATH, _use_new_zipfile_serialization=False)
    # 可视化reward
    plt.plot(range(len(return_List)), return_List)
    plt.show()


if __name__ == '__main__':
    import time

    # 完善：定时清空经验池 抽取固定数量的经验四元组
    start = time.time()
    DQN_train(episode=100, stock_Name='000938_SZ', minimum=1500)
    end = time.time()
    print('训练时间', end - start)
