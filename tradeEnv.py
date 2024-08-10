import numpy as np


class portfolio_tradeEnv:
    # action = [-1, 0, 1]
    # all balance 

    def __init__(self, day, stock, balance, cost) -> None:
        self.day = day

        # 数据的结构是 windows大小的df 数组
        self.stock = stock  # 数据

        # state就是第day日的数据， 也就是会看window大小的数据
        self.stock_state = self.stock[self.day]
        self.balance = balance

        # 账户的股票份额，按天记录
        self.shares = [0] * 1
        self.transaction_cost = cost
        self.terminal = False

        # 资产每日收益率序列
        self.rate = []
        self.reward = 0

    def step(self, action):
        # 如果到达最后一天，则终止
        self.terminal = self.day >= len(self.stock) - 1
        if self.terminal:
            # 返回状态，奖励，是否终止，其他信息
            # print('Balance:', self.balance, 'Close Price:', self.stock_state.Close.values[-1], 'Shares:', self.shares[-1])
            return self.stock_state, self.reward, self.terminal, {}

        else:
            # 如果没结束的话
            # 账户净值 = 账户余额 + 当前股价 * 股票份额
            begin_assert_value = self.balance + self.stock_state.Close.values[-1] * self.shares[-1]
            if action == -1:
                # 执行卖出动作
                self.sell(action)
            if action == 0:
                # 执行持有动作，即不执行任何交易行为
                self.hold(action)
            if action == 1:
                # 执行买入动作
                self.buy(action)

            # 执行完动作， day+1 表示进入下一步，为了获取下一个step的状态
            self.day += 1
            # print('Day:', self.day)
            self.stock_state = self.stock[self.day]
            end_assert_value = self.balance + self.stock_state.Close.values[-1] * self.shares[-1]
            self.rate.append((end_assert_value - 100000) / 100000 + 1)

            # 计算奖励：下一步资产/上一步资产 - 1， 资产增值率大小作为奖励
            self.reward = (end_assert_value - begin_assert_value) / begin_assert_value
            # print('Day:', self.day, 'Balance:', self.balance, 'Close Price:', self.stock_state.Close.values[-1],
            #       'Shares:', self.shares[-1], 'Value:', end_assert_value, 'Action:', action)

            return self.stock_state, self.reward, self.terminal, {}

    def buy(self, action):
        # 账户的钱是否支持买入行为

        if self.balance > 0:
            self.shares.append((1 - self.transaction_cost) * self.balance / self.stock_state.Close.values[-1])
            self.balance = 0
            # print('Buy Share:', action * self.HMAX_SHARE)
        else:
            pass

    def hold(self, action):
        pass

    def sell(self, action):
        # 全部卖出
        cash = self.stock_state.Close.values[-1] * self.shares[-1] * (1 - self.transaction_cost)
        self.balance += cash
        # 更改份额
        self.shares.append(0)

    def reset(self, ):
        # 重置环境
        self.day = 0
        # 重置账户的钱
        self.balance = 100000
        self.stock_state = self.stock[self.day]
        self.terminal = False

        # 初始状态， 就是股票第一天的回看window的股票指标数据
        return self.stock_state
