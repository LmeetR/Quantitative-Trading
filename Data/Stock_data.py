import pandas as pd


class data:

    def __init__(self, link, window_length, t):
        self.b = []
        self.t = t 
        self.data = pd.read_csv(link)
        self.window_length = window_length

    def process(self):
        """
        处理金融数据以生成用于分析的窗口数据集。

        Returns:
            list: 包含各个子区间数据集的列表，每个数据集包含从窗口开始到某个时间点的数据。
        """
        # 选择特定的列并设置索引为日期，以便于后续的数据处理和分析。
        df = self.data.loc[:, ['Close', 'Open', 'High', 'Low', 'RSI', 'ROC', 'CCI', 'MACD', 'EXPMA', 'VMACD']].set_index(self.data['Date'])

        # 生成一个列表，包含从窗口长度开始到数据集结束的每个子区间数据集。
        # 每个数据集是一个window_length长度的df
        self.b.insert(-1, [df[i-self.window_length:i] for i in range(self.window_length, len(self.data))])

        return self.b

    # 训练集数据， 前self.t个数据
    def train_data(self, ):
        self.b = self.process()
        return self.b[0][:self.t]

    # 测试集数据， self.t之后的数据
    def trade_data(self, ):
        self.b = self.process()
        return self.b[0][self.t:] 


if __name__ == "__main__":
    D = data(link=r'..\Data\000001_SZ.csv', window_length=15, t=2000)
    b = D.trade_data()
    print(type(b), len(b), type(b[100]))
    print(b[100])

    state = b[100]
    state = (state - state.mean()) / (state.std())
    print(state, '\n', state.mean())
