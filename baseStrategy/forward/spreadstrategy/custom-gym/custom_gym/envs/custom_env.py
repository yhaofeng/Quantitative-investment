import gym
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
import matplotlib.pyplot as plt


class CustomEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,data,gamma,dates,action_shape,inital_account,fee_A,fee_B):
        """

        :param data: 包含当前交易环境的在不同时间点的状态，为m*n的数组，其中最后三维数据分别是证券A、证券B的当前的价格和证券A的错误定价指数
        """
        print('CustomEnv Environment initialized')
        self.m,self.n = data.shape
        self.data = data
        self.numA = 0. #证券A的仓位
        self.numB = 0. #证券B的仓位
        self.current_loc = 0
        self.gamma = gamma
        self.reward = 0
        self.cost = 0
        self.terminal = False
        self.dates = dates
        self.hold = False
        self.account = inital_account #初始账户金额
        pl_ratio = 0 #盈亏比
        self.fee_A = fee_A # 有价证券A交易涉及的费用
        self.fee_B = fee_B # 有价证券B交易涉及的费用

        # action_space normalization

        self.action_space = spaces.Box(low=-np.inf, high=np.inf, dtype=np.float32, shape=(action_shape,))
        # observation_space normalization
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, dtype=np.float32,shape=(self.n+2,))

        # memorize  return each step
        self.return_memory = []
        self.bgn_asset_memory = [inital_account]
        self.end_asset_memory = []
        self.A_position_memory =[]
        self.B_position_memory = []
        self.actions_memory = []
        self.used_margin = []
        self.date_memory = [dates]
        self.trades = 0
        self._seed()

    def step(self,a):
        self.terminal = self.current_loc >= self.m - 1

        if self.terminal:
            #当前状态为结束状态,平仓处理
            df = pd.DataFrame(self.return_memory)
            df.columns = ['daily_return']
            plt.plot(df.daily_return.cumsum(), 'r')
            plt.savefig('cumulative_reward.png')
            plt.close()
            print("total_cost: ", self.cost)
            print("total_trades: ", self.trades)

            return self.state, self.reward, self.terminal, {}

        else:
            last_numA = self.numA  # 上一时刻A的仓位
            last_numB = self.numB  # 上一时刻B的仓位
            price_A = self.data[self.current_loc,-3]
            price_B = self.data[self.current_loc,-2]
            misprice_index = self.data[self.current_loc, -1]  # 当前状态下错误定价指数
            min_a = np.sort(a)[0]  # 平仓阈值
            median_a = np.sort(a)[1]  # 开仓阈值
            max_a = np.sort(a)[2]  # 止损阈值
            self.current_loc += 1


            self.actions_memory.append(a)
            if self.hold == False:
                if median_a<abs(misprice_index)<max_a:# 开仓
                    self.hold = True
                    if misprice_index > 0:#表明A被高估
                        self.numA = -1
                        self.numB = +1

                    else:#表明A被低估
                        self.numA = 1
                        self.numB = -1
                else:#持有
                    self.numA = 0
                    self.numB = 0
            else:
                if abs(misprice_index)<=min_a:#平仓
                    self.numA = 0
                    self.numB = 0
                    self.hold = False
                elif abs(misprice_index)>=max_a:#止损
                    self.numA = 0
                    self.numB = 0
                    self.hold = False
                else:
                    self.numA = last_numA
                    self.numB = last_numB
        self.state = np.concatenate((self.data[self.current_loc], [self.numA, self.numB]))
        self.A_position_memory.append(self.numA)
        self.B_position_memory.append(self.numB)

        self.reward = self.data[self.current_loc,-3]*(self.numA-last_numA)+self.data[self.current_loc,-2]*(self.numB-last_numB)
        self.return_memory.append(self.reward)
        return self.state, 0, self.terminal, {}


    def reset(self):
        self.current_loc = 0
        self.numA = 0
        self.numB = 0
        # load states
        self.state = np.concatenate((self.data[self.current_loc], [self.numA, self.numB]))

        self.cost = 0
        self.trades = 0
        self.terminal = False
        self.return_memory = [self.account]
        self.actions_memory = []
        self.date_memory = [self.dates]
        return self.state

    def save_action_memory(self):
        # date and close price length must match actions length
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ['date']

        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = ['action']
        df_actions.index = df_date.date
        # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]




