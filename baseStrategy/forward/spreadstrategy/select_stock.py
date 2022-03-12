import itertools
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from scipy.spatial.distance import pdist
from sklearn.gaussian_process import GaussianProcessRegressor as GPR

# 定义SoftMax函数，求概率
def softmax(x):
    probs = np.exp(x)
    probs /= np.sum(probs)
    return probs

class Select_Stock():

    def __init__(self, data, partner_nums):
        """

        :param data: 包含各个行业股票的pannel数据，为DataFrame类型
        :param partner_nums: partner股票数目
        """
        self.data = data
        self.partner_nums = partner_nums

    # 计算spearman系数
    def spearman(self,i,combination):
        column_index = [i]
        for c in combination:
            column_index.append(c)
        data_copy = self.data.iloc[:,column_index]
        corr = data_copy.corr('spearman')
        return corr.sum().sum()

    # 依据线性模型计算对数似然
    def linear_loglikelihood(self,i,combination):
        """

        :param i: int类型，为目标数据的位置索引
        :param combination: list类型，元素是各个partner的对应位置索引
        :return:对数似然
        """
        combination = list(combination)
        X = self.data.iloc[:,combination].values
        n = len(self.data)
        y = self.data.iloc[:,i].values
        XtX = X.T.dot(X)
        Xty = X.T.dot(y)
        theta = np.linalg.solve(XtX, Xty)
        y_err = X.dot(theta) - y
        err = y_err.mean()
        return -err

    # 在线性模型基础上引入多项式特征
    def poly_loglikelihood(self,i,combination):
        """

        :param i: int类型，为目标数据的位置索引
        :param combination: list类型，元素是各个partner的对应位置索引
        :return:对数似然
        """
        combination = list(combination)
        X = self.data.iloc[:, combination].values
        n = len(self.data)
        self.poly = PolynomialFeatures()
        X = self.poly.fit_transform(X)[:,1:]
        y = self.data.iloc[:, i].values
        XtX = X.T.dot(X)
        Xty = X.T.dot(y)
        theta = np.linalg.solve(XtX, Xty)
        y_err = X.dot(theta) - y
        err = y_err.mean()
        return err

    def gpr(self,i, combination):
        combination = list(combination)
        X = self.data.iloc[:, combination].values
        y = self.data.iloc[:, i].values
        med_w = np.median(pdist(X, 'euclidean'))
        model = GPR().fit(X / med_w, y)
        err =  y.reshape(-1, 1) - model.predict(X / med_w).reshape(-1, 1)
        lnL = -np.log(np.sum(np.square(err)))
        return lnL

    # 标准化
    def standard(self):
        self.scale = StandardScaler()
        columns = self.data.columns
        scale = StandardScaler()
        values = scale.fit_transform(self.data)
        self.data = pd.DataFrame(values, columns=columns)

    def select_best(self,target_index):
        """
        :param target_index: 为int类型，是self.data中对应的目标数据的位置编号
        :return:得分最高的partners所对应的编号，以及对应的最大得分
        """
        methods = [self.spearman,self.linear_loglikelihood,self.poly_loglikelihood,self.gpr]
        index = ['spearman','linear_loglikelihood','poly_loglikelihood','GPR']
        parter_index = np.delete(np.array(range(len(self.data.columns))), target_index)
        combinations = [i for i in itertools.combinations(parter_index,self.partner_nums)]
        temp = pd.DataFrame(index=index,columns=combinations)
        for method,i in zip(methods,index):
            for combination in combinations:
                temp.loc[i][combination] = method(target_index,combination)
        for i in index:
            temp.loc[i] = softmax(np.array(temp.loc[i].values,dtype=np.float))
        total = temp.sum(axis=0)
        best_partner_ids = combinations[total.values.argmax()]
        max_score = total.max()
        return list(best_partner_ids),max_score


    def select_stock(self):
        stock_names = self.data.columns
        stocks = pd.DataFrame(columns=['stock name','stock id','partner stock id','partner stock name','max score'])
        stocks['stock name'] = stock_names
        stocks['stock id'] = np.array(range(len(self.data.columns)))
        ids = []
        for i,name in enumerate(stock_names):
            # print(name)
            id,max_score = self.select_best(i)
            ids.append(id)
            stocks.loc[i,'max score'] = max_score
            stocks.loc[i,'partner stock name'] = str([stock_names[i] for i in id])
        stocks['partner stock id'] = ids
        return stocks



