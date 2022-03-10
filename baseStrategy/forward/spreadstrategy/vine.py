import numpy as np
from copulas.multivariate import VineCopula
from copulas.bivariate.base import Bivariate

class vine():
    def __init__(self,data,vine_type='center'):
        """
        对数据构建vine模型，用于输出目标数据在给定partner数据下的条件概率
        :param data: 建模所对应的数据，包含目标数据，为最后一列，其余前几列为partner数据
        :param vine_type:vine类型，本文采用的是center
        """
        self.data = data
        self.vine_type = vine_type

    def fit(self):
        self.model = VineCopula(self.vine_type)
        self.model.fit(self.data)

    def conditional_probability(self):
        self.fit()
        num_tree = len(self.model.trees)
        tree = self.model.trees[num_tree-1]
        edge = tree.edges[0]
        copulas = Bivariate(copula_type=edge.name)
        copulas.theta = edge.theta
        U = edge.to_dict()['U']
        probs = []
        for i in range(len(self.data)):
            right_u = U[0][i]
            left_u = U[1][i]
            x_right_left = np.array([[left_u,right_u]])
            prob = copulas.partial_derivative(x_right_left)[0]
            probs.append(prob)
        return probs
