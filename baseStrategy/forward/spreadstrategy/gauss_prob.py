import numpy as np
from scipy.stats import norm


class multi_gauss():
    def __init__(self,data,target_index):
        self.data = data
        self.target_index = target_index
        self.n = len(self.data)

    def MLE(self):

        mu_hat = self.data.mean(axis=0).values
        A = self.data.cov()*(self.n-1).values
        return mu_hat,A

    def conditional_probability(self):
        probs = []
        mu_hats,A = self.MLE()
        # 分别求得target和partner的期望
        mu_target = mu_hats[self.target_index] # target的期望
        mu_partner = mu_hats[:self.target_index] # partner的期望
        # 获得相应的协方差，用于估计条件分布
        A21 = A[3, :self.target_index]
        A12 = A[:self.target_index, 3]
        A11 = A[:self.target_index, :self.target_index]
        A11_inv = np.linalg.inv(A11)
        A22 = A[self.target_index,self.target_index]
        sigma_hat = (A22-A21.dot(A11_inv).dot(A12))/self.n #target在partner给定下的条件方差
        B_hat = A21.dot(A11_inv) #target对partner的回归系数阵
        target_data = self.data.iloc[:,self.target_index]
        label = self.data.columns[self.target_index]
        partner_data = self.data.drop(label,axis=1)
        for td,pd in zip(target_data.values,partner_data.values):
            mu_temp = mu_target + B_hat.dot(pd-mu_partner)
            std = np.sqrt(sigma_hat)
            prob = norm.cdf(td,local=mu_temp,scale=std)
            probs.append(prob)
        return probs
