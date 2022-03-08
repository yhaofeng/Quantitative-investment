from torch import nn
import numpy as np
from pandas import DataFrame

class BaseAlphaNet(nn.Module):
    def __init__(self):
        super(BaseAlphaNet).__init__()

    def ts_corr(self, Y, d):
        unfold = nn.Unfold(kernel_size=2)

    def ts_stddev(self,d):
        pass