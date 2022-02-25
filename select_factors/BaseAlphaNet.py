from torch import nn
import numpy as np
from pandas import DataFrame

class BaseAlphaNet(nn.Module):
    def __init__(self):
        super(BaseAlphaNet).__init__()

    def ts_corr(X, Y, d):
        pass