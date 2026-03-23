import numpy as np
from lipo import GlobalOptimizer

class LIPOOptimizer:
    """Parameter-free global optimizer via the lipo package."""
    def __init__(self, search_space, n_iter, experiment, maximize=True):
        self.search_space = search_space
        self.n_iter = n_iter
        self.experiment = experiment
        self.maximize = maximize