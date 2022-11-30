import numpy as np

class softmax():

    def __init__(self, coef, estimation):
        self.coef = coef
        self.estimation = estimation
    def softmax_estimate(self):
        return np.exp(self.estimation*self.coef)/np.sum(np.exp(self.estimation*self.coef))
    def linear_estimation(self):
        return self.estimation*self.coef/np.sum(self.estimation*self.coef)

    def uniform_estimation(self):
        return np.ones(len(self.estimation))/len(self.estimation)