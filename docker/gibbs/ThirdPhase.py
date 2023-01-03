import numpy as np
import FairnessTrial

class Phase3:

    def __init__(self, lcb, lamb, count, t, T, mu, probability, total_reward):
        self.lcb = lcb
        self.lamb = lamb
        self.count = count
        self.t = t
        self.T = T
        self.mu = mu
        self.probability = probability
        self.total_reward = total_reward

    def Phase3_estimation(self):
        softmax1 = FairnessTrial.softmax(coef=0.2, estimation=self.mu)
        time = 0
        for i in range(len(self.lcb)):
            if self.lcb[i] < self.lamb and self.count[i] < self.T*softmax1.softmax_estimate()[i]:
                M = min(self.T - self.t, int(self.T*softmax1.softmax_estimate()[i] - self.count[i]))
                a = 0
                while a < M:
                    a += 1
                    if np.random.random() < self.probability[i]:
                        r = 1
                    else:
                        r = 0
                    self.total_reward += r
                self.count[i] += M
                self.t += M
        return self.count, self.t, self.lcb, self.total_reward

