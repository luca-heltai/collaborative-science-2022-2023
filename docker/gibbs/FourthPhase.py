import numpy as np
import FairnessTrial

class Phase4():
    def __init__(self, mu, t, T, probability, count, total_reward):
        self.mu = mu
        self.t = t
        self.T = T
        self.probability = probability
        self.count = count
        self.total_reward = total_reward

    def phase4_estimation(self):
        j = np.argmax(self.mu)
        a = 0
        while self.t < self.T:
            if np.random.random() < self.probability[j]:
                r = 1
            else:
                r = 0
            self.total_reward += r
            self.t += 1
            a += 1
            self.count += 1
        return self.count, self.t, self.mu, self.total_reward

