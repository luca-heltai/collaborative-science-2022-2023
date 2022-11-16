import FairnessTrial
import numpy as np


class Phase2:
    def __init__(self, mu, alpha, lamb, lcb, count, t, probability, T, hypercube, total_reward):
        self.mu = mu
        self.alpha = alpha
        self.lamb = lamb
        self.lcb = lcb
        self.count = count
        self.t = t
        self.probability = probability
        self.T = T
        self.hypercube = hypercube
        self.total_reward = total_reward

    def second_phase(self):
        lesser = np.array([self.hypercube[i][0] for i in range(len(self.probability))])
        upper = np.array([self.hypercube[i][0] for i in range(len(self.probability))])
        softmax1 = FairnessTrial.softmax(coef=0.2, estimation=lesser)
        softmax2 = FairnessTrial.softmax(coef=0.2, estimation=upper)
        fairness_function_lesser = softmax1.softmax_estimate()
        fairness_function_upper = softmax2.softmax_estimate()
        update_bound = np.zeros(len(self.mu))
        while True in [fairness_function_upper[i] - fairness_function_lesser[i] > self.alpha and self.lcb[i] < self.lamb for i in range(len(self.lcb))]:
            for i in range(len(self.mu)):
                if np.random.random() < self.probability[i]:
                    r = 1
                else:
                    r = 0
                self.count[i] += 1
                self.t += 1
                self.mu[i] = ((self.count[i] - 1) * self.mu[i] + r) / self.count[i]
                self.total_reward += r
            if self.t >= self.T:
                break
            j = np.argmax(self.mu)
            for i in range(len(self.mu)):
                update_bound[i] = np.sqrt(2 * np.log(self.T) / self.count[i])
                self.lcb[i] = max(0.0, self.mu[j] - self.mu[i] - 2 * update_bound[i])
                self.hypercube[i] = [max(0, self.mu[i] - update_bound[i]), min(1, self.mu[i] + update_bound[i])]
        return self.hypercube, self.lcb, self.mu, self.count, self.t, self.total_reward
