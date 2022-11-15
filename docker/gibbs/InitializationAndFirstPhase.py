import numpy as np
np.random.seed(0)


class Environment:
    def __init__(self, K, T, probability):
        self.K = K
        self.probability = probability
        self.T = T

    def Explorations(self, lamb, bet):
        count = np.zeros(self.K)
        t = 0
        mu = np.zeros(self.K)
        update_bound = np.zeros(self.K)
        hypercube = np.array([[0.0, 1.0] for _ in range(self.K)])
        lcb = np.zeros(self.K)
        ucb = np.ones(self.K)
        total_reward = 0
        while True in [lcb[i] < lamb - bet and ucb[i] > lamb + bet for i in range(self.K)]:
            for i in range(self.K):
                if np.random.random() < self.probability[i]:
                    r = 1
                else:
                    r = 0
                total_reward += r
                count[i] += 1
                t += 1
                mu[i] = ((count[i]-1)*mu[i] + r)/count[i]
                if t >= self.T:
                    break
            j = np.argmax(mu)
            for i in range(self.K):
                update_bound[i] = np.sqrt(2*np.log(self.T)/count[i])
                lcb[i] = max(0, mu[j] - mu[i] - 2*update_bound[i])
                ucb[i] = min(1, mu[j] - mu[i] + 2*update_bound[i])
                hypercube[i] = [max(0, mu[i] - update_bound[i]), min(1, mu[i] + update_bound[i])]
            if t >= self.T:
                break
        return lcb, ucb, mu, hypercube, t, update_bound, count, total_reward

