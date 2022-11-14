import numpy as np
import matplotlib.pyplot as plt
#np.random.seed(0)

class Environment:

    def __init__(self, probs):
        self.probs = probs # Success probabilities for each arm

    def step(self, action):
        # Pull arm and get stochastic reward (1 for success, 0 for failure)
        return 1 if (np.random.random() < self.probs[action]) else 0

class Agent:

    def __init__(self, nActions, eps):
        self.nActions = nActions
        self.eps = eps
        self.n = np.zeros(nActions, dtype=np.int64) # action counts n(a)
        self.Q = np.zeros(nActions, dtype=np.float64) #Value of Q(a)

    def update_Q(self, action, reward):
        #Update Q action-value given (action, reward)
        self.n[action] += 1
        self.Q[action] += (1.0/self.n[action])*(reward - self.Q[action])

    def get_action(self):
        # Epsilon-greedy policy
        if np.random.random() < self.eps: # explore
            return np.random.randint(self.nActions)
        else:
            return np.random.choice(np.flatnonzero(self.Q == self.Q.max()))

# Start multi-armed bandit simulation
def experiment(probs, N_episodes, eps):
    env = Environment(probs) # initialize arm probabilities
    agent = Agent(len(env.probs), eps) # initialize agent
    actions, rewards = [], []
    for episode in range(N_episodes):
        action = agent.get_action() # sample policy
        reward = env.step(action) # take step + get reward
        agent.update_Q(action, reward) # Update Q
        actions.append(action)
        rewards.append(reward)
    return np.array(actions), np.array(rewards)
