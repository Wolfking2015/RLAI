#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Tian Jun(tianjun.cpp@gmail.com)                                #
# 2016 Artem Oboturov(oboturov@gmail.com)                             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange  # For progress bar. HH
    
#matplotlib.use('Agg')  # Agg -> non-GUI backend. HH
matplotlib.use('qt5agg')  # We can see the figure by qt5. HH

class Bandit:
    # @k_arm: # of arms
    # @epsilon: probability for exploration in epsilon-greedy algorithm
    # @initial: initial estimation for each action
    # @step_size: constant step size for updating estimations
    # @sample_averages: if True, use sample averages to update estimations instead of constant step size
    # @UCB_param: if not None, use UCB algorithm to select action
    # @gradient: if True, use gradient based bandit algorithm
    # @gradient_baseline: if True, use average reward as baseline for gradient based bandit algorithm
    def __init__(self, k_arm=10, epsilon=0., initial=0., step_size=0.1, sample_averages=False, UCB_param=None,
                 gradient=False, gradient_baseline=False, true_reward=0.):
        self.k = k_arm
        self.step_size = step_size
        self.sample_averages = sample_averages
        self.indices = np.arange(self.k)
        self.time = 0
        self.UCB_param = UCB_param
        self.gradient = gradient
        self.gradient_baseline = gradient_baseline
        self.average_reward = 0
        self.true_reward = true_reward
        self.epsilon = epsilon
        self.initial = initial

    def reset(self):
        
        # real reward for each action
        # Use np.random.randn(size):
        #???
        
        #???
        
        # estimation for each action (initial values)
        #???
        
        #???
        
        # # of chosen times for each action
        self.action_count = np.zeros(self.k)

        self.best_action = np.argmax(self.q_true)

        self.time = 0

    # get an action for this bandit
    def act(self):   # Return choice
        # 1. If epsilon-greedy. Use np.random.choice(N)
        #???
        
        #???

        # 2. If using UCB
        if self.UCB_param is not None:
            # UCB_estimation = ???            
            
            # Return choice. Use np.where(). Note that argmax only return the first occurrence
            return #???

        # 3. If using stochastic GA
        if self.gradient:
            # Calculate pi using q_estimation, do softmax
            #???
            
            #???
            return #???
        
        # 4. Else, do hardmax
        return #???

    # Update value/preference estimation
    # take an action, update estimation for this action
    
    def step(self, action):
        # generate the reward under N(real reward, 1)
        reward = #???
        self.time += 1
        self.action_count[action] += 1
        
        # Incrementally track the average reward to date
        self.average_reward += #???

        if self.sample_averages:
            
            # update estimation using sample averages
            #???
            
            #???
        
        elif self.gradient:
            
            # If use baseline
            if self.gradient_baseline:
                baseline = self.average_reward
            else:
                baseline = 0
            
            # Stochastic GA. (Eq 2.12)
            one_hot = np.zeros(self.k)
            one_hot[action] = 1
            #???
            
            #???
        
        else:
            # update estimation with constant step size
            #???
            
            #???
        
        return reward


def simulate(runs, time, bandits):   # Run stimulations. Return best_action_counts and total rewards. HH
    
    rewards = np.zeros((len(bandits), runs, time))
    best_action_counts = np.zeros(rewards.shape)
    
    for i, bandit in enumerate(bandits):
        for r in trange(runs):     # trange: progress bar. HH
            
            # Perform one run
            bandit.reset()
            for t in range(time):
                
                # act, step
                #???
                
                #???
                
                # get outcomes (reward and if-best-action)
                #???
                
                #???
                
                    
    # Calculate mean_best_action_counts and reward. HH
    #???                
    
    
    #???
    
    return mean_best_action_counts, mean_rewards


def figure_2_1():
    plt.violinplot(dataset=np.random.randn(200, 10) + np.random.randn(10))
    plt.xlabel("Action")
    plt.ylabel("Example of Reward distribution")
    plt.savefig('../images/figure_2_1.png')
#    plt.close()


def figure_2_2(runs=2000, time=1000):
    
    epsilons = [0, 0.1, 0.01]
    
    # Generate a series of Bandit objects using different eps. HH
    #???
    
    #???
    
    # Run simulations, return best_action_counts and rewards. HH
    #???
    
    #???

    plt.figure(figsize=(10, 20))

    plt.subplot(2, 1, 1)
    
    # Plotting average rewards. Use zip(epsilons, rewards), and plt.plot(rewards, label = 'xxx %X %X' %(X,X))
    #???
    
    #???
    
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()

    plt.subplot(2, 1, 2)
    
    # Plotting % best actions
    #???
    
    #???    
    
    plt.xlabel('steps')
    plt.ylabel('% optimal action')
    plt.legend()

    plt.savefig('../images/figure_2_2.png')
#    plt.close()


def figure_2_3(runs=2000, time=1000):
    
    # Compare optimistic intial value VS epsi-greedy
    #???

    #???
    
    plt.xlabel('Steps')
    plt.ylabel('% optimal action')
    plt.legend()

    plt.savefig('../images/figure_2_3.png')
#    plt.close()


def figure_2_4(runs=2000, time=1000):

    # Compare UCB VS epsi-greedy
    #???

    #???

    plt.xlabel('Steps')
    plt.ylabel('Average reward')
    plt.legend()

    plt.savefig('../images/figure_2_4.png')
#    plt.close()


def figure_2_5(runs=2000, time=1000):
    
    # Compare gradient ascent bandits, 0.1 & 0.4 stepsize, w or w/o baseline
    #???
    #???
        
    plt.xlabel('Steps')
    plt.ylabel('% Optimal action')
    plt.legend()

    plt.savefig('../images/figure_2_5.png')
#    plt.close()


def figure_2_6(runs=2000, time=1000):
    labels = ['epsilon-greedy', 'gradient bandit',
              'UCB', 'optimistic initialization']
    
    # Use lambda to define "superparameters" for a model
    generators = [lambda epsilon: #???
                  lambda alpha: #???
                  lambda coef: #??? 
                  lambda initial: #???  
                 ]
                      
    parameters = [np.arange(-7, -1, dtype=np.float),
                  np.arange(-5, 2, dtype=np.float),
                  np.arange(-4, 3, dtype=np.float),
                  np.arange(-2, 3, dtype=np.float)]

    # Generate bandits. Use zip
    bandits = []
    
    #???
    
    #???
    
    # Run simulation. Get rewards.
    #???
    
    #???
        
    # Plot curves. Use for loop
    #???
    
    #???
    
    plt.xlabel('Parameter(2^x)')
    plt.ylabel('Average reward')
    plt.legend()

    plt.savefig('../images/figure_2_6.png')
#    plt.close()


if __name__ == '__main__':
#    figure_2_1()
    figure_2_2()
#    figure_2_3()
#    figure_2_4()
#    figure_2_5()
#    figure_2_6()