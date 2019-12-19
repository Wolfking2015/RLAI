#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Tian Jun(tianjun.cpp@gmail.com)                                #
# 2016 Artem Oboturov(oboturov@gmail.com)                             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################


import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange, tqdm  # For progress bar. HH

# For parallel
import ray, time
import multiprocessing as mp
    
#matplotlib.use('Agg')  # Agg -> non-GUI backend. HH
# matplotlib.use('qt5agg')  # We can see the figure by qt5. HH

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
        self.true_reward = true_reward   # As a baseline
        self.epsilon = epsilon
        self.initial = initial

    def reset(self):
        
        # real reward for each action
        # Use np.random.randn(size):
        #???
        self.q_true = np.random.randn(self.k) + self.true_reward  # q_true for each bandit = noise + baseline
        #???
        
        # estimation for each action (initial values)
        #???
        self.q_estimation = np.zeros(self.k) + self.initial
        #???
        
        # # of chosen times for each action
        self.action_count = np.zeros(self.k)

        self.best_action = np.argmax(self.q_true)

        self.time = 0

    # get an action for this bandit
    def act(self):   # Return choice
        # 1. If epsilon-greedy. Use np.random.choice(N)
        #???
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.k)
        #???

        # 2. If using UCB
        if self.UCB_param is not None:
            # UCB_estimation = ???            
            # Should add a small number (1e-5) to get initialized. HH
            UCB_estimation = self.q_estimation + \
                             self.UCB_param * np.sqrt(np.log(self.time + 1) / (self.action_count + 1e-5))
            
            # Return choice. Use np.where(). Note that argmax only return the first occurrence
            # Also note that we should use the first return of np.where, i.e., np.where(xxx)[0]
            return np.random.choice(np.where(UCB_estimation == UCB_estimation.max())[0])   #???

        # 3. If using stochastic GA
        if self.gradient:
            # Calculate pi using q_estimation, do softmax. 
            # Use np.random.choice(xxx, p = [xxx]) to specify the probabilities!
            #???
            exp_est = np.exp(self.q_estimation)
            self.pi =  exp_est / np.sum(exp_est)
            #???
            return np.random.choice(self.indices, p = self.pi) #???
        
        # 4. Else, do hardmax
        return np.random.choice(np.where(self.q_estimation == self.q_estimation.max())[0])#???

    # Update value/preference estimation
    # take an action, update estimation for this action
    
    def step(self, action):
        # generate the reward under N(real reward, 1)
        reward = self.q_true[action] + np.random.randn() #???
        self.time += 1
        self.action_count[action] += 1
        
        # Incrementally track the average reward to date
        self.average_reward += (reward - self.average_reward) / self.time #???

        if self.sample_averages:
            
            # update estimation using sample averages
            #???
            self.q_estimation[action] += (reward - self.q_estimation[action]) / self.action_count[action]
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
            self.q_estimation += self.step_size * (reward - baseline) * (one_hot - self.pi)
            #???
        
        else:
            # update estimation with constant step size
            #???
            self.q_estimation[action] += (reward - self.q_estimation[action]) * self.step_size            
            #???
        
        return reward

# Make one-run independent    
def one_run(time, bandit):
    
    # Perform one run
    bandit.reset()
    rewards_this_run = np.zeros(time)
    best_action_counts_this_run = np.zeros(time)
    
    for t in range(time):
        
        # act, step
        #???
        action = bandit.act()
        reward = bandit.step(action)
        #???
        
        # get outcomes (reward and if-best-action)
        #???
        rewards_this_run[t]= reward
        best_action_counts_this_run[t] = action == bandit.best_action
        #???
    
    return rewards_this_run, best_action_counts_this_run

# Make ray task
    
one_run_ray = ray.remote(one_run)
# This is actually equvalent to:
# @ray.remote
# def one_run_ray(*arg):
#     return one_run(*arg)

                
# tqdm progressbar for ray
def to_iterator_ray(obj_ids):
    while obj_ids:
        done, obj_ids = ray.wait(obj_ids)
        yield ray.get(done[0])   # Return the recent done id, trigger the tdqm progressbar, and continue to wait

def simulate(runs, times, bandits):   # Run stimulations. Return best_action_counts and total rewards. HH
    
    rewards = np.zeros((len(bandits), runs, times))
    best_action_counts = np.zeros(rewards.shape)
    
    for i, bandit in enumerate(bandits):
        
        # print(' Model %g: eps = %5.2g, init = %5s, Sample-aver = %5s, UCB = %5s, GA = %5s, GA_base = %5s, step_size = %5g '\
        #       % (i+1, bandit.epsilon, bandit.initial, bandit.sample_averages, bandit.UCB_param, \
        #          bandit.gradient, bandit.gradient_baseline, bandit.step_size))  # HH
        
        start = time.time()
        if method == 'Serial':
        
            for r in trange(runs):     # trange: progress bar. HH
                rewards[i,r], best_action_counts[i,r] = one_run(times, bandit) 
                
        elif method == 'ray':
            result_ids = [one_run_ray.remote(times,bandit) for r in range(runs)]
            
            # --- Use tqdm for ray ---
            for _ in tqdm(to_iterator_ray(result_ids), total=len(result_ids)):
                pass
            
            # --- Manual progress_bar for ray ---
            # Use result_ids ("future") to refer each ray.remote
            # len_finished = 0
            
            # while result_ids: 
            #     len_finished +=1
            #     done_id, result_ids = ray.wait(result_ids)   # Once anyone is done. Note that done_id here has only one object because result_ids is shrinking.
            #     rewards[i,len_finished-1], best_action_counts[i,len_finished-1] = ray.get(done_id[0])  # Put it in results
            #     aver_speed = len_finished / (time.time() - start)                                
            #     print('\r', '%g / %g, Aver: %.2f iters/s' % (len_finished, runs, aver_speed), end='')  # Print progress
            
        elif method == 'apply_sync':
            result_ids = [pool.apply_async(one_run, args=(times, bandit)) for r in range(runs)]
            result_ids = [p.get() for p in result_ids]


            
        print('\nModel %g: %s finished in %g s' % (i+1, method,time.time()-start))
                    
    # Calculate mean_best_action_counts and reward. HH
    #???                
    mean_rewards = rewards.mean(axis = 1)  # Average over runs (2nd dimension) HH
    mean_best_action_counts = best_action_counts.mean(axis = 1)
    #???
    
    return mean_best_action_counts, mean_rewards


def figure_2_1():
    plt.violinplot(dataset=np.random.randn(200, 10) + np.random.randn(10))
    plt.xlabel("Action")
    plt.ylabel("Example of Reward distribution")
    plt.savefig('figure_2_1.png')
#    plt.close()


def figure_2_2(runs=2000, time=1000):
    
    title_txt = '=== Figure 2.2: Sample-average, different eps ==='
    print(title_txt)
    
    epsilons = [0, 0.1, 0.01]
    
    # Generate a series of Bandit objects using different eps. HH
    #???
    bandits = [Bandit(epsilon=eps, sample_averages=True) for eps in epsilons]   # Use the [f(xxx) for xxx in yyy] trick. HH!!!
    #???
    
    # Run simulations, return best_action_counts and rewards. HH
    #???
    best_action, rewards = simulate(runs, time, bandits)
    #???

    plt.figure(figsize=(10, 20))

    plt.subplot(2, 1, 1)
    
    # Plotting average rewards. Use zip(epsilons, rewards), and plt.plot(rewards, label = 'xxx %X %X' %(X,X))
    #???
    for eps, rew in zip(epsilons, rewards):
        plt.plot(rew, label = 'epsilon = %2g' %eps)
    #???
    
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()

    plt.subplot(2, 1, 2)
    
    # Plotting % best actions
    #???
    for eps, best_action in zip(epsilons, best_action):
        plt.plot(best_action, label = 'epsilon = %2g' %eps)
    
    #???    
    
    plt.xlabel('steps')
    plt.ylabel('% optimal action')
    plt.legend()
    plt.title(title_txt)

    plt.savefig('figure_2_2.png')
#    plt.close()


def figure_2_3(runs=2000, time=1000):
    
    title_txt = '=== Figure 2.3: Fixed step, Optimistic initial ==='
    print(title_txt)

    # Compare optimistic intial value VS epsi-greedy
    #???
    epsilons = [0, 0, 0.1, 0.1]
    q_inits = [0, 5, 0, 5]
    bandits = [Bandit(epsilon = eps, initial = init, sample_averages = False)\
               for eps, init in zip(epsilons, q_inits)]
    
    best_action, _ = simulate(runs, time, bandits)
    
    for eps, init, best_action in zip(epsilons, q_inits, best_action):
        plt.plot(best_action, label = 'epsilon = %2g, Q_init = %2g' % (eps, init))
    #???
    
    plt.xlabel('Steps')
    plt.ylabel('% optimal action')
    plt.legend()
    plt.title(title_txt)
    
    plt.savefig('figure_2_3.png')
#    plt.close()


def figure_2_4(runs=2000, time=1000):

    title_txt = '=== Figure 2.4: Compare UCB and eps-greedy ==='
    print(title_txt)

    # Compare UCB VS epsi-greedy
    #???
    bandits = []
    bandits.append(Bandit(epsilon=0, UCB_param=2, sample_averages=True))
    bandits.append(Bandit(epsilon=0.1, UCB_param=None, sample_averages=True))
    
    _, aver_rewards = simulate(runs, time, bandits)
    
    plt.plot(aver_rewards[0], label = 'epsi = 0,   UCB = 2')
    plt.plot(aver_rewards[1], label = 'epsi = 0.1, UCB = 0')
    
    #???
    
    plt.xlabel('Steps')
    plt.ylabel('Average reward')
    plt.legend()
    plt.title(title_txt)

    plt.savefig('figure_2_4.png')
#    plt.close()


def figure_2_5(runs=2000, time=1000):
    
    title_txt = '=== Figure 2.5: Stochastic Gradient Ascent ==='
    print(title_txt)
    
    # Compare gradient ascent bandits, 0.1 & 0.4 stepsize, w or w/o baseline
    #???
    bandits = []
    bandits.append(Bandit(gradient=True, step_size=0.1, gradient_baseline=True, true_reward=4))
    bandits.append(Bandit(gradient=True, step_size=0.1, gradient_baseline=False, true_reward=4))
    bandits.append(Bandit(gradient=True, step_size=0.4, gradient_baseline=True, true_reward=4))
    bandits.append(Bandit(gradient=True, step_size=0.4, gradient_baseline=False, true_reward=4))
    best_action_counts, _ = simulate(runs, time, bandits)
    labels = ['alpha = 0.1, with baseline',
              'alpha = 0.1, without baseline',
              'alpha = 0.4, with baseline',
              'alpha = 0.4, without baseline']

    for i in range(len(bandits)):
        plt.plot(best_action_counts[i], label=labels[i])
    
    #???
        
    plt.xlabel('Steps')
    plt.ylabel('% Optimal action')
    plt.legend()
    plt.title(title_txt)

    plt.savefig('figure_2_5.png')
    plt.show()
#    plt.close()


def figure_2_6(runs=2000, time=1000):

    title_txt = '=== Figure 2.6: Parameter Scan ==='
    print(title_txt)
    labels = ['epsilon-greedy', 'gradient bandit',
              'UCB', 'optimistic initialization']
    
    # Use lambda to define "superparameters" for a model
    generators = [lambda epsilon: Bandit(sample_averages=True, epsilon=epsilon), #???
                  lambda alpha: Bandit(gradient=True, step_size=alpha, gradient_baseline=True), #???
                  lambda coef: Bandit(UCB_param=coef, epsilon=0, sample_averages=True), #??? 
                  lambda initial: Bandit(initial=initial, epsilon=0, sample_averages=False, step_size=0.1) #???  
                 ]
    
    parameters = [np.arange(-7, -1, dtype=np.float),
                  np.arange(-5, 2, dtype=np.float),
                  np.arange(-4, 3, dtype=np.float),
                  np.arange(-2, 3, dtype=np.float)]

    # Generate bandits. Use zip
    bandits = [] #???
    #???
    for generator, parameter in zip(generators, parameters):
        for para in parameter:
            bandits.append(generator(pow(2,para)))
    #???
    
    # Run simulation. Get rewards.
    #???
    _, aver_rewards = simulate(runs, time, bandits)
    rewards_each_paras = np.mean(aver_rewards, axis=1)
    #???
        
    # Plot curves. Use for loop
    #???
    i = 0
    for label, parameter in zip(labels, parameters):
        len_this = len(parameter)
        
        # Open upper limit: [i, i+len_this) !!!
        plt.plot(parameter, rewards_each_paras[i:i+len_this], label=label)
        i += len_this
    #???
    
    plt.xlabel('Parameter(2^x)')
    plt.ylabel('Average reward')
    plt.legend()
    plt.title(title_txt)

    plt.savefig('figure_2_6.png')
#    plt.close()
    
def exercise_2_5():
    
    return
        
def exercise_2_11():
        
    return


if __name__ == '__main__':
    
    # method = 'Serial'
    method = 'ray'  
    # method = 'apply_sync'
    
    if method == 'ray':
        ray.shutdown()
        ray.init(num_cpus = 16)
    elif method == 'apply_sync':
        pool = mp.Pool(processes = 16)
        
    
#    figure_2_1()
    figure_2_2(2000, 2000)
#    figure_2_3(2000,2000)
#    figure_2_4(2000,2000)
#    figure_2_5(500,1000)
    # figure_2_6(2000,1000)
    
    exercise_2_5()
    exercise_2_11()
    