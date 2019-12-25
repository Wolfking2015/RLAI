#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.table import Table

# matplotlib.use('Agg')

# Initialize, use list to store states
WORLD_SIZE = 5
A_POS = [0, 1]
A_PRIME_POS = [4, 1]
B_POS = [0, 3]
B_PRIME_POS = [2, 3]
DISCOUNT = 0.9

# left, up, right, down
ACTIONS = [np.array([0, -1]),
           np.array([-1, 0]),
           np.array([0, 1]),
           np.array([1, 0])]
ACTION_PROB = 0.25


def step(state, action):
    '''
    Do one step: (s,a --> s',r)

    Returns
    -------
    next_state, reward
                    
    '''
    #???
    # Special cases
    if state == A_POS:
        return A_PRIME_POS, 10
    elif state == B_POS:
        return B_PRIME_POS, 5
    
    # Normal move
    x, y = (np.array(state) + action).tolist()
    if x < 0 or y < 0 or x >= WORLD_SIZE or y>= WORLD_SIZE:
        next_state = state
        reward = -1
    else:
        next_state = [x, y]
        reward = 0
    #???
    
    return next_state, reward


def draw_image(image):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = image.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # Add cells
    for (i, j), val in np.ndenumerate(image):
        tb.add_cell(i, j, width, height, text=val,
                    loc='center', facecolor='white')

    # Row and column labels...
    for i in range(len(image)):
        tb.add_cell(i, -1, width, height, text=i+1, loc='right',
                    edgecolor='none', facecolor='none')
        tb.add_cell(-1, i, width, height/2, text=i+1, loc='center',
                    edgecolor='none', facecolor='none')

    ax.add_table(tb)


def figure_3_2():
    '''
    Use policy evaluation (actually in Chapter 04) to evaluate the random policy

    Returns
    -------
    None.

    '''
    # Initialized
    value = np.zeros((WORLD_SIZE, WORLD_SIZE))
    step_n = 0
    
    while True:
        # keep iteration until convergence
        ''' Policy evaluation in DP'''
        #???
        # new_value = np.zeros(value.shape())
        new_value = np.zeros_like(value)   # np.zeros_like !!!  HH
        for i in range(WORLD_SIZE):  # For all states
            for j in range(WORLD_SIZE):
                for action in ACTIONS:
                    (next_i, next_j), next_reward = step([i, j], action)  # Broadcast in this way!!! HH
                    
                    # Update by Bellman equation for state value, random policy
                    new_value[i, j] += ACTION_PROB * (next_reward + DISCOUNT * value[next_i, next_j])
        #???

        # Termination        
        # Note that max() is incorrect, because max() doesn't work with ndarrays (list is ok)
        step_n += 1
        increment = np.max(np.abs(value - new_value))
        print(step_n, increment)
        if increment < 1e-5: #???  
            draw_image(np.round(new_value, decimals=2))
            plt.savefig('figure_3_2.png')
            # plt.close()
            return
        else :
            value = new_value        
        


def figure_3_5():
    '''
    Use value iteration (Chapter 04) to find the optimal value
    '''
    # value = np.zeros((WORLD_SIZE, WORLD_SIZE))
    value = np.random.randn(WORLD_SIZE, WORLD_SIZE)  # Same result for random initialization
    step_n = 0
    
    while True:
        # keep iteration until convergence
        #???
        new_value = np.zeros_like(value)   # np.zeros_like !!!  HH
        for i in range(WORLD_SIZE):  # For all states
            for j in range(WORLD_SIZE):
                next_values = []
                for action in ACTIONS:
                    (next_i, next_j), next_reward = step([i, j], action)  # Broadcast in this way!!! HH
                    next_values.append(next_reward + DISCOUNT * value[next_i, next_j])
                
                # Update by Bellman optimality equation for state value (value iteration)
                new_value[i, j] = np.max(next_values)
    
        #???
        step_n += 1
        increment = np.max(np.abs(value - new_value))
        print(step_n, increment)
        
        if increment < 1e-5:  #???
            draw_image(np.round(new_value, decimals=2))
            plt.savefig('figure_3_5.png')
            # plt.close()
            return
        else :
            value = new_value        

        
        
if __name__ == '__main__':
    # figure_3_2()
    figure_3_5()
