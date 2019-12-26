#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
'''
Figure 4.1: Policy evaluation of another gridworld
'''

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.table import Table

# matplotlib.use('Agg')

WORLD_SIZE = 4
# left, up, right, down
ACTIONS = [np.array([0, -1]),
           np.array([-1, 0]),
           np.array([0, 1]),
           np.array([1, 0])]
ACTION_PROB = 0.25


def is_terminal(state):
    #???
    x, y = state
    return (x == 0 and y == 0) or (x == WORLD_SIZE - 1 and y == WORLD_SIZE - 1 ) #???


def step(state, action):
    #???
    
    if is_terminal(state):   # Put termination here is more general!!! HH
        return state, 0
    
    next_state = (np.array(state) + action).tolist()
    next_x, next_y = next_state
    
    # Boundary
    if next_x < 0 or next_y < 0 or next_x >= WORLD_SIZE or next_y >= WORLD_SIZE:
        next_state = state
        
    reward = -1
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


def compute_state_value(in_place=True, discount=1.0):
    new_state_values = np.zeros((WORLD_SIZE, WORLD_SIZE))
    iteration = 0
    #???
    while True:   # One "Delta"
        old_state_values = new_state_values.copy()
        # old_state_values = new_state_values # This is wrong!!! The reference is copyed!!!
        
        # One "sweep"
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                    new_value = 0
                    for action in ACTIONS:
                        (next_x, next_y), reward = step([i, j], action)
                
                        # Bellman equation update
                        if in_place: # Use new_state_values immediately
                            new_value += ACTION_PROB * (reward + discount * new_state_values[next_x, next_y])
                        else:
                            new_value += ACTION_PROB * (reward + discount * old_state_values[next_x, next_y])
                        
                    new_state_values[i, j] = new_value
 
        Delta = abs(old_state_values - new_state_values).max() 
        iteration += 1
        print(iteration, Delta)

        # Termination
        if Delta < 1e-4:
            break
    
    #???
    return new_state_values, iteration


def figure_4_1():
    # While the author suggests using in-place iterative policy evaluation,
    # Figure 4.1 actually uses out-of-place version.
    
    values_async, asycn_iteration = compute_state_value(in_place=True)
    values_sync, sync_iteration = compute_state_value(in_place=False)
    
    draw_image(np.round(values_sync, decimals=2))
    plt.savefig('figure_4_1_sync.png')
    
    draw_image(np.round(values_async, decimals=2))
    plt.savefig('figure_4_1_async.png')
      
    
    print('In-place: {} iterations'.format(asycn_iteration))
    print('Synchronous: {} iterations'.format(sync_iteration))

    # plt.close()


if __name__ == '__main__':
    figure_4_1()
