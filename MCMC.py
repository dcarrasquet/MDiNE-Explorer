## Scripts venant de https://www.tweag.io/blog/2019-10-25-mcmc-intro1/ pour comprendre le fonctionnement des MCMC sampling

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 6]
np.random.seed(42)

state_space = ("sunny", "cloudy", "rainy")

transition_matrix = np.array(((0.6, 0.3, 0.1),
                              (0.3, 0.4, 0.3),
                              (0.2, 0.3, 0.5)))

n_steps = 20000
states = [0]
for i in range(n_steps):
    #print(states)
    #print(transition_matrix[states[-1]])
    states.append(np.random.choice((0, 1, 2), p=transition_matrix[states[-1]]))
    
    
states = np.array(states)
print(states)

def despine(ax, spines=('top', 'left', 'right')):
    for spine in spines:
        ax.spines[spine].set_visible(False)

fig, ax = plt.subplots()
width = 1000
offsets = range(1, n_steps, 5)
#print(offsets)
for i, label in enumerate(state_space):
    ax.plot(offsets, [np.sum(states[:offset] == i) / offset
            for offset in offsets], label=label)
ax.set_xlabel("number of steps")
ax.set_ylabel("likelihood")
ax.legend(frameon=False)
despine(ax, ('top', 'right'))
plt.show()