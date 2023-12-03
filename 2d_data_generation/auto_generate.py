import numpy as np


def create_from_seed(seed):
    np.random.seed(seed)
    n_circle = 20
    circles = np.random.uniform(low=0.1, high=0.9, size=(n_circle, 2))
    radius = np.random.uniform(low=0.05, high=0.1, size=(n_circle,))
    return n_circle, circles, radius


def create_line(start, end):
    n_circle = 5
    circles = np.linspace(start, end, n_circle)
    radius = np.ones(n_circle) * 0.1
    return n_circle, circles, radius


examples = [create_from_seed(42), create_from_seed(3), create_from_seed(0), create_from_seed(7),
            (1, np.array([[0.5, 0.5]]), np.array([0.2])),
            create_line(np.array([0.25, 0.25]), np.array([0.75, 0.75])),
            create_line(np.array([0.2, 0.5]), np.array([0.8, 0.5]))]

from data_generation.traj_optimizer import traj_obs

for i in range(len(examples)):
    print('example '+str(i)+' started')
    n_circle, circles, radius = examples[i]
    traj, cost = traj_obs(n_circle, circles, radius)
    print(traj.shape)

    np.savez('../2d_trajectories/trajectory_'+str(i)+'.npz', trajectories=traj, costs=cost,
             n_obstacles=n_circle, x_obstacles=circles, r_obstacles=radius)
