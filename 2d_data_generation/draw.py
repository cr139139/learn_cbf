import numpy as np
import matplotlib.pyplot as plt


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

for i in range(len(examples)):
    data = np.load('../2d_trajectories/trajectory_' + str(i) + '.npz')
    trajectories = data['trajectories']
    n_obstacles = data['n_obstacles']
    x_obstacles = data['x_obstacles']
    r_obstacles = data['r_obstacles']
    print(trajectories.shape)

    figure, axes = plt.subplots()
    axes.set_xlim([0, 1])
    axes.set_ylim([0, 1])
    axes.grid()
    axes.set_aspect(1)
    for j in range(trajectories.shape[0]):
        axes.plot(trajectories[j, :, 0], trajectories[j, :, 1], '-r')
    for j in range(n_obstacles):
        axes.add_artist(plt.Circle(x_obstacles[j], r_obstacles[j]))
    plt.title('2D environment example' + str(i))
    plt.show()
    plt.close(figure)
