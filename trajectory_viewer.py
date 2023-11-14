import numpy as np
import matplotlib.pyplot as plt

data = np.load('trajectories.npz')

x = data['x']
u = data['u']
cost = data['cost']
circles = data['circle_xy']
radius = data['circle_r']

n_circle = circles.shape[0]

figure, axes = plt.subplots()
axes.set_xlim([0, 1])
axes.set_ylim([0, 1])
axes.grid()
axes.set_aspect(1)

for i in range(x.shape[0]):
    axes.plot(x[i, :, 0], x[i, :, 1], '-r')

for i in range(n_circle):
    axes.add_artist(plt.Circle(circles[i], radius[i]-0.01))

plt.title('2D environment')
plt.show()