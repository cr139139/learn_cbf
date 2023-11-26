import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from qpth.qp import QPFunction
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('bmh')
torch.set_default_dtype(torch.float64)


class ObsOpt(nn.Module):
    def __init__(self):
        super().__init__()
        # creating 2D grid positions for circles
        n_grid = 5
        x_temp = torch.linspace(0.1, 0.9, n_grid)
        y_temp = torch.linspace(0.1, 0.9, n_grid)
        grid_x, grid_y = torch.meshgrid(x_temp, y_temp, indexing='ij')
        grid = torch.stack([grid_x, grid_y]).T.reshape((-1, 2))

        # initializing the parameters
        # x_obstacles : n_obstacles x 2
        # r_obstacles : n_obstacles
        self.n_obstacles = n_grid ** 2
        self.x_obstacles = Parameter(grid)
        self.r_obstacles = Parameter(torch.ones(grid.shape[0]) * 0.01)
        self.lbd = Parameter(torch.ones(1) * 0.01)
        self.alpha = Parameter(torch.ones(1) * 0.01)

    def forward(self, x_current, x_target):
        # x_current: n x 2
        # x_target : n x 2
        # output: u_pred: n x 2
        n = x_current.shape[0]

        # Lyapunov function: (x_target - x_current) ** 2
        # CLF: 2 * (x_current - x_target) * u <= -lambda * (x_target - x_current) ** 2 + delta
        # | 2 * (x_current - x_target) |T | u | < -lambda * (x_target - x_current) ** 2
        # |            -1              |  | d | =
        lbd = self.lbd
        clf_l = torch.cat([2 * (x_current - x_target), -torch.ones(n, 1)], dim=1)
        clf_r = -lbd * torch.linalg.norm(x_current - x_target, dim=1, keepdim=True) ** 2

        # Barrier function: ||x_current - x_obstacles||_2 - r_obstacles
        # CBF: (x_current - x_obstacles) / ||x_current - x_obstacles||_2 * u >= -alpha * (||x_current - x_obstacles||_2 - r_obstacles)
        # | (x_current - x_obstacles) / ||x_current - x_obstacles||_2 |T | u | > -alpha * (||x_current - x_obstacles||_2 - r_obstacles)
        # |                            0                              |  | d | =
        alpha = self.alpha
        r_obstacles = self.r_obstacles
        xcur_xobs = x_current[:, None, :] - self.x_obstacles[None, :, :]
        xcur_xobs_norm = torch.linalg.norm(xcur_xobs, dim=2, keepdim=True)
        cbf_l = torch.cat([xcur_xobs / (xcur_xobs_norm + 1e-6), torch.zeros(n, self.n_obstacles, 1)], dim=2)
        cbf_r = -alpha * (xcur_xobs_norm - r_obstacles[None, :, None])
        cbf_l = torch.swapaxes(cbf_l, 0, 1)
        cbf_r = torch.swapaxes(cbf_r, 0, 1)

        # Input limit: -0.05 <= u <= 0.05
        # |  1  0  0  | | u |  <= | 0.05 |
        # | -1  0  0  | | d |     | 0.05 |
        # |  0  1  0  |           | 0.05 |
        # |  0 -1  0  |           | 0.05 |
        input_l = torch.zeros((4, 3))
        input_l[0, 0] = 1
        input_l[1, 0] = -1
        input_l[2, 1] = 1
        input_l[3, 1] = -1
        input_r = torch.ones(4) * 0.05

        # 0 <= Delta
        # | 0  0  -1  | | u |  <= 0
        #               | d |
        delta_l = torch.zeros((1, 3))
        delta_l[0, 2] = -1
        delta_r = torch.zeros(1)

        # u Q[:2,:2] u + p * d**2
        Q = torch.eye(3)
        p = 0.05
        Q[2::3, 2::3] *= p

        # no equality constraint
        e = torch.Tensor()

        u_pred = []

        for i in range(n):
            # inequality: A z <= b
            inequality_l = torch.cat([clf_l[None, i, :], -cbf_l[:, i, :], input_l, delta_l], dim=0)
            inequality_r = torch.cat([clf_r[None, i, 0], -cbf_r[:, i, 0], input_r, delta_r], dim=0)

            u_pred.append(QPFunction(verbose=-1)(Q, torch.zeros(3),
                                                 inequality_l, inequality_r,
                                                 e, e)[:, :2])

        collision_loss = F.relu(-(xcur_xobs_norm - r_obstacles[None, :, None])).sum()

        return torch.cat(u_pred), collision_loss


data = np.load('trajectories.npz')
x_all = data['x']
u_current = data['u']
x_obstacle = data['circle_xy']
r_obstacle = data['circle_r']

x_target = x_all[:, -1, :]
x_current = x_all[:, :-1, :].reshape((-1, 2))
u_current = u_current.reshape((-1, 2))

x_current = torch.from_numpy(x_current)[:99]
u_current = torch.from_numpy(u_current)[:99]
x_target = torch.from_numpy(x_target).repeat_interleave(99, dim=0)[:99]

# Initialize the model.
model = ObsOpt()
cosine = True
if cosine: loss_fn = torch.nn.CosineEmbeddingLoss()
else: loss_fn = torch.nn.MSELoss()
learning_rate = 1e-2
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(50):
    u_pred, collision_loss = model(x_current, x_target)

    # compute loss
    if cosine: loss = loss_fn(u_current, u_pred, torch.ones(u_current.shape[0]))
    else: loss = loss_fn(u_current, u_pred[:, :2])
    loss += collision_loss
    print("iteration: ", t, ", loss: ", loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.x_obstacles.data.clamp_(0.0, 1.0)
    model.r_obstacles.data.clamp_(0.01, 0.1)
    model.lbd.data.clamp_(0.01)
    model.alpha.data.clamp_(0.01)

# initialize the plot
figure, axes = plt.subplots()
axes.set_xlim([0, 1])
axes.set_ylim([0, 1])
axes.grid()
axes.set_aspect(1)

# draw all trajectories
for i in range(x_all.shape[0]):
    axes.plot(x_all[i, :, 0], x_all[i, :, 1], '-r')

# draw all real obstacles
for i in range(x_obstacle.shape[0]):
    axes.add_artist(plt.Circle(x_obstacle[i], r_obstacle[i] - 0.01))

# draw all predicted obstacles
for i in range(model.x_obstacles.detach().numpy().shape[0]):
    axes.add_artist(
        plt.Circle(model.x_obstacles.detach().numpy()[i], model.r_obstacles.detach().numpy()[i] - 0.01, color='g'))

# draw input vector of the first trajectory
plt.title('2D environment')
plt.quiver(x_all[0, :, 0][:99], x_all[0, :, 1][:99], u_pred.detach().numpy()[:99, 0], u_pred.detach().numpy()[:99, 1])
plt.show()
