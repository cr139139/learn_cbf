import torch
import torch.nn as nn
from torch.autograd import Function, Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from qpth.qp import QPFunction
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.style.use('bmh')
torch.set_default_dtype(torch.float64)


class OptNet(nn.Module):
    def __init__(self, circles, radius):
        super().__init__()
        x = torch.linspace(0.1, 0.9, 5)
        y = torch.linspace(0.1, 0.9, 5)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        grid = torch.stack([grid_x, grid_y]).T.reshape((-1, 2))

        # self.circles = Parameter(circles)
        # self.radius = Parameter(radius)

        self.circles = Parameter(grid)
        self.radius = Parameter(torch.ones(grid.shape[0]) * 0.01)
        self.lbd = Parameter(torch.ones(1) * 1.0)
        self.alpha = Parameter(torch.ones(1) * 1.0)

    def forward(self, x, xt):
        x_xt = x - xt

        lbd = self.lbd
        alpha = self.alpha

        n = x.shape[0]
        n_obstacles = self.circles.shape[0]

        # <=
        clf_l = torch.cat([2 * x_xt, torch.ones(n, 1)], dim=1)
        clf_r = -lbd * torch.linalg.norm(x_xt, dim=1, keepdim=True) ** 2

        # >=
        x_xobs = x[:, None, :] - self.circles[None, :, :]
        x_xobs_norm = torch.linalg.norm(x_xobs, dim=2, keepdim=True)
        cbf_l = torch.cat([x_xobs / (x_xobs_norm + 1e-6), torch.ones(n, n_obstacles, 1)], dim=2)
        cbf_r = -alpha * (x_xobs_norm - self.radius[None, :, None])
        cbf_l = torch.swapaxes(cbf_l, 0, 1)
        cbf_r = torch.swapaxes(cbf_r, 0, 1)

        left = torch.cat([clf_l[None, :, :], -cbf_l], dim=0)
        right = torch.cat([clf_r[None, :, :], -cbf_r], dim=0)

        Q = torch.eye(3)
        p = 0.05
        Q[2::3, 2::3] *= p
        e = torch.Tensor()
        u = []

        for i in range(n):
            left_input_limit = torch.zeros((4, 3))
            left_input_limit[0, 0] = 1
            left_input_limit[1, 0] = -1
            left_input_limit[2, 1] = 1
            left_input_limit[3, 1] = -1
            right_input_limit = torch.ones(4) * 0.05

            left_var = torch.zeros((1, 3))
            left_var[0, 2] = -1
            right_var = torch.zeros(1)

            u.append(QPFunction(verbose=-1)(Q, torch.zeros(3),
                                            torch.cat([left[:, i, :], left_input_limit, left_var], dim=0),
                                            torch.cat([right[:, i, 0], right_input_limit, right_var], dim=0),
                                            e, e))

        return torch.cat(u)


data = np.load('trajectories.npz')

x = data['x']
u = data['u']
cost = data['cost']
circles = data['circle_xy']
radius = data['circle_r']

xt = x[:, -1, :]
x = x[:, :-1, :]
x = x.reshape((-1, 2))
u = u.reshape((-1, 2))
x = torch.from_numpy(x)[:99]
u = torch.from_numpy(u)[:99]
xt = torch.from_numpy(xt).repeat_interleave(99, dim=0)[:99]
circles = torch.from_numpy(circles)
radius = torch.from_numpy(radius)

# Initialize the model.
model = OptNet(circles, radius)
loss_fn = torch.nn.CosineEmbeddingLoss()
# loss_fn = torch.nn.MSELoss()
learning_rate = 1e-2
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(100):
    u_pred = model(x, xt)

    # Compute and print loss.
    loss = loss_fn(u, u_pred[:, :2], torch.ones(u.shape[0]))
    # loss = loss_fn(u, u_pred[:, :2])
    print(t, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    index = 0
    for p in model.parameters():
        if index == 0:
            index += 1
            p.data.clamp_(0.0, 1.0)
            continue
        elif index == 1:
            index += 1
            p.data.clamp_(0.01, 0.1)
        elif index == 2:
            index += 1
            print(p.data)
            p.data.clamp_(0.01)
        else:
            print(p.data)
            p.data.clamp_(0.01)


print(x.shape, u_pred.shape)


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

for i in range(model.circles.detach().numpy().shape[0]):
    axes.add_artist(plt.Circle(model.circles.detach().numpy()[i], model.radius.detach().numpy()[i]-0.01, color='g'))

plt.title('2D environment')

plt.quiver(x[0, :, 0][:99], x[0, :, 1][:99], u_pred.detach().numpy()[:99, 0], u_pred.detach().numpy()[:99, 1])
plt.show()