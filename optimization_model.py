import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from qpth.qp import QPFunction
torch.set_default_dtype(torch.float64)


class ObsOpt(nn.Module):
    def __init__(self, x_obstacles=None, r_obstacles=None, lbd=None, alpha=None, input_limit=0.05, n_grid=5, real_demonstration=False):
        super().__init__()
        self.fixed_obstacles = True
        self.fixed_lbd = True
        self.fixed_alpha = True
        self.real_demonstration = real_demonstration
        self.input_limit = input_limit

        if x_obstacles is None or r_obstacles is None:
            self.fixed_obstacles = False
            # creating 2D grid positions for circles
            if self.real_demonstration:
                x_temp = torch.linspace(-4.6, -4.15, n_grid)
                y_temp = torch.linspace(-0.1, 0.6, n_grid)
                radius = 0.01
            else:
                x_temp = torch.linspace(0.1, 0.9, n_grid)
                y_temp = torch.linspace(0.1, 0.9, n_grid)
                radius = 0.01
            grid_x, grid_y = torch.meshgrid(x_temp, y_temp, indexing='ij')
            grid = torch.stack([grid_x, grid_y]).T.reshape((-1, 2))

            # initializing the parameters
            # x_obstacles : n_obstacles x 2
            # r_obstacles : n_obstacles
            self.n_obstacles = n_grid ** 2
            self.x_obstacles = Parameter(grid)
            self.r_obstacles = Parameter(torch.ones(grid.shape[0]) * radius)
        else:
            self.n_obstacles = x_obstacles.shape[0]
            self.x_obstacles = x_obstacles
            self.r_obstacles = r_obstacles

        if lbd is None:
            self.fixed_lbd = False
            self.lbd = Parameter(torch.ones(1) * 0.01)
        else:
            self.lbd = lbd

        if alpha is None:
            self.fixed_alpha = False
            self.alpha = Parameter(torch.ones(1) * 0.01)
        else:
            self.alpha = alpha

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
        input_r = torch.ones(4) * self.input_limit

        # 0 <= Delta
        # | 0  0  -1  | | u |  <= 0
        #               | d |
        delta_l = torch.zeros((1, 3))
        delta_l[0, 2] = -1
        delta_r = torch.zeros(1)

        # u Q[:2,:2] u + p * d**2
        Q = torch.eye(3)
        p = 5.0
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

    def clamp_parameters(self):
        if not self.fixed_obstacles:
            if self.real_demonstration:
                self.x_obstacles[:, 0].data.clamp_(-4.7, -4.05)
                self.x_obstacles[:, 1].data.clamp_(-0., 0.7)
                self.r_obstacles.data.clamp_(0.01, 0.1)
            else:
                self.x_obstacles.data.clamp_(0.0, 1.0)
                self.r_obstacles.data.clamp_(0.01, 0.1)
        if not self.fixed_lbd:
            self.lbd.data.clamp_(0.01)
        if not self.fixed_alpha:
            self.alpha.data.clamp_(0.01)