import torch
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('bmh')
torch.set_default_dtype(torch.float64)

from optimization_model import ObsOpt
from data_loading_tool import data_load
from draw_tool import draw_prediction, draw_prediction_3d

# Load dataset
# real_demonstration = False
# filename = './2d_trajectories/trajectory_0.npz'
# trajectories, x_obstacles, r_obstacles, n_obstacles, x_current, u_current, x_target = data_load(filename, real_demonstration=real_demonstration)
# input_limit = 0.05

real_demonstration = True
filename = './real_trajectories/obs3.npz'
trajectories, x_obstacles, r_obstacles, n_obstacles, \
    x_current, u_current, x_target, \
    plane_pcd, obstacle_pcd, obstacle_label, height = data_load(filename, real_demonstration=real_demonstration)
input_limit = np.abs(u_current).max()

x_obstacles = torch.from_numpy(x_obstacles)
r_obstacles = torch.from_numpy(r_obstacles)
x_current = torch.from_numpy(x_current)
u_current = torch.from_numpy(u_current)
x_target = torch.from_numpy(x_target)

# Downsample training datapoints
# n_datapoints = 99
# x_current = x_current[:n_datapoints]
# u_current = u_current[:n_datapoints]
# x_target = x_target[:n_datapoints]

# Initialize the model.
model = ObsOpt(x_obstacles=x_obstacles, r_obstacles=r_obstacles, real_demonstration=real_demonstration, input_limit=input_limit)
loss_fn = torch.nn.CosineEmbeddingLoss()
learning_rate = 1e-2
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(1):
    u_pred, collision_loss = model(x_current, x_target)

    # compute loss
    loss = loss_fn(u_current, u_pred, torch.ones(u_current.shape[0]))
    loss += collision_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.clamp_parameters()
    print("iteration: ", t, ", loss: ", loss.item())

draw_prediction(trajectories,
                x_obstacles, r_obstacles,
                model.x_obstacles.detach().numpy(), model.r_obstacles.detach().numpy(),
                x_current.detach().numpy(), u_pred.detach().numpy(), real_demonstration=real_demonstration)

# 3D visualization for real demonstration
if real_demonstration:
    x_target = x_target.detach().numpy()[0].tolist() + [height]
    x_obstacles = np.concatenate([x_obstacles, np.ones((x_obstacles.shape[0], 1)) * height], axis=1)
    x_obstacles_pred = model.x_obstacles.detach().numpy()
    r_obstacles_pred = model.r_obstacles.detach().numpy()
    x_obstacles_pred = np.concatenate([x_obstacles_pred, np.ones((x_obstacles_pred.shape[0], 1)) * height], axis=1)
    x_current = x_current.detach().numpy()
    u_pred = u_pred.detach().numpy()
    x_current = np.concatenate([x_current, np.ones((x_current.shape[0], 1)) * height], axis=1)
    u_pred = np.concatenate([u_pred, np.zeros((u_pred.shape[0], 1))], axis=1)

    draw_prediction_3d(trajectories, x_target,
                       x_obstacles, r_obstacles,
                       x_obstacles_pred, r_obstacles_pred,
                       x_current, u_pred,
                       plane_pcd, obstacle_pcd, obstacle_label)