import torch
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('bmh')
torch.set_default_dtype(torch.float16)

from optimization_model import ObsOpt
from data_loading_tool import data_load
from draw_tool import draw_prediction_3d, drawer_2d
from evaluation_tool import get_evaluation_metric

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
x_current = x_current[::10]
u_current = u_current[::10]
x_target = x_target[::10]

# Initialize the model.
model = ObsOpt(alpha=0.01, lbd=0.01, input_limit=input_limit, n_grid=10, real_demonstration=real_demonstration)
loss_fn = torch.nn.CosineEmbeddingLoss()
learning_rate = 1e-2
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

max_iteration = 50
drawer = drawer_2d(trajectories, x_obstacles.detach().numpy(), r_obstacles.detach().numpy(),
                   model.x_obstacles.detach().numpy(), model.r_obstacles.detach().numpy(),
                   x_current.detach().numpy(), u_current.detach().numpy(), real_demonstration,
                   max_iteration=max_iteration)

for iteration in range(1, max_iteration+1):
    u_pred, collision_loss = model(x_current, x_target)

    # compute loss
    loss = loss_fn(u_current, u_pred, torch.ones(u_current.shape[0]))
    loss += collision_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.clamp_parameters()
    evaluation = get_evaluation_metric(model.x_obstacles.detach().numpy(), model.r_obstacles.detach().numpy(),
                                       x_obstacles, r_obstacles, real_demonstration=real_demonstration)
    print("iteration: ", iteration, ", loss: ", loss.item(), ", (IoU SoT ToS): ", evaluation)

    drawer.update_and_show(model.x_obstacles.detach().numpy(),
                           model.r_obstacles.detach().numpy(),
                           u_pred.detach().numpy(), iteration,
                           loss.item(), evaluation[1])
drawer.save_model_data("testing.npz")
drawer.show()

# 3D visualization for real demonstration
if real_demonstration:
    x_target = x_target.detach().numpy()[0].tolist() + [height]
    x_obstacles = np.concatenate([x_obstacles, np.ones((x_obstacles.shape[0], 1)) * height], axis=1)
    x_obstacles_pred = model.x_obstacles.detach().numpy()
    r_obstacles_pred = model.r_obstacles.detach().numpy()
    x_obstacles_pred = x_obstacles_pred[r_obstacles_pred > 0]
    r_obstacles_pred = r_obstacles_pred[r_obstacles_pred > 0]
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
