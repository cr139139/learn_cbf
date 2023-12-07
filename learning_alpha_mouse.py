import torch
import numpy as np
import matplotlib.pyplot as plt
import os

cur_path = os.path.dirname(os.path.realpath(__file__))
plt.style.use('bmh')
torch.set_default_dtype(torch.float64)

from optimization_model import ObsOpt
from data_loading_tool import data_load
from draw_tool import draw_prediction_3d, drawer_2d

trial = '1'
r_obstacles = float(trial)
r_obstacles = np.array(r_obstacles).reshape((-1))
x_obstacles = np.array([0, 0]).reshape((-1, 2))
n_obstacles = x_obstacles.shape[0]

gmr_data = cur_path + './mouse_trajectories/mouse_data_1.npy'
gmr_data = np.array(gmr_data)
x_current = np.load(cur_path + "./gmr_mouse_data/mouse_gmr_1.npy", allow_pickle=True)
u_current = np.diff(x_current, axis=0).reshape((-1, 2))
x_current = x_current[:-1, :].reshape((-1, 2))
x_target = x_current[-1, :].reshape((-1, 2)).repeat(x_current.shape[0], axis=0)
input_limit = np.abs(u_current).max()

x_obstacles = torch.from_numpy(x_obstacles)
r_obstacles = torch.from_numpy(r_obstacles)
x_current = torch.from_numpy(x_current)
u_current = torch.from_numpy(u_current)
# x_target = torch.from_numpy(x_target)

print("1 Obstacle : ", x_obstacles, r_obstacles)
print("Target : ", x_target[0, :])
print("x_current :   ", x_current)

print(x_current.shape, u_current.shape, x_target.shape, x_obstacles.shape, r_obstacles.shape)

# Initialize the model.
model = ObsOpt(x_obstacles=x_obstacles, r_obstacles=r_obstacles, real_demonstration=False, input_limit=input_limit)
loss_fn = torch.nn.CosineEmbeddingLoss()
learning_rate = 1e-2
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

n_steps = 400
convergence = np.zeros((n_steps, 2))
alpha_learnt = np.zeros((n_steps, 2))
lbd_learnt = np.zeros((n_steps, 2))

for t in range(n_steps):
    u_pred, collision_loss = model(x_current, x_target)

    # compute loss
    loss = loss_fn(u_current, u_pred, torch.ones(u_current.shape[0]))
    loss += collision_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.clamp_parameters()

    convergence[t, :] = [t, loss.item()]
    alpha_learnt[t, :] = [t, model.alpha.data[0]]
    lbd_learnt[t, :] = [t, model.lbd.data[0]]

    print("iteration: ", t, ", loss: ", round(loss.item(), 6), ",   alpha: ", model.alpha.data, ",   gamma: ",
          model.lbd.data)

print("r_obstacles =  ", r_obstacles)

figure, axes = plt.subplots()
axes.plot(convergence[:, 0], convergence[:, 1], label="error")
plt.legend()
plt.show()

figure, axes = plt.subplots()
axes.plot(alpha_learnt[:, 0], alpha_learnt[:, 1], 'red', label="alpha")
axes.plot(lbd_learnt[:, 0], lbd_learnt[:, 1], 'green', label="gamma")
plt.legend()
plt.show()

# draw_prediction(trajectories,
#                 x_obstacles, r_obstacles,
#                 model.x_obstacles.detach().numpy(), model.r_obstacles.detach().numpy(),
#                 x_current.detach().numpy(), u_pred.detach().numpy(), real_demonstration=real_demonstration)

# # 3D visualization for real demonstration
# if real_demonstration:
#     x_target = x_target.detach().numpy()[0].tolist() + [height]
#     x_obstacles = np.concatenate([x_obstacles, np.ones((x_obstacles.shape[0], 1)) * height], axis=1)
#     x_obstacles_pred = model.x_obstacles.detach().numpy()
#     r_obstacles_pred = model.r_obstacles.detach().numpy()
#     x_obstacles_pred = np.concatenate([x_obstacles_pred, np.ones((x_obstacles_pred.shape[0], 1)) * height], axis=1)
#     x_current = x_current.detach().numpy()
#     u_pred = u_pred.detach().numpy()
#     x_current = np.concatenate([x_current, np.ones((x_current.shape[0], 1)) * height], axis=1)
#     u_pred = np.concatenate([u_pred, np.zeros((u_pred.shape[0], 1))], axis=1)

#     draw_prediction_3d(trajectories, x_target,
#                        x_obstacles, r_obstacles,
#                        x_obstacles_pred, r_obstacles_pred,
#                        x_current, u_pred,
#                        plane_pcd, obstacle_pcd, obstacle_label)
