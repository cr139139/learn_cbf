import glob
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

trial = 'obs1'
filelist = glob.glob("./" + trial + "/*.npz")
print(filelist)

trajectories_all = []
trajectories_labels = []
plane_pcd_all = []
obstacle_pcd_all = []
obstacle_label_all = []
u_all = []

x_obstacles_mean = []
r_obstacles_mean = []
x_target_mean = []
plane_equation_mean = []

for file_index in range(len(filelist)):
    file = filelist[file_index]
    data = np.load(file)
    trajectories = data['agent_xyz']
    u = np.diff(trajectories, axis=0)
    trajectories = trajectories[:-1]
    x_obstacles = data['sphere_center']
    r_obstacles = data['sphere_radius']
    x_target = data['target_xyz']
    plane_equation = data['plane_eqation']
    plane_pcd = data['plane_pcd']
    obstacle_pcd = data['obstacle_pcd']
    obstacle_label = data['obstacle_label']

    if len(trajectories_all) != 0:
        cost_matrix = cdist(x_obstacles, x_obstacles_mean[0], 'euclidean')
        orig, alternative = linear_sum_assignment(cost_matrix)
        masks = []
        for i in range(alternative.shape[0]):
            masks.append(obstacle_label == orig[i])
        for i in range(alternative.shape[0]):
            obstacle_label[masks[i]] = alternative[i]
        x_obstacles[alternative] = x_obstacles[orig]
        r_obstacles[alternative] = r_obstacles[orig]

    trajectories_all.append(trajectories)
    trajectories_labels.append(np.ones(trajectories.shape[0]) * file_index)
    plane_pcd_all.append(plane_pcd)
    obstacle_pcd_all.append(obstacle_pcd)
    obstacle_label_all.append(obstacle_label)
    u_all.append(u)

    x_target_mean.append(x_target)
    plane_equation_mean.append(plane_equation)
    x_obstacles_mean.append(x_obstacles)
    r_obstacles_mean.append(r_obstacles)

trajectories_all = np.concatenate(trajectories_all)
trajectories_labels = np.concatenate(trajectories_labels).astype(int)
plane_pcd_all = np.concatenate(plane_pcd_all)
obstacle_pcd_all = np.concatenate(obstacle_pcd_all)
obstacle_label_all = np.concatenate(obstacle_label_all)
u_all = np.concatenate(u_all)

x_target_mean = np.stack(x_target_mean).mean(0)
plane_equation_mean = np.stack(plane_equation_mean).mean(0)
x_obstacles_mean = np.stack(x_obstacles_mean).mean(0)
r_obstacles_mean = np.stack(r_obstacles_mean).mean(0)

np.savez(trial+'.npz', agent_xyz=trajectories_all, agent_label=trajectories_labels,
         sphere_center=x_obstacles_mean, sphere_radius=r_obstacles_mean,
         target_xyz=x_target_mean, plane_eqation=plane_equation_mean,
         plane_pcd=plane_pcd_all, obstacle_pcd=obstacle_pcd_all,
         obstacle_label=obstacle_label_all, u=u_all)