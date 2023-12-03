import numpy as np


def data_load(filename, real_demonstration=False):
    data = np.load(filename)

    if real_demonstration:
        trajectories = data['agent_xyz']
        trajectory_length = trajectories.shape[0]
        try:
            u_current = data['u'][:, :2]
            x_target = data['target_xyz'][np.newaxis, :2].repeat(trajectory_length, axis=0).reshape((-1, 2))
            x_current = trajectories[:, :2].reshape((-1, 2))
            trajectories_labels = data['agent_label']
            trajectories = [trajectories[trajectories_labels == index] for index in np.unique(trajectories_labels)]
            height = trajectories[0][-1, 2]
        except:
            x_target = data['target_xyz'][np.newaxis, :2].repeat(trajectory_length - 1, axis=0).reshape((-1, 2))
            x_current = trajectories[:-1, :2].reshape((-1, 2))
            u_current = np.diff(trajectories[:, :2], axis=0).reshape((-1, 2))
            height = trajectories[-1, 2]
            trajectories = trajectories[np.newaxis, :, :]

        x_obstacles = data['sphere_center'][:, :2]
        r_obstacles = data['sphere_radius']
        n_obstacles = x_obstacles.shape[0]

        plane_pcd = data['plane_pcd']
        obstacle_pcd = data['obstacle_pcd']
        obstacle_label = data['obstacle_label']

        return trajectories, x_obstacles, r_obstacles, n_obstacles, x_current, u_current, x_target, plane_pcd, obstacle_pcd, obstacle_label, height

    trajectories = data['trajectories']
    x_obstacles = data['x_obstacles']
    r_obstacles = data['r_obstacles']
    n_obstacles = data['n_obstacles']
    trajectory_length = trajectories.shape[1]
    x_target = trajectories[:, -2:-1, :].repeat(trajectory_length - 1, axis=1).reshape((-1, 2))
    x_current = trajectories[:, :-1, :].reshape((-1, 2))
    u_current = np.diff(trajectories, axis=1).reshape((-1, 2))

    return trajectories, x_obstacles, r_obstacles, n_obstacles, x_current, u_current, x_target
