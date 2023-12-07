import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import glob


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


def draw_all_2d():
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


def draw_all_3d():
    filelist = glob.glob("./real_trajectories/*.npz")
    fileindex = 0

    def readnpz(filename):
        print(filename)
        data = np.load(filename)
        agent_xyz = data['agent_xyz']
        target_xyz = data['target_xyz']
        sphere_center = data['sphere_center']
        sphere_radius = data['sphere_radius']
        plane_equation = data['plane_eqation']
        plane_pcd = data['plane_pcd']
        obstacle_pcd = data['obstacle_pcd']
        obstacle_label = data['obstacle_label']

        print(plane_equation)

        lines = []
        for i in range(agent_xyz.shape[0] - 1):
            lines.append([i, i + 1])
        lines = np.array(lines)
        colors = [[1, 0, 0] for i in range(len(lines))]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(agent_xyz)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)

        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=target_xyz)

        plane = o3d.geometry.PointCloud()
        plane.points = o3d.utility.Vector3dVector(plane_pcd)
        plane.paint_uniform_color([0, 0, 0])

        obstacle = o3d.geometry.PointCloud()
        obstacle.points = o3d.utility.Vector3dVector(obstacle_pcd)
        max_label = obstacle_label.max()
        colors = plt.get_cmap("tab20")(obstacle_label / (max_label if max_label > 0 else 1))
        colors[obstacle_label < 0] = 0
        obstacle.colors = o3d.utility.Vector3dVector(colors[:, :3])

        geometries = [line_set, plane, obstacle, mesh_frame]

        for i in range(max_label + 1):
            sphere = o3d.geometry.TriangleMesh.create_sphere(sphere_radius[i]).translate(sphere_center[i])
            sphere.paint_uniform_color(plt.get_cmap("tab20")(i / (max_label if max_label > 0 else 1))[:3])
            geometries.append(sphere)

        return geometries

    def rerun(vis):
        global fileindex
        vis.clear_geometries()
        shapes = readnpz(filelist[fileindex])
        for shape in shapes:
            vis.add_geometry(shape)

    def nextfile(vis):
        global fileindex
        vis.clear_geometries()
        if fileindex >= len(filelist) - 1:
            print("end reached")
        else:
            fileindex += 1
            shapes = readnpz(filelist[fileindex])
            for shape in shapes:
                vis.add_geometry(shape)

    def prevfile(vis):
        global fileindex
        vis.clear_geometries()
        if fileindex <= 0:
            print("start reached")
        else:
            fileindex -= 1
            shapes = readnpz(filelist[fileindex])
            for shape in shapes:
                vis.add_geometry(shape)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.clear_geometries()
    shapes = readnpz(filelist[fileindex])
    for shape in shapes:
        vis.add_geometry(shape)
    vis.register_key_callback(ord("R"), rerun)
    vis.register_key_callback(ord("A"), prevfile)
    vis.register_key_callback(ord("D"), nextfile)
    vis.run()


class drawer_2d:
    def __init__(self, trajectories=None, x_obstacles=None, r_obstacles=None,
                 x_obstacles_pred=None, r_obstacles_pred=None,
                 x_current=None, u_current=None, real_demonstration=None,
                 max_iteration=1000, filename=None):

        if filename is not None:
            data = np.load(filename)

            self.trajectories = []
            trajectory_index = 0
            while True:
                if 'trajectory' + str(trajectory_index) in data:
                    self.trajectories.append(data['trajectory' + str(trajectory_index)])
                else:
                    break
                trajectory_index += 1

            self.x_obstacles = data['x_obstacles']
            self.r_obstacles = data['r_obstacles']
            self.x_obstacles_pred = data['x_obstacles_pred']
            self.r_obstacles_pred = data['r_obstacles_pred']
            self.x_current = data['x_current']
            self.u_current = data['u_current']
            self.real_demonstration = data['real_demonstration']
            self.max_iteration = data['max_iteration']
            self.iterations = [i + 1 for i in range(self.max_iteration)]
            self.losses = data['losses']
            self.metric = data['metric']
            self.u_pred = data['u_pred']
        else:
            self.trajectories = trajectories
            self.x_obstacles = x_obstacles
            self.r_obstacles = r_obstacles
            self.x_obstacles_pred = [x_obstacles_pred]
            self.r_obstacles_pred = [r_obstacles_pred]
            self.x_current = x_current
            self.u_current = u_current
            self.real_demonstration = real_demonstration
            self.max_iteration = max_iteration
            self.iterations = []
            self.losses = []
            self.metric = []
            self.u_pred = []

        self.figure, self.axes = plt.subplots(1, 3)
        if self.real_demonstration:
            self.axes[0].set_xlim([-4.75, -4.])
            self.axes[0].set_ylim([-0.5, 1.])
        else:
            self.axes[0].set_xlim([0, 1])
            self.axes[0].set_ylim([0, 1])
        self.axes[0].grid(True)
        self.axes[0].set_aspect(1)
        self.axes[0].set_xlabel('World coordinate x [m]')
        self.axes[0].set_ylabel('World coordinate y [m]')
        self.axes[0].set_title('Demonstrated trajectories\n,'
                               ' actual/predicted obstacles\n,'
                               ' and predicted velocity vectors')

        # draw all trajectories
        if self.trajectories is not None:
            for i in range(len(self.trajectories)):
                self.axes[0].plot(self.trajectories[i][:, 0], self.trajectories[i][:, 1], '-r')

        # draw all real obstacles
        if self.x_obstacles is not None and self.r_obstacles is not None:
            for i in range(self.x_obstacles.shape[0]):
                self.axes[0].add_artist(plt.Circle(self.x_obstacles[i], self.r_obstacles[i]))

        self.pyplot_predicted_obstacle = []
        # draw all predicted obstacles
        if self.x_obstacles_pred is not None and self.r_obstacles_pred is not None:
            for i in range(self.x_obstacles_pred[-1].shape[0]):
                self.pyplot_predicted_obstacle.append(self.axes[0].add_artist(
                    plt.Circle(self.x_obstacles_pred[-1][i], self.r_obstacles_pred[-1][i], color='g')))

        if self.x_current is not None:
            if filename is None and self.u_current is not None:
                self.pyplot_predicted_input = self.axes[0].quiver(self.x_current[:, 0], self.x_current[:, 1],
                                                                  self.u_current[:, 0], self.u_current[:, 1])
            elif filename is not None and self.u_pred is not None:
                self.pyplot_predicted_input = self.axes[0].quiver(self.x_current[:, 0], self.x_current[:, 1],
                                                                  self.u_pred[-1][:, 0], self.u_pred[-1][:, 1])

        self.loss_line, = self.axes[1].plot(self.iterations, self.losses, color='k')
        self.axes[1].set_xlabel('Iterations')
        self.axes[1].set_ylabel('Loss')
        self.axes[1].set_title('Loss over iterations')
        self.axes[1].axis([1, self.max_iteration, 0, 0.3])

        self.metric_line, = self.axes[2].plot(self.iterations, self.metric, color='k')
        self.axes[2].set_xlabel('Iterations')
        self.axes[2].set_ylabel('Intersection over Predicted Circles')
        self.axes[2].set_title('Evaluation metric over iterations')
        self.axes[2].axis([1, self.max_iteration, 0, 1.0])

        self.control_x = None
        self.control_u = None

    def update_and_show(self, x_obstacles=None, r_obstacles=None, u_pred=None, iteration=None, loss=None, metric=None):
        figure_temp, _ = plt.subplots()
        plt.close(figure_temp)
        if x_obstacles is not None and r_obstacles is not None:
            for i in range(x_obstacles.shape[0]):
                self.pyplot_predicted_obstacle[i].center = x_obstacles[i]
                self.pyplot_predicted_obstacle[i].set_radius(r_obstacles[i])
            self.x_obstacles_pred.append(x_obstacles)
            self.r_obstacles_pred.append(r_obstacles)
        if u_pred is not None:
            self.pyplot_predicted_input.set_UVC(u_pred[:, 0], u_pred[:, 1])
            self.u_pred.append(u_pred)

        if iteration is not None:
            self.iterations.append(iteration)
        if loss is not None:
            self.losses.append(loss)
            self.loss_line.set_data(self.iterations, self.losses)
        if metric is not None:
            self.metric.append(metric)
            self.metric_line.set_data(self.iterations, self.metric)

        plt.show(block=False)

    def show_control(self, x, u):
        figure_temp, _ = plt.subplots()
        plt.close(figure_temp)
        if self.control_x is None:
            self.control_x, = self.axes[0].plot(x[:, 0], x[:, 1], '-b')
        else:
            self.control_x.set_data(x[:, 0], x[:, 1])
        if self.control_u is None:
            self.control_u = self.axes[0].quiver(x[:, 0], x[:, 1], u[:, 0], u[:, 1])
        else:
            self.control_u.remove()
            self.control_u = self.axes[0].quiver(x[:, 0], x[:, 1], u[:, 0], u[:, 1])
        plt.show(block=False)

    def save_model_data(self, filename):
        data = {}
        for i in range(len(self.trajectories)):
            data['trajectory' + str(i)] = self.trajectories[i]
        data['x_obstacles'] = self.x_obstacles
        data['r_obstacles'] = self.r_obstacles
        data['x_current'] = self.x_current
        data['u_current'] = self.u_current
        data['real_demonstration'] = self.real_demonstration
        data['max_iteration'] = self.max_iteration
        data['losses'] = np.array(self.losses)
        data['metric'] = np.array(self.metric)
        data['x_obstacles_pred'] = np.stack(self.x_obstacles_pred)
        data['r_obstacles_pred'] = np.stack(self.r_obstacles_pred)
        data['u_pred'] = np.stack(self.u_pred)

        np.savez(filename, **data)

    def show(self):
        plt.show(block=True)


def get_cross_prod_mat(pVec_Arr):
    # pVec_Arr shape (3)
    qCross_prod_mat = np.array([
        [0, -pVec_Arr[2], pVec_Arr[1]],
        [pVec_Arr[2], 0, -pVec_Arr[0]],
        [-pVec_Arr[1], pVec_Arr[0], 0],
    ])
    return qCross_prod_mat


def caculate_align_mat(pVec_Arr):
    scale = np.linalg.norm(pVec_Arr)
    pVec_Arr = pVec_Arr / scale
    # must ensure pVec_Arr is also a unit vec.
    z_unit_Arr = np.array([0, 0, 1])
    z_mat = get_cross_prod_mat(z_unit_Arr)

    z_c_vec = np.matmul(z_mat, pVec_Arr)
    z_c_vec_mat = get_cross_prod_mat(z_c_vec)

    if np.dot(z_unit_Arr, pVec_Arr) == -1:
        qTrans_Mat = -np.eye(3, 3)
    elif np.dot(z_unit_Arr, pVec_Arr) == 1:
        qTrans_Mat = np.eye(3, 3)
    else:
        qTrans_Mat = np.eye(3, 3) + z_c_vec_mat + np.matmul(z_c_vec_mat,
                                                            z_c_vec_mat) / (1 + np.dot(z_unit_Arr, pVec_Arr))

    qTrans_Mat *= scale
    return qTrans_Mat


def draw_prediction_3d(trajectories, x_target,
                       x_obstacles, r_obstacles,
                       x_obstacles_pred, r_obstacles_pred, x_current, u_pred,
                       plane_pcd, obstacle_pcd, obstacle_label):
    geometries = []
    for trajectory in trajectories:
        lines = []
        for i in range(trajectory.shape[0] - 1):
            lines.append([i, i + 1])
        lines = np.array(lines)
        colors = [[1, 0, 0] for i in range(len(lines))]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(trajectory)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(line_set)

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=x_target)

    plane = o3d.geometry.PointCloud()
    plane.points = o3d.utility.Vector3dVector(plane_pcd)
    plane.paint_uniform_color([0, 0, 0])

    obstacle = o3d.geometry.PointCloud()
    obstacle.points = o3d.utility.Vector3dVector(obstacle_pcd)
    max_label = obstacle_label.max()
    colors = plt.get_cmap("tab20")(obstacle_label / (max_label if max_label > 0 else 1))
    colors[obstacle_label < 0] = 0
    obstacle.colors = o3d.utility.Vector3dVector(colors[:, :3])

    geometries += [plane, obstacle, mesh_frame]

    for i in range(max_label + 1):
        sphere = o3d.geometry.TriangleMesh.create_sphere(r_obstacles[i]).translate(x_obstacles[i])
        sphere.paint_uniform_color(plt.get_cmap("tab20")(i / (max_label if max_label > 0 else 1))[:3])
        geometries.append(sphere)

    for i in range(x_obstacles_pred.shape[0]):
        sphere = o3d.geometry.TriangleMesh.create_sphere(r_obstacles_pred[i]).translate(x_obstacles_pred[i])
        sphere.paint_uniform_color([0, 1, 0])
        geometries.append(sphere)

    from draw_arrow import get_arrow
    temp = np.concatenate(trajectories)
    for i in range(u_pred.shape[0]):
        mesh_arrow = get_arrow(temp[i], vec=u_pred[i])
        mesh_arrow.paint_uniform_color([0, 0, 1])
        geometries.append(mesh_arrow)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.clear_geometries()
    for geometry in geometries:
        vis.add_geometry(geometry)
    vis.run()
