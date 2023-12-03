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


def draw_prediction(trajectories, x_obstacles, r_obstacles, x_obstacles_pred, r_obstacles_pred, x_current, u_pred,
                    real_demonstration=True):
    # initialize the plot
    figure, axes = plt.subplots()
    if real_demonstration:
        axes.set_xlim([-4.75, -4.])
        axes.set_ylim([-0.5, 1.])
    else:
        axes.set_xlim([0, 1])
        axes.set_ylim([0, 1])
    axes.grid(True)
    axes.set_aspect(1)

    # draw all trajectories
    for i in range(len(trajectories)):
        axes.plot(trajectories[i][:, 0], trajectories[i][:, 1], '-r')

    # draw all real obstacles
    for i in range(x_obstacles.shape[0]):
        axes.add_artist(plt.Circle(x_obstacles[i], r_obstacles[i]))

    # draw all predicted obstacles
    for i in range(x_obstacles_pred.shape[0]):
        axes.add_artist(
            plt.Circle(x_obstacles_pred[i], r_obstacles_pred[i], color='g'))

    # draw input vector of the first trajectory
    plt.title('2D environment')
    plt.quiver(x_current[:, 0], x_current[:, 1], u_pred[:, 0], u_pred[:, 1])
    plt.show()
    plt.close(figure)


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
