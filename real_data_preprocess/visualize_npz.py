import numpy as np
import open3d as o3d
import glob
import matplotlib.pyplot as plt

filelist = glob.glob("../real_trajectories/*/*.npz")
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
    print(plane_pcd.min(axis=0), plane_pcd.max(axis=0))
    print(obstacle_pcd.min(axis=0), obstacle_pcd.max(axis=0))

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
