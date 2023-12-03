import rosbag
import numpy as np
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d

# bag = rosbag.Bag('./obs1/2023-11-25-17-43-30.bag')
# bag = rosbag.Bag('./obs1/2023-11-25-17-43-38.bag')
# bag = rosbag.Bag('./obs3/2023-11-25-17-50-06.bag')
bag = rosbag.Bag('./obs4_3d/2023-11-25-17-54-59.bag')
topics = bag.get_type_and_topic_info()[1].keys()
types = []
for val in bag.get_type_and_topic_info()[1].values():
    types.append(val[0])

agent_xyz = []
target_xyz = []
for topic, msg, t in bag.read_messages(topics=['/tf', '/tf_static']):
    for i in msg.transforms:
        if i.child_frame_id == 'agent':
            agent_xyz.append([i.transform.translation.x, i.transform.translation.y, i.transform.translation.z])
        elif i.child_frame_id == 'target':
            target_xyz.append([i.transform.translation.x, i.transform.translation.y, i.transform.translation.z])
        # print(i.header.stamp.secs)
        # print(i.header.stamp.nsecs)
        # print(i.child_frame_id)
        # print(i.transform.translation)
        # print(i.transform.rotation)

agent_xyz = np.array(agent_xyz)
target_xyz = np.array(target_xyz).mean(axis=0)
start_xyz = agent_xyz[0]

start_index = 0
end_index = agent_xyz.shape[0] - 1
start_check = True
end_check = True
for i in range(agent_xyz.shape[0]):
    if np.linalg.norm(agent_xyz[i] - start_xyz) < 0.05 and start_check:
        start_index = i
    else:
        start_check = False
    if np.linalg.norm(agent_xyz[i] - target_xyz) > 0.08 and end_check:
        end_index = i
    else:
        end_check = False
agent_xyz = agent_xyz[start_index:end_index + 1]

lines = []
for i in range(agent_xyz.shape[0] - 1):
    lines.append([i, i + 1])
lines = np.array(lines)
colors = [[0, 0, 0] for i in range(len(lines))]
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(agent_xyz)
line_set.lines = o3d.utility.Vector2iVector(lines)
line_set.colors = o3d.utility.Vector3dVector(colors)

for topic, msg, t in bag.read_messages(topics=['/rtabmap/cloud_map']):
    data_xyz = []
    gen = pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z"))
    for data in gen:
        data_xyz.append([data[0], data[1], data[2]])
    data_xyz = np.array(data_xyz)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(data_xyz)
bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-4.75, -0.5, 0.87), max_bound=(-4., 1, 2))
pcd = pcd.crop(bbox)

plane_model, inliers = pcd.segment_plane(distance_threshold=0.02,
                                         ransac_n=3,
                                         num_iterations=1000)
[a, b, c, d] = plane_model
print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
inlier_cloud = pcd.select_by_index(inliers)
inlier_cloud.paint_uniform_color([1.0, 0, 0])
outlier_cloud = pcd.select_by_index(inliers, invert=True)

import matplotlib.pyplot as plt

labels = np.array(outlier_cloud.cluster_dbscan(eps=0.03, min_points=15, print_progress=False))
max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
outlier_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])

mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=target_xyz)

vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window()
vis.add_geometry(line_set)
# vis.add_geometry(pcd)
vis.add_geometry(inlier_cloud)
vis.add_geometry(outlier_cloud)

import pyransac3d as pyrsc

for i in range(max_label + 1):
    point = pyrsc.Point()
    center_point, inliers_point = point.fit(np.asarray(outlier_cloud.points)[labels == i], thresh=0.02,
                                            maxIteration=5000)

    sph = pyrsc.Sphere()
    center_sphere, radius_sphere, inliers_sphere = sph.fit(np.asarray(outlier_cloud.points)[labels == i], thresh=0.02,
                                                           maxIteration=5000)

    if np.linalg.norm(center_sphere - center_point) <= 0.08:
        center = center_sphere
        radius = radius_sphere
    else:
        center = center_point
        radius = np.linalg.norm(np.asarray(outlier_cloud.points)[labels == i] - center_point, axis=1).max(axis=0)

    if np.linalg.norm(agent_xyz - center, axis=1).min() < radius:
        radius = np.linalg.norm(agent_xyz - center, axis=1).min() - 0.02

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius).translate(center)
    sphere.paint_uniform_color(plt.get_cmap("tab20")(i / (max_label if max_label > 0 else 1))[:3])
    vis.add_geometry(sphere)


def hey(vis):
    print("hello")


vis.add_geometry(mesh_frame)
vis.register_key_callback(ord("S"), hey)
vis.run()
