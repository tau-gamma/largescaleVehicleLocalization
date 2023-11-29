import open3d as o3d
import numpy as np
import copy
from os import listdir
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from winsound import Beep
import time


def rotate(point_cloud, x_degrees,y_degrees = 0, z_degrees = 0):
    R = point_cloud.get_rotation_matrix_from_xyz(np.pi/180 * np.array([z_degrees, y_degrees,x_degrees]))
    point_cloud.rotate(R)
    return R

source_folder = "C:\\Users\\amoff\\Documents\\Meine Textdokumente\\Masterarbeit\\Daten\\rotated_and_translated\\100000000\\cropped_frames\\"

target_path = "C:\\Users\\amoff\\Documents\\Meine Textdokumente\\Masterarbeit\\LiDAR_Punktwolke-RevA-201455_Vermessungsdaten\\Cleaned Export\\intersection_dirty\\intersection_dirty2_xyzi_down_transformed_compressed_ground_removed.pcd"
target_path = "C:\\Users\\amoff\\Documents\\Meine Textdokumente\\Masterarbeit\\LiDAR_Punktwolke-RevA-201455_Vermessungsdaten\\Cleaned Export\\intersection_dirty\\intersection_dirty2_xyzi_down_transformed_compressed_cropped.pcd"
target = o3d.io.read_point_cloud(target_path)
voxel_size = 1
target = target.voxel_down_sample(voxel_size)
target.estimate_normals()

filenames = listdir(source_folder)[:]
frames = [target]
for file in filenames:
    frames.append(o3d.io.read_point_cloud(source_folder + file))


# source = copy.deepcopy(frames[1])
# source.translate([5,-15,0])
# rotate(source,30,0,0)

source = copy.deepcopy(frames[1])
source.translate([5,5,0])
rotate(source,20,0,0)


colors = [[1, 0.706, 0], [0, 0.651, 0.929], [0.3, 0.351, 0.529]]
target.paint_uniform_color(colors[0])

source.paint_uniform_color(colors[1])




vis = o3d.visualization.Visualizer()
vis.create_window()

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(source)
vis.add_geometry(target)
threshold = 10
icp_iteration = 20

for i in range(icp_iteration):
    time.sleep(0.5)    # Pause 5.5 seconds
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source, target, threshold, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1))
    source.transform(reg_p2l.transformation)
    print(reg_p2l.transformation)
    vis.update_geometry(source)
    vis.poll_events()
    vis.update_renderer()
vis.destroy_window()
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)