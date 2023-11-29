import open3d as o3d
import numpy as np
import copy
from os import listdir
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import alexutils as alx
from winsound import Beep
import time
import signal
import sys

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
print('Press Ctrl+C')

def visualize_registration(source, target):
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(source)
    vis.add_geometry(target)
    max_correspondence_distance = 10
    icp_iteration = 300
    
    for i in range(icp_iteration):
        reg_p2l = o3d.pipelines.registration.registration_icp(
            source, target, max_correspondence_distance , np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1))
        source.transform(reg_p2l.transformation)
        print(reg_p2l.transformation)
        vis.update_geometry(source)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.1)    # Pause 5.5 seconds
    vis.destroy_window()
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)

def visualize_robust_registration(source, target):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(source)
    vis.add_geometry(target)
    max_correspondence_distance = 1
    icp_iteration = 1000
    
    for i in range(icp_iteration):
        loss = o3d.pipelines.registration.TukeyLoss(k=5)
        p2l = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
        reg_p2l = o3d.pipelines.registration.registration_icp(
            source, target, max_correspondence_distance , np.identity(4),
            p2l,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1))
        source.transform(reg_p2l.transformation)
        print(reg_p2l.transformation)
        vis.update_geometry(source)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.1)    # Pause 5.5 seconds
    vis.destroy_window()
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)

def visualize_multiscale_registration(source, target):
    callback_after_iteration = lambda loss_log_map : print("Iteration Index: {}, Scale Index: {}, Scale Iteration Index: {}, Fitness: {}, Inlier RMSE: {},".format(
        loss_log_map["iteration_index"].item(),
        loss_log_map["scale_index"].item(),
        loss_log_map["scale_iteration_index"].item(),
        loss_log_map["fitness"].item(),
        loss_log_map["inlier_rmse"].item()))

    voxel_sizes = o3d.utility.DoubleVector([5, 1, 0.5])
    
    # List of Convergence-Criteria for Multi-Scale ICP:
    criteria_list = [
    # o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=0.0001, relative_rmse=0.0001, max_iteration=20),
    o3d.cpu.pybind.t.pipelines.registration.ICPConvergenceCriteria(max_iteration=30 ),
    o3d.cpu.pybind.t.pipelines.registration.ICPConvergenceCriteria(max_iteration=30 ),
    o3d.cpu.pybind.t.pipelines.registration.ICPConvergenceCriteria(max_iteration=30 ),
    ]

    # `max_correspondence_distances` for Multi-Scale ICP (o3d.utility.DoubleVector):
    max_correspondence_distances = o3d.cpu.pybind.utility.DoubleVector([30, 20, 10])
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()

    registration_ms_icp = o3d.t.pipelines.registration.multi_scale_icp(
                                            source,
                                            target, 
                                            voxel_sizes,
                                            criteria_list,
                                            max_correspondence_distances,
                                            # np.identity(4), 
                                            # estimation, callback_after_iteration
                                            )
    
    print("DONE")
    alx.draw_point_clouds([target, alx.transform(source, registration_ms_icp.transformation)])

def visualize_my_multiscale_registration(source, target):
    voxel_sizes = o3d.utility.DoubleVector([10, 5, 1, 0.5])
    
    # List of Convergence-Criteria for Multi-Scale ICP:
    criteria_list = [
        # o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=0.0001, relative_rmse=0.0001, max_iteration=20),
        o3d.cpu.pybind.t.pipelines.registration.ICPConvergenceCriteria(max_iteration=30 ),
        o3d.cpu.pybind.t.pipelines.registration.ICPConvergenceCriteria(max_iteration=30 ),
        o3d.cpu.pybind.t.pipelines.registration.ICPConvergenceCriteria(max_iteration=30 ),
    ]

    # `max_correspondence_distances` for Multi-Scale ICP (o3d.utility.DoubleVector):
    max_correspondence_distances = o3d.cpu.pybind.utility.DoubleVector([50, 30, 20, 10])

    for voxel_size, criteria, max_correspondence in zip(voxel_sizes, criteria_list, max_correspondence_distances):
        source_tmp = source.voxel_down_sample(voxel_size)
        target_tmp = target.voxel_down_sample(voxel_size)

        # loss = o3d.pipelines.registration.TukeyLoss(k=5)
        method = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        evaluation = o3d.pipelines.registration.registration_icp(source_tmp, target_tmp, max_correspondence, np.identity(4),estimation_method=method)#, criteria)
        # reg_p2l = o3d.pipelines.registration.registration_icp(source, target, max_correspondence_distance , np.identity(4), o3d.pipelines.registration.TransformationEstimationPointToPlane(), o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1))
        source = alx.transform(source, evaluation.transformation)
        alx.draw_point_clouds([target_tmp, alx.transform(source_tmp, evaluation.transformation)])
    
    print("Final result")
    alx.draw_point_clouds([target, source])

target = r"C:\Users\amoff\Documents\Meine Textdokumente\Masterarbeit\different_slams\MULLS\demo_data\pcd\transformed\00001.pcd"
source = r"C:\Users\amoff\Documents\Meine Textdokumente\Masterarbeit\different_slams\MULLS\demo_data\pcd\transformed\merged_vehicle_frames_no_ground.pcd"
source = o3d.io.read_point_cloud(source)
target = o3d.io.read_point_cloud(target)

voxel_size = 1
source = source.voxel_down_sample(voxel_size)
target = target.voxel_down_sample(voxel_size)
target.estimate_normals()

colors = [[1, 0.706, 0], [0, 0.651, 0.929], [0.3, 0.351, 0.529]]
target.paint_uniform_color(colors[0])
source.paint_uniform_color(colors[1])

source.translate([-30,-30,0])
alx.rotate(source,10,0,0)

alx.draw_point_clouds([target, source])


# visualize_robust_registration(source, target)
visualize_my_multiscale_registration(source, target)
# visualize_registration(source, target)
