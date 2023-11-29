import open3d as o3d
import numpy as np
import copy
from os import listdir
import os
# import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from winsound import Beep
import math
import alexutils as alx


target = r"C:\Users\amoff\Documents\Meine Textdokumente\Masterarbeit\different_slams\MULLS\demo_data\pcd\transformed\00001.pcd"
source = r"C:\Users\amoff\Documents\Meine Textdokumente\Masterarbeit\different_slams\MULLS\demo_data\pcd\transformed\merged_vehicle_frames_no_ground_kiss_icp_registered.pcd"
source = o3d.io.read_point_cloud(source)
target = o3d.io.read_point_cloud(target)

voxel_size = 1
source = source.voxel_down_sample(voxel_size)
target = target.voxel_down_sample(voxel_size)

source_down, source_fpfh = alx.preprocess_point_cloud(source, voxel_size)
target_down, target_fpfh = alx.preprocess_point_cloud(target, voxel_size)



for method in ["fast"]:
    mean_t_error = []
    mean_r_error = []
    for x, y in alx.get_circular_coordinates(10, 4):
        translation = [x,y,0]
        temp_frame = copy.deepcopy(source)

        temp_frame.translate(translation)
        ground_truth_rotation = alx.rotate(temp_frame, 0,0,0)
        
        if method == "p2p":
            evaluation = alx.compute_icp_p2p(temp_frame, target)
        if method == "p2plane":
            evaluation = alx.compute_icp_p2plane(temp_frame, target)
        if method == "global":
            # evaluation = alx.compute_global_registration_no_threshold(temp_frame, target_down, source_fpfh, target_fpfh, voxel_size)
            distance_threshold = voxel_size * 1.5
            evaluation = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(temp_frame, target_down, source_fpfh, target_fpfh, True, distance_threshold)
            
        if method == "fast":
            evaluation = alx.compute_fast_global_registration(temp_frame, target_down, source_fpfh, target_fpfh, voxel_size)
        
        if method == "multiscale":
            evaluation = alx.compute_my_multiscale_registration(source, target)
            
        print(evaluation.transformation)

        t_error, r_error = alx.compute_relative_errors(translation, ground_truth_rotation, evaluation.transformation)
        mean_t_error.append(t_error)
        mean_r_error.append(r_error)
    
    mean_t_error = np.array(mean_t_error).mean()
    mean_r_error = np.array(mean_r_error).mean()

        # temp_frame.transform(evaluation.transformation)
        # alx.draw_point_clouds([target, temp_frame])