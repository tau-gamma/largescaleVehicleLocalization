import open3d as o3d
import numpy as np
import copy

def remove_ground(point_cloud, ground_height=0):
    column = np.asarray(point_cloud.points)[:,2]
    filtered_frame = np.asarray(point_cloud.points)[column > ground_height]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_frame)
    return pcd


def remove_ground_with_infos(point_cloud, ground_height=0):
    column = np.asarray(point_cloud.points)[:,2]
    print("Number of points before filtering", column.shape[0])
    filtered_frame = np.asarray(point_cloud.points)[column > ground_height]
    print("Number of points after filtering", filtered_frame.shape[0])
    print("Reduction", (1 - filtered_frame.shape[0]/column.shape[0])*100)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_frame)
    return pcd

def crop(point_cloud, dimension = 2, lower_bound = 0, upper_bound = None):
    column = np.asarray(point_cloud.points)[:,dimension]
    filtered_frame = np.asarray(point_cloud.points)[column > lower_bound ]
    if upper_bound is not None:
        column = filtered_frame[:,dimension]
        filtered_frame = filtered_frame[column < upper_bound]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_frame)
    return pcd

def draw_point_clouds(point_clouds, black_point_cloud = None):
    pc_copy = []
    colors = [[1, 0.706, 0], [0, 0.651, 0.929], [203/255,44/255,49/255], [0.3, 0.351, 0.529]]
    for idx, x in enumerate(point_clouds):
        temp = copy.deepcopy(x)
        if idx < len(colors):
            temp.paint_uniform_color(colors[idx])
        else:
            temp.paint_uniform_color(np.random.rand(3))
        pc_copy.append(temp)
    
    if black_point_cloud is not None:
        temp = copy.deepcopy(black_point_cloud)
        temp.paint_uniform_color([0,0,0])
        pc_copy.append(temp)

    o3d.visualization.draw_geometries(pc_copy)

def compute_registration(source, target):
    max_correspondence_distance = 0.5
    evaluation = o3d.pipelines.registration.evaluate_registration( source, target, max_correspondence_distance)
    print(evaluation.fitness, evaluation.inlier_rmse, evaluation)

def transform(point_cloud, transformation):
    temp = copy.deepcopy(point_cloud) 
    return temp.transform(transformation)

def show_transformation(target, source, transformation):
    draw_point_clouds([target, transform(source, transformation)], source)
    # draw_point_clouds([target, ], source)

def rotate(point_cloud, x_degrees,y_degrees = 0, z_degrees = 0):
    R = point_cloud.get_rotation_matrix_from_xyz(np.pi/180 * np.array([z_degrees, y_degrees,x_degrees]))
    point_cloud.rotate(R)
    return R

def relative_translation_error(translation_vector1, translation_vector2):
    return np.linalg.norm(translation_vector1-translation_vector2)

def relative_rotation_error(rotation, ground_truth_rotation):
    return np.arccos((np.trace(np.matmul(rotation, ground_truth_rotation))-1)/2)

def decompose_transformation_matrix(transformation):
    translation = transformation[0:3, 3]
    rotation = transformation[0:3, 0:3]
    return translation, rotation

def compute_relative_errors(ground_truth_translation, ground_truth_rotation, transformation_matrix):
    translation_vector, rotation_matrix = decompose_transformation_matrix(transformation_matrix)
    t_error = relative_translation_error(-np.array(ground_truth_translation), translation_vector)
    r_error = relative_rotation_error(rotation_matrix,ground_truth_rotation.T)
    return t_error, r_error


def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd
    if voxel_size > 0:
        pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature( pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def get_circular_coordinates(radius, number_of_points):
    import math
    arr = [2* math.pi/number_of_points * x for x in range(number_of_points)]
    return [(math.sin(x) * radius, math.cos(x) * radius) for x in arr]

def compute_global_registration_no_threshold(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 3
    return o3d.pipelines.registration.registration_ransac_based_on_feature_matching( source_down, target_down, source_fpfh, target_fpfh, True, distance_threshold)

def compute_global_registration_ransac(source_down, target_down, source_fpfh, target_fpfh, voxel_size,  max_iteration = 100000):
    distance_threshold = voxel_size * 3
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True, distance_threshold, ransac_n = 4,
        # o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4, 
        # [   o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        #     o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold) ], 
        # o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration)
            # criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration, 0.999)
        )
    return result








def compute_global_registration_ransac_parameter_test(source_down, target_down, source_fpfh, target_fpfh, voxel_size,  max_iteration = 100000):
    distance_threshold = 10
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True, distance_threshold,
        ransac_n = 4, 
        # checkers  = [   o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9), o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold) ], 
        #     criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration, 0.99999)
        )
    return result

def compute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 3
    return o3d.pipelines.registration.registration_fgr_based_on_feature_matching( source_down, target_down, source_fpfh, target_fpfh, o3d.pipelines.registration.FastGlobalRegistrationOption( maximum_correspondence_distance=distance_threshold))
    # return o3d.pipelines.registration.registration_fgr_based_on_feature_matching( source_down, target_down, source_fpfh, target_fpfh)

def compute_generalized_icp(source, target, max_correspondence_distance = 10):
    return o3d.pipelines.registration.registration_generalized_icp( source, target, max_correspondence_distance, np.identity(4), o3d.pipelines.registration.TransformationEstimationForGeneralizedICP())

def compute_icp_p2plane(source, target, max_correspondence_distance = 10):
    return o3d.pipelines.registration.registration_icp( source, target, max_correspondence_distance, np.identity(4), o3d.pipelines.registration.TransformationEstimationPointToPlane())

def compute_icp_p2p(source, target, max_correspondence_distance = 10):
    # return o3d.pipelines.registration.registration_icp( source, target, threshold, np.identity(4), o3d.pipelines.registration.TransformationEstimationPointToPoint(),  o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1.000000e-07, relative_rmse=1.000000e-07,))#max_iteration = 2000,
    return o3d.pipelines.registration.registration_icp( source, target, max_correspondence_distance, np.identity(4), o3d.pipelines.registration.TransformationEstimationPointToPoint())

def compute_robust_registration(source, target, loss_idx, max_correspondence_distance = 10, k = 2):
    losses = [o3d.pipelines.registration.TukeyLoss(k),
        o3d.pipelines.registration.L2Loss(),
        o3d.pipelines.registration.L1Loss(),
        o3d.pipelines.registration.HuberLoss(k),
        o3d.pipelines.registration.CauchyLoss(k),
        o3d.pipelines.registration.GMLoss(k)]
    loss = losses[loss_idx]
    estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
    return o3d.pipelines.registration.registration_icp(source, target, max_correspondence_distance , np.identity(4), estimation_method)

def compute_my_multiscale_registration(source, target):
    voxel_sizes = o3d.utility.DoubleVector([4, 2, 0.5])
    # List of Convergence-Criteria for Multi-Scale ICP:
    criteria_list = [
        # o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=0.0001, relative_rmse=0.0001, max_iteration=20),
        o3d.cpu.pybind.t.pipelines.registration.ICPConvergenceCriteria(max_iteration=30 ),
        o3d.cpu.pybind.t.pipelines.registration.ICPConvergenceCriteria(max_iteration=30 ),
        o3d.cpu.pybind.t.pipelines.registration.ICPConvergenceCriteria(max_iteration=30 ),
        o3d.cpu.pybind.t.pipelines.registration.ICPConvergenceCriteria(max_iteration=30 ),
    ]
    max_correspondence_distances = o3d.cpu.pybind.utility.DoubleVector([10, 5, 2])

    cumulative_transfomation = np.eye(4)

    for voxel_size, criteria, max_correspondence in zip(voxel_sizes, criteria_list, max_correspondence_distances):
        source_tmp = source.voxel_down_sample(voxel_size)
        target_tmp = target.voxel_down_sample(voxel_size)

        method = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        evaluation = o3d.pipelines.registration.registration_icp(source_tmp, target_tmp, max_correspondence, np.identity(4),estimation_method=method)#, criteria)
        source = transform(source, evaluation.transformation)
        cumulative_transfomation = np.matmul(cumulative_transfomation, evaluation.transformation)
        
        # print(cumulative_transfomation)
        # draw_point_clouds([source_tmp, target_tmp])


    class Bunch:
        def __init__(self, **kwds):
            self.__dict__.update(kwds)

    bunch = Bunch(transformation=cumulative_transfomation)
    return bunch

def get_transform_matrices_from_kitti(src):
    def line_to_transformation_matrix(line):
        l = line.replace("\n", "")
        l = (l + " 0 0 0 1").split(" ") 
        return np.reshape(np.array(l, dtype=float), (4,4))

    with open(src) as f:
        lines = f.readlines()
    return [line_to_transformation_matrix(line) for line in lines]


def np_to_pcd(np_array):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_array)
    return pcd

def pc_norm(pc, pc2, m):
    pc = np.asarray(pc.points)
    pc2 = np.asarray(pc2.points)
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    pc2 = pc2 - centroid
    print("Centroid", centroid)
    # m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    print("Scaling factor", m)
    pc = pc / m
    pc2 = pc2 / m
    return np_to_pcd(pc), np_to_pcd(pc2)



