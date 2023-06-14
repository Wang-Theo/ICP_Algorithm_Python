#!/usr/bin/env python

# If run in vscode, change the path to site-packages in anaconda env
# import sys
# sys.path.append('/home/<your_workspace_name>/anaconda3/lib/python3.8/site-packages')

import numpy as np
import open3d as o3d
import datetime
# import SVD function
from numpy import linalg as la

def icp_core(points1, points2):
    """
    solve transformation from points1 to points2, points of the same index are well matched
    :param points1: numpy array of points1, size = nx3, n is num of point
    :param points2: numpy array of points2, size = nx3, n is num of point
    :return: transformation matrix T, size = 4x4
    """
    assert points1.shape == points2.shape, 'point could size not match'

    # Initialize transformation matrix T
    T = np.zeros(shape=(4, 4))
    T[0:3, 0:3] = np.eye(3)
    T[3, 3] = 1

    # Todo: step1: calculate centroid
    centroid1 = np.mean(points1, axis=0)   # get mean x,y,z value of points1
    centroid2 = np.mean(points2, axis=0)   # get mean x,y,z value of points2

    # Todo: step2: de-centroid of points1 and points2
    p1i = (points1 - centroid1).T          # subtract centroids by points1
    p2i = (points2 - centroid2).T          # subtract centroids by points2

    # Todo: step3: compute H, which is sum of p1i'*p2i'^T
    H = np.matmul(p1i, p2i.T)              # H = p1i'*p2i'^T

    # Todo: step4: SVD of H (can use 3rd-part lib), solve R and t
    U,sigma,VT = la.svd(H)                 # U * sigma * V^T = H

    # Todo: step5, combine R and t into transformation matrix T
    R = np.matmul(VT.T,U.T)                # R = V * U^T
    t = centroid2 - np.matmul(R,centroid1) # t = centroid2 - R*centroid1
    print('------------Rotation matrix------------')
    print(R)
    print('------------translation matrix------------')
    print(t)
    # combine R and t into T
    # T = [ R t ]
    #     [ 0 1 ]
    T[0:3,0:3] = R
    T[0:3,3] = t.reshape(1,3)
    return T


def svd_based_icp_matched(points1, points2):

    # icp_core should be finished first
    T = icp_core(points1, points2)
    print('------------transformation matrix------------')
    print(T)

    # Todo: calculate transformed point cloud 1 based on T solved above, and name it pcd1_transformed (same format as point1)
    # pcd1_transformed = points1.copy() # comment this line, this only indicates the size of points_2_nearest should be the same as points1
    N = np.size(points1,0)
    pcd1_transformed = np.zeros(shape=(N, 3))
    for i in range(0,N):
        # turn every point array from [x,y,z] to [x,y,z,1]
        # multiply every point array with T 
        # and get x_transformed, y_transformed, z_transformed in first three value
        pcd1_transformed[i] = np.matmul(T, np.r_[points1[i], [1]])[0:3]

    # visualization
    mean_distance = mean_dist(pcd1_transformed, points2)
    print('mean_error= ' + str(mean_distance))
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points1)
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points2)
    pcd1_tran = o3d.geometry.PointCloud()
    pcd1_tran.points = o3d.utility.Vector3dVector(pcd1_transformed)
    pcd1.paint_uniform_color([1, 0, 0])
    pcd2.paint_uniform_color([0, 1, 0])
    pcd1_tran.paint_uniform_color([0, 0, 1])
    o3d.visualization.draw_geometries([pcd1, pcd2, pcd1_tran, axis_pcd])


def svd_based_icp_unmatched(points1, points2, n_iter, threshold):
    points_1 = points1.copy()
    points_2 = points2.copy()
    T_accumulated = np.eye(4)

    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points_2)
    pcd2.paint_uniform_color([0, 0, 1])
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(axis_pcd)
    vis.add_geometry(pcd2)

    start_time = datetime.datetime.now()

    for i in range(n_iter):

        # Todo: for all point in points_1, find nearest point in points_2, and generate points_2_nearest
        # points_2_nearest = points_1.copy() # comment this line, this only indicates the size of points_2_nearest should be the same as points_1
        N = np.size(points_1, 0)
        points_2_nearest = np.zeros(shape=(N, 3))           # initialize 
        for j in range(N):                                  # traverse points1
            dif = points_1[j] - points_2                      # difference between one point in points1 and all points in points2
            dis = np.linalg.norm(dif, axis=1)               # compute the distance array
            points_2_nearest[j] = points_2[np.argmin(dis)]   # find points1[j]'s nearest point in points2, and put it in points_2_nearest


        # solve icp
        T = icp_core(points_1, points_2_nearest)

        # Todo: update accumulated T
        # T_accumulated = ?
        T_accumulated = np.dot(T, T_accumulated)            # update


        print('-----------------------------------------')
        print('iteration = ' + str(i+1))
        print('T = ')
        print(T)
        print('accumulated T = ')
        print(T_accumulated)

        # Todo: update points_1
        points_1_homogeneous = np.append(points_1, np.ones(shape=(N, 1)), axis=1)   # homogeneous coordinate (N, 4)
        points_1_homogeneous = (np.dot(T, points_1_homogeneous.T)).T        # [(4, 4) * (4, N)]' = (N, 4)
        points_1 = points_1_homogeneous[:, 0:3]         # (N, 3)


        mean_distance = mean_dist(points_1, points2)
        print('mean_error= ' + str(mean_distance))

        # visualization
        pcd1_transed = o3d.geometry.PointCloud()
        pcd1_transed.points = o3d.utility.Vector3dVector(points_1)
        pcd1_transed.paint_uniform_color([1, 0, 0])
        vis.add_geometry(pcd1_transed)
        vis.poll_events()
        vis.update_renderer()
        vis.remove_geometry(pcd1_transed)

        if mean_distance < 0.00001 or mean_distance < threshold:
            print('fully converged!')
            break
    end_time = datetime.datetime.now()
    time_difference = (end_time - start_time).total_seconds()
    print('time cost: ' + str(time_difference) + ' s')
    vis.destroy_window()
    o3d.visualization.draw_geometries([axis_pcd, pcd2, pcd1_transed])


def mean_dist(points1, points2):
    dis_array = []
    for i in range(points1.shape[0]):
        dif = points1[i] - points2[i]
        dis = np.linalg.norm(dif)
        dis_array.append(dis)
    return np.mean(np.array(dis_array))


def main():
    pcd1 = o3d.io.read_point_cloud('data/bunny1.ply')
    pcd2 = o3d.io.read_point_cloud('data/bunny2.ply')
    points1 = np.array(pcd1.points)
    points2 = np.array(pcd2.points)

    # task 1:
    svd_based_icp_matched(points1, points2)
    # task 2:
    # svd_based_icp_unmatched(points1, points2, n_iter=30, threshold=0.1)

if __name__ == '__main__':
    main()
