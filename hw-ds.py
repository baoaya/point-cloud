import numpy as np
import open3d as o3d
import pandas as pd
import pyntcloud as PyntCloud
from matplotlib import pyplot as plt


# PCA
# x是点云矩阵，返回排序好的特征向量
def pca(x):
    x_mean = np.mean(x, axis=0)
    normalize_x = x - x_mean
    normalize_x = normalize_x.T
    h = normalize_x.dot(normalize_x.T)
    eigen_values, eigen_vectors = np.linalg.eig(h)
    return eigen_vectors


def hw_pca(x):
    u = pca(x)
    # 投影2维坐标
    projection_matrix = (u.T[:][:2]).T
    x_pca = x.dot(projection_matrix)
    return x_pca


# 计算法向量
# x为点云矩阵，n为临近点个数
def hw_surface_normal(x, n):
    point_cloud_o3d = o3d.geometry.PointCloud()
    point_cloud_o3d.points = o3d.utility.Vector3dVector(point_cloud_o3d)
    pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)  # 对点云建立kd树 方便搜索
    normals = []
    print(x.shape[0])  # 10000
    for i in range(x.shape[0]):
        # search_knn_vector_3d函数 ， 输入值[每一点，x]      返回值 [int, open3d.utility.IntVector, open3d.utility.DoubleVector]
        [_, idx, _] = pcd_tree.search_knn_vector_3d(point_cloud_o3d.points[i], n)
        # asarray和array 一样 但是array会copy出一个副本，asarray不会，节省内存
        k_nearest_point = np.asarray(point_cloud_o3d.points)[idx, :]  # 找出每一点的10个临近点，类似于拟合成曲面，然后进行PCA找到特征向量最小的值，作为法向量
        eigen_vectors = pca(k_nearest_point)
        # 取最后的那个
        normals.append(eigen_vectors[:, 2])
    return normals


# 下采样
# x为点云矩阵
# r为Voxel Grid的大小
# take为取点方式，默认为random随机取点，centroid取中心点
def hw_downsample(x, r, take="random"):
    x_max = np.max(x, axis=0)
    x_min = np.min(x, axis=0)
    d = (x_max - x_min) / r
    h = np.floor((x - x_min) / r)
    print("x_max:",x_max)
    print("x_min:",x_min)
    print("d:",d)
    print("h:",h.shape)
    voxel = {}
    is_centroid = take == "centroid"
    for i in range(h.shape[0]):
        hi = h[i, :]
        # hx + hy * dx + hz * dx * dy
        key = hi[0] + hi[1] * d[0] + hi[2] * d[0] * d[1]
        if is_centroid:
            if key in voxel.keys():
                n, point = voxel[key]
                voxel[key] = (n + 1, point + x[i, :])
            else:
                voxel[key] = (1, x[i, :])
        else:
            # random 直接覆盖
            voxel[key] = x[i, :]
    size = len(voxel)
    ret = np.zeros(shape=(size, 3))
    i = 0
    for key in voxel.keys():
        if is_centroid:
            n, point = voxel[key]
            point = point / n
        else:
            point = voxel[key]
        ret[i] = point
        i = i + 1
    return ret


def main():
    # 加载文件
    x = np.loadtxt('E:\\Documents\\Desktop\\Point Cloud\homework\\1\\data\\airplane\\airplane_0001.txt', delimiter=',')[:, 0:3]

    ds_random = hw_downsample(x, 0.1 , take="centroid")
    df = pd.DataFrame(ds_random)
    df.columns = ["x", "y", "z"]
    pc = PyntCloud.PyntCloud(df)
    p3d = pc.to_instance("open3d", mesh=False)
    p3d.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries([p3d])
    return


if __name__ == "__main__":
    main()
