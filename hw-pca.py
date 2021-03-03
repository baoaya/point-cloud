import numpy as np
import open3d as o3d
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
def hw_downsampling(x, r, take="random"):
    max = np.max(axis=0)
    min = np.min(axis=0)
    # x_min, y_min, z_min = min[0], min[1], min[2]
    # x_max, y_max, z_max = max[0], max[1], max[2]
    d = (max - min)/r
    # dx, dy, dz = (x_max - x_min) / r, (y_max - y_min) / r, (z_max - z_min) / r
    hash = {}
    X
    return


def main():
    # 加载文件
    x = np.loadtxt('/Users/jimmy/Desktop/DeepLearning/homework/1/data/airplane_0027.txt', delimiter=',')[:, 0:3]
    x_pca = hw_pca(x)
    # 显示
    plt.scatter(x_pca[:, 0], x_pca[:, 1])
    plt.show()

    normals = hw_surface_normal(x, 10)  # 10个临近点

    ds_random = hw_downsampling(x, take="random")
    ds_centroid = hw_downsampling(x, take="centroid")

    return


if __name__ == "__main__":
    main()
