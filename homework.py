import numpy as np
import open3d as o3d
import os
from matplotlib import pyplot as plt

INPUT_DIR = "/Users/jimmy/Desktop/PointCloud/homework/1/test/"
OUTPUT_DIR = "/Users/jimmy/Desktop/PointCloud/homework/1/output/"


def get_file_list(directory, ext=".txt"):
    """获取目录下所有制定扩展名的文件.
    directory 为需要获取文件的目录
    ext为文件的扩展名
    返回文件列表
    """
    ret = []
    for parent, dir_list, file_list in os.walk(directory):
        for file in file_list:
            if file.endswith(ext):
                ret.append(os.path.join(parent, file))
        for dir_name in dir_list:
            get_file_list(os.path.join(parent, dir_name))
    return ret


def pca(x):
    """ PCA主成分分析
    :param x: 是点云矩阵
    :return: 排序好的特征向量
    """
    x_mean = np.mean(x, axis=0)
    normalize_x = x - x_mean
    normalize_x = normalize_x.T
    h = normalize_x.dot(normalize_x.T)
    eigen_values, eigen_vectors = np.linalg.eig(h)
    sort = eigen_values.argsort()[::-1]  # 降序排列
    eigen_vectors = eigen_vectors[:, sort]
    return eigen_vectors


def hw_pca(x):
    """
    作业的PCA部分
    :param x: 输入的点云矩阵
    :return: 降维的点云矩阵
    """
    u = pca(x)
    # 投影2维坐标
    projection_matrix = (u.T[:][:2]).T
    x_pca = x.dot(projection_matrix)
    return x_pca


def hw_surface_normal(x, n):
    """计算法向量
    :param x: 点云矩阵
    :param n: 临近点个数
    :return: 法线向量矩阵
    """
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(x)
    pcd_tree = o3d.geometry.KDTreeFlann(pc)
    normals = []
    for i in range(x.shape[0]):
        [_, idx, _] = pcd_tree.search_knn_vector_3d(pc.points[i], n)
        knp = np.asarray(pc.points)[idx, :]
        # 计算PCA
        eigen_vectors = pca(knp)
        # 取最后的那个
        normals.append(eigen_vectors[:, 2])
    return normals


def hw_downsample(x, r, take="random"):
    """ 下采样
    :param x: 点云矩阵
    :param r: Voxel Grid的大小
    :param take: 取点方式，默认为 random 随机取点，centroid 取中心点
    :return: 下采样结果的点云矩阵
    """
    x_max = np.max(x, axis=0)
    x_min = np.min(x, axis=0)
    d = (x_max - x_min) / r
    h = np.floor((x - x_min) / r)
    voxel = {}  # hash table
    is_centroid = take == "centroid"
    for i in range(h.shape[0]):
        hi = h[i, :]
        # hx + hy * dx + hz * dx * dy
        key = hi[0] + hi[1] * d[0] + hi[2] * d[0] * d[1]
        if is_centroid:
            if key in voxel.keys():
                n, point = voxel[key]
                # 先累加点的位置，最后除以个数做平均
                voxel[key] = (n + 1, point + x[i, :])
            else:
                voxel[key] = (1, x[i, :])
        else:
            # random 直接覆盖
            voxel[key] = x[i, :]
    size = len(voxel)
    ret = np.zeros(shape=(size, 3))
    for i, key in enumerate(voxel.keys()):
        if is_centroid:
            n, point = voxel[key]
            point = point / n  # 除以个数就是中心点
        else:
            point = voxel[key]
        ret[i] = point
    return ret


def show_pca(ax, x):
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.scatter(x[:, 0], x[:, 1], s=1)
    return


def show_normals(ax, x, normals):
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    for i in range(x.shape[0]):
        nor = np.zeros((2, 3))
        nor[0] = x[i]
        nor[1] = x[i] + (normals[i] * 0.02)
        ax.plot(nor[:, 0], nor[:, 1], nor[:, 2], linewidth=0.1)
    return


def show_downsample(ax, x):
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], s=1)
    return


def main():
    files = get_file_list(INPUT_DIR)
    for file in files:
        path, ext = os.path.splitext(file)
        path, name = os.path.split(path)
        fig = plt.figure()
        plt.rcParams['savefig.dpi'] = 600  # 图片像素
        plt.rcParams['figure.dpi'] = 600  # 分辨率

        # 加载文件
        x = np.loadtxt(file, delimiter=',')[:, 0:3]

        # 计算PCA
        x_pca = hw_pca(x)

        ax = fig.add_subplot(2, 2, 1)
        ax.set_title('PCA')
        show_pca(ax, x_pca)

        # 计算法向量
        normals = hw_surface_normal(x, 15)

        ax = fig.add_subplot(2, 2, 2, projection='3d')
        ax.set_title('Normals')
        show_normals(ax, x, normals)

        # 计算 下采样 随机取点
        ds_random = hw_downsample(x, 0.1, take="random")
        ax = fig.add_subplot(2, 2, 3, projection='3d')
        ax.set_title('Downsample Random')
        show_downsample(ax, ds_random)

        # 计算 下采样 中心取点
        ds_centroid= hw_downsample(x, 0.1, take="centroid")
        ax = fig.add_subplot(2, 2, 4, projection='3d')
        ax.set_title('Downsample Centroid')
        show_downsample(ax, ds_centroid)

        plt.savefig(os.path.join(OUTPUT_DIR, name + ".jpg"))
        plt.cla()
    return


if __name__ == "__main__":
    main()
