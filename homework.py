import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


# PCA
# x是矩阵，返回排序好的特征向量
def pca(x):
    x_mean = np.mean(x, axis=0)
    normalize_x = x - x_mean.T
    normalize_x = normalize_x.T
    h = normalize_x.dot(normalize_x.T)
    eigen_values, eigen_vectors = np.linalg.eig(h)
    sort = eigen_values.argsort()[::-1]
    u = eigen_vectors[:, sort]
    return u


# 加载文件
X = np.loadtxt('/Users/jimmy/Desktop/DeepLearning/homework/1/data/airplane_0027.txt', delimiter=',')[:, 0:3]
U = pca(X)
projection_matrix = (U.T[:][:2]).T
print(projection_matrix)

# 显示
# fig4 = plt.figure()
# ax4 = plt.axes(projection='3d')
# ax4.scatter(X[:, 0], X[:, 1])
# plt.show()

X_pca = X.dot(projection_matrix)
# sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1])
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.show()