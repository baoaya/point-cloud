import numpy as np
from scipy.spatial import KDTree
import timeit

INPUT_DIR = 'E:\\Documents\\Desktop\\Point Cloud\homework\\1\\data\\airplane\\airplane_0001.txt'


def search_brute_force(x, p, k):
    dic = {}  # 字典,保存结果
    worst = float('inf')  # 最差值初始为最大的float
    for i in range(x.shape[0]):
        dist = np.linalg.norm(x[i] - p)  # 计算与p的距离
        if dist < worst:
            dic[dist] = x[i]
            worst = dist
            if len(dic) > k:
                # 挤出最大的一个
                dic = sorted(dic)
                del dic [-1]
        elif len(dic) < k:
            dic[dist] = x[i]
            worst = dist
    return dic


def hw_brute_force(x, k):
    for i in range(x.shape[0]):
        p = x[i]  # 需要搜索8-nn的点
        search_brute_force(x, p, k)


def search_scipy_kdtree(x, k, leafsize):
    tree = KDTree(x, leafsize=leafsize)
    tree.query(x, k=k)


def hw_scipy(x, k, leafsize=10):
    search_scipy_kdtree(x, k, leafsize)


def main():
    x = np.loadtxt(INPUT_DIR, delimiter=',')[:, 0:3]

    list = hw_brute_force(x, 8)


if __name__ == '__main__':
    main()
