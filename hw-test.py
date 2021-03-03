import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# from pyntcloud import PyntCloud
#
# cloud = PyntCloud.from_file("/Users/jimmy/Desktop/DeepLearning/homework/1/data/airplane_0027.txt",
#                             sep=",",
#                             header=0,
#                             names=["x","y","z"])
# print(cloud)
#
#
# vis = o3d.visualization.Visualizer()
# vis.create_window()
# vis.add_geometry(cloud)
# vis.run()
###########################
# source = o3d.io.read_point_cloud('/Users/jimmy/Desktop/DeepLearning/homework/1/airplane.txt', format="xyzn")
# print(source)
# o3d.visualization.draw_geometries([source])
###########################
points = np.loadtxt('/Users/jimmy/Desktop/DeepLearning/homework/1/data/airplane_0001.txt', delimiter=',')[:, 0:3]
# source = o3d.geometry.PointCloud()
# source.points = o3d.utility.Vector3dVector(points)
# print(source)



fig4 = plt.figure()
ax4 = plt.axes(projection='3d')
ax4.scatter(points[:, 0],points[:, 1],points[:, 2])
plt.show()
