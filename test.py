import numpy as np

a = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

b = np.array([[0,2],
              [1,1],
              [0,3]])
c = a + b
# b = np.sum(a,axis=1, keepdims=True)
print(c)
# %%
a = np.array([1, 2, 3])
print(a)

# %%
a = np.array([[1, 2, 3], [4, 5, 6]])
b = a.T
print(b)
# %%
a = np.ones([2, 3], dtype=np.int32)
print(a, type)

# %%
a = np.full([3, 5], 7)
print(a)
# %%
a = np.identity(7)
a[3, -1] = 7
a[2, -1] = 7
a[1, -1] = 7
a[4, -1] = 7
a[5, -1] = 7
a[6, -1] = 7
b = a[2:, 3:]
print(b)
# %%
a = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
n = np.array([[1, 1],
              [2, 0]])
m = np.array([[1, 0],
              [0, 1]])
c = a[n, m]
print(c)

# %%
a = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

b = np.split(a, 2)
print(b)

# %%
n = np.array([[1], [1], [5]])

m = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
# a = n / m
a = n + m
print(a)
# %%
a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(np.mean(a))
pian = np.std(a, ddof = 0) # 有偏
print("std有偏计算结果：",pian)
orig = np.sqrt(((a - np.mean(a)) ** 2).sum() / a.size)
print("有偏公式计算结果：",orig)
no_pian = np.std(a, ddof = 1) # 无偏
print("std无偏计算结果：",no_pian)
orig1 = np.sqrt(((a - np.mean(a)) ** 2).sum() / (a.size - 1))
print("无偏公式计算结果：",orig1)

#%%
a = np.array([1,2,3])
b = np.array([2,5,6])
print((b - a)/3)