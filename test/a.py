import numpy as np

# 创建一个矩阵
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 提取下三角矩阵
lower_triangle = np.tril(A)

# 对下三角矩阵进行转置
transposed_lower_triangle = lower_triangle.T

# 将转置的下三角矩阵加到原矩阵上，减去对角线元素避免重复
symmetric_matrix = (
    lower_triangle + transposed_lower_triangle - np.diag(np.diag(lower_triangle))
)

print(np.diag(np.diag(lower_triangle)))
print("原始矩阵:")
print(A)

print("对称后的矩阵:")
print(symmetric_matrix)
