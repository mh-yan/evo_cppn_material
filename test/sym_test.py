import matplotlib.pyplot as plt
import numpy as np
import copy


# 创建一个简单的2D点云
def sym4_pcd(shapex=10, shapey=10):
    l = 2 * shapex + 1
    w = 2 * shapey + 1
    x = np.linspace(-1, 1, l)
    y = np.linspace(-1, 1, w)
    X = np.zeros((w, l))
    Y = np.zeros((w, l))
    X[:, :] = x
    Y[:, :] = y.reshape(-1, 1)
    input_xy = np.stack((X.flatten(), Y.flatten()), axis=1)
    #   第一象限y=x对称
    for i, p in enumerate(input_xy):
        if p[1] >= p[0] and all(p >= 0):
            temp = input_xy[i][0]
            input_xy[i][0] = input_xy[i][1]
            input_xy[i][1] = temp
        # 旋转变换s
        mid_x = (w - 1) / 2
        mid_y = (l - 1) / 2
        last_x = w - 1
        last_y = l - 1
        # 左右对称
        for j in range(l):
            for i in range(w):
                if i < mid_x:
                    input_xy[j * w + i] = input_xy[j * w + last_x - i]
        # 上下对称
        for j in range(w):
            for i in range(l):
                if i < mid_y:
                    input_xy[j + i * w] = input_xy[j + w * (last_y - i)]
    return input_xy


# 变换的是下标，而不是实际点值，点坐标需要一致
def sym_rotate(shapex, shapey):
    R_90_clw = [[0, -1], [1, 0]]
    R_90_aclw = [[0, 1], [-1, 0]]
    R_180_aclw = [[-1, 0], [0, -1]]
    l = 2 * shapex + 1
    w = 2 * shapey + 1
    x = np.linspace(-1, 1, l)
    y = np.linspace(-1, 1, w)
    X = np.zeros((w, l))
    Y = np.zeros((w, l))
    X[:, :] = x
    Y[:, :] = y.reshape(-1, 1)
    input_xy = np.stack((X.flatten(), Y.flatten()), axis=1)
    input_xy = np.round(input_xy, decimals=4)
    old_input_xy = copy.deepcopy(input_xy)

    # # 旋转 使用pcd来索引会更简单
    for p in old_input_xy:
        if all(p >= 0):
            # 4
            p_rotate_90 = np.dot(p, R_90_clw)
            p_index = np.where(
                (old_input_xy[:, 0] == p_rotate_90[0])
                & (old_input_xy[:, 1] == p_rotate_90[1])
            )
            input_xy[p_index] = p

            # 3
            p_rotate_180 = np.dot(p, R_180_aclw)
            p_index = np.where(
                (old_input_xy[:, 0] == p_rotate_180[0])
                & (old_input_xy[:, 1] == p_rotate_180[1])
            )
            input_xy[p_index] = p

            # w
            p_rotate_neg90 = np.dot(p, R_90_aclw)
            p_index = np.where(
                (old_input_xy[:, 0] == p_rotate_neg90[0])
                & (old_input_xy[:, 1] == p_rotate_neg90[1])
            )
            input_xy[p_index] = p

    return input_xy


sympcd = sym_rotate(30, 30)
sympcd = sympcd.reshape(61, 61, 2)

import pandas as pd
import numpy as np


# 创建一个空列表来存储格式化后的坐标
formatted_data = []

# 遍历 sympcd 的第一个维度
for i in range(sympcd.shape[0]):
    # 创建一个临时列表来存储当前行的坐标字符串
    row_data = []
    # 遍历 sympcd 的第二个维度
    for j in range(sympcd.shape[1]):
        # 格式化坐标对为 "(x, y)" 形式的字符串
        coord_string = f"({sympcd[i, j, 0]}, {sympcd[i, j, 1]})"
        # 将坐标字符串添加到行数据列表中
        row_data.append(coord_string)
    # 将行数据添加到总数据列表中
    formatted_data.append(row_data)

# 通过列表创建一个 DataFrame
df = pd.DataFrame(formatted_data)

# 将 DataFrame 保存为 Excel 文件
excel_path = "sympcd.xlsx"  # 指定你想要保存的文件路径和名称
df.to_excel(excel_path, index=False, header=False)

print(f"sympcd has been saved to {excel_path}")
