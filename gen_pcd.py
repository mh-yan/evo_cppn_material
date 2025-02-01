import numpy as np
import os
import neat
import tools.utils as utils
import copy
from tools.shape import triangulation


def paralell_pcd(shapex, shapey):
    shear_mat = [
        [1, 0],
        [1 / 2, 3**0.5 / 2],
    ]
    shear_mat = np.array(shear_mat)
    l = 2 * shapex + 1
    w = 2 * shapey + 1
    x = np.linspace(0, 1, l)
    y = np.linspace(0, 1, w)
    X = np.zeros((w, l))
    Y = np.zeros((w, l))
    X[:, :] = x
    Y[:, :] = y.reshape(-1, 1)
    input_xy = np.stack((X.flatten(), Y.flatten()), axis=1)
    # input_xy = np.round(input_xy, decimals=10)
    input_xy = np.dot(input_xy, shear_mat)
    return input_xy


def paralell_pcd_6(shapex, shapey):
    e = 1e-10
    shear_mat = [
        [1, 0],
        [1 / 2, 3**0.5 / 2],
    ]
    shear_mat = np.array(shear_mat)
    l = 2 * shapex + 1
    w = 2 * shapey + 1
    x = np.linspace(0, 1, l)
    y = np.linspace(0, 1, w)
    X = np.zeros((w, l))
    Y = np.zeros((w, l))
    X[:, :] = x
    Y[:, :] = y.reshape(-1, 1)
    input_xy = np.stack((X.flatten(), Y.flatten()), axis=1)
    # input_xy = np.round(input_xy, decimals=10)
    input_xy = np.dot(input_xy, shear_mat)
    input_xy = np.round(input_xy, decimals=10)
    # input_xy = input_xy.reshape(l, w, 2)
    # # print(input_xy[0, 0, :], input_xy[0, 1, :])
    is_visit = [0 for i in range(input_xy.shape[0])]
    index_xy = {i: i for i in range(input_xy.shape[0])}
    num = 0
    # 第一对称
    for i, p1 in enumerate(input_xy):
        min_dis = {}
        for j, p2 in enumerate(input_xy):
            if np.abs(
                np.abs(utils.d1(p1[0], p1[1])) - np.abs(utils.d1(p2[0], p2[1]))
            ) <= e and np.abs((p2[1] - p1[1]) / (p2[0] - p1[0] + 1e-30) <= 1e-5):
                if (
                    p2[0] > p1[0]
                    and p2[1] <= utils.diag_line(p2[0]) + 1e-5
                    and is_visit[j] != 1
                    and is_visit[i] != 1
                ):
                    min_dis[j] = np.abs(
                        ((p2[1] - p1[1]) / (p2[0] - p1[0] + 1e-30)) * (3**0.5) - 1
                    )
        if len(min_dis) != 0:
            min_key = min(min_dis, key=min_dis.get)
            index_xy[min_key] = i
            is_visit[i] = 1
            is_visit[min_key] = 1
    is_visit = [0 for i in range(input_xy.shape[0])]
    # 第二对称
    for i, p1 in enumerate(input_xy):
        min_dis = {}
        for j, p2 in enumerate(input_xy):
            if np.abs(
                np.abs(utils.d2(p1[0], p1[1])) - np.abs(utils.d2(p2[0], p2[1]))
            ) <= e and np.abs(
                ((p2[1] - p1[1]) / (p2[0] - p1[0] + 1e-30)) * 1 / 3**0.5 + 1 <= 1e-5
            ):
                if (
                    p2[1] < p1[1]
                    and p2[1] <= utils.diag_line(p2[0]) + 1e-5
                    and is_visit[j] != 1
                    and is_visit[i] != 1
                ):
                    min_dis[j] = np.abs(
                        ((p2[1] - p1[1]) / (p2[0] - p1[0] + 1e-30)) * (3**0.5) - 1
                    )
        if len(min_dis) != 0:
            min_key = min(min_dis, key=min_dis.get)
            index_xy[min_key] = i
            is_visit[i] = 1
            is_visit[min_key] = 1
    is_visit = [0 for i in range(input_xy.shape[0])]
    # 第三对称
    for i, p1 in enumerate(input_xy):
        min_dis = {}
        for j, p2 in enumerate(input_xy):
            if (
                np.abs(np.abs(utils.d3(p1[0], p1[1])) - np.abs(utils.d3(p2[0], p2[1])))
                <= e
                and np.abs((p2[1] - p1[1]) / (p2[0] - p1[0] + 1e-30) * 1 / 3**0.5 - 1)
                <= 1e-5
            ):
                if (
                    p2[0] < p1[0]
                    and p2[1] <= utils.diag_line(p2[0]) + 1e-5
                    and is_visit[j] != 1
                    and is_visit[i] != 1
                ):
                    min_dis[j] = np.abs(
                        ((p2[1] - p1[1]) / (p2[0] - p1[0] + 1e-30)) * (3**0.5) - 1
                    )
        if len(min_dis) != 0:
            min_key = min(min_dis, key=min_dis.get)
            index_xy[min_key] = i
            is_visit[i] = 1
            is_visit[min_key] = 1
    is_visit = [0 for i in range(input_xy.shape[0])]
    # 对角线对称
    for i, p1 in enumerate(input_xy):
        min_dis = {}
        for j, p2 in enumerate(input_xy):
            if (
                np.abs(utils.d4(p1[0], p1[1])) - np.abs(utils.d4(p2[0], p2[1])) <= 1e-9
                and np.abs(((p2[1] - p1[1]) / (p2[0] - p1[0] + 1e-30)) * (3**0.5) - 1)
                < 1e-5
            ):
                if (
                    p2[1] > p1[1]
                    and p2[1] > (utils.diag_line(p2[0]))
                    and is_visit[i] != 1
                    and is_visit[j] != 1
                ):
                    min_dis[j] = np.abs(
                        ((p2[1] - p1[1]) / (p2[0] - p1[0] + 1e-30)) * (3**0.5) - 1
                    )
        if len(min_dis) != 0:
            min_key = min(min_dis, key=min_dis.get)
            index_xy[min_key] = i
            is_visit[i] = 1
            is_visit[min_key] = 1

    # 并查集 index 映射回去
    old_input = copy.deepcopy(input_xy)
    fixed_list = {}
    for i, p in enumerate(input_xy):
        k = index_xy[i]
        oldk = i
        while k != oldk:
            oldk = k
            k = index_xy[k]
        input_xy[i] = old_input[k]
        fixed_list[k] = 1
    fixed_ps = {
        idx: input_xy[idx]
        for idx in fixed_list.keys()
        if np.abs(input_xy[idx][1] - input_xy[idx][0] * 3**0.5) <= 1e-7
    }
    fixed_ps_index = list(fixed_ps.keys())
    all_fixed = []
    for i, p in enumerate(input_xy):
        k = index_xy[i]
        oldk = i
        while k != oldk:
            oldk = k
            k = index_xy[k]
        try:
            idx = fixed_ps_index.index(k)
            all_fixed.append(i)
        except Exception:
            pass
    all_fixed_ps = [input_xy[idx] for idx in all_fixed]
    all_fixed_ps = np.array(all_fixed_ps)
    # print("has identical idx ", len(all_fixed) != len(set(all_fixed)))
    # print(f"len is{len(all_fixed_ps)}")
    # print(all_fixed)
    return input_xy, all_fixed


def paralell_pcd_rotate(shapex, shapey):
    def disP2P(x1, y1, x2, y2):
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    shear_mat = [
        [1, 0],
        [1 / 2, 3**0.5 / 2],
    ]
    shear_mat = np.array(shear_mat)
    l = 2 * shapex + 1
    w = 2 * shapey + 1
    x = np.linspace(0, 1, l)
    y = np.linspace(0, 1, w)
    X = np.zeros((w, l))
    Y = np.zeros((w, l))
    X[:, :] = x
    Y[:, :] = y.reshape(-1, 1)
    input_xy = np.stack((X.flatten(), Y.flatten()), axis=1)
    input_xy = np.dot(input_xy, shear_mat)
    old_xy = copy.deepcopy(input_xy)
    is_visit = [0 for i in range(input_xy.shape[0])]
    index_xy = {i: i for i in range(input_xy.shape[0])}
    fir_part = {}
    e = 1e-10
    midx = 0.5
    midy = 0.5 / 3**0.5

    # for i, p in enumerate(old_xy):
    #     if p[1] - 3**0.5 * p[0] < e and p[0] - 0.5 < e and p[1] - utils.vf1(p[0]) < e:
    #         fir_part[i] = p
    # second

    for i, p in enumerate(old_xy):
        if (
            p[1] - utils.dig(p[0]) <= e
            and p[0] - 0.5 >= -e
            and p[1] - utils.vf2(p[0]) <= e
        ):
            newx, newy = utils.rotate_point_around_pivot(p[0], p[1], midx, midy, 240)
            input_xy[i, 0] = newx
            input_xy[i, 1] = newy

    # third part
    for i, p in enumerate(old_xy):
        if (
            p[1] - utils.dig(p[0]) < e
            and p[1] - utils.vf1(p[0]) > -e
            and p[1] - utils.vf2(p[0]) > -e
        ):
            newx, newy = utils.rotate_point_around_pivot(p[0], p[1], midx, midy, 120)
            input_xy[i, 0] = newx
            input_xy[i, 1] = newy

    # symmetric
    is_visit = [0 for i in range(input_xy.shape[0])]
    # 对角线对称
    for i, p1 in enumerate(old_xy):
        min_dis = {}
        for j, p2 in enumerate(old_xy):
            if (
                np.abs(utils.diag_line(p1[0], p1[1]))
                - np.abs(utils.diag_line(p2[0], p2[1]))
                <= e
                and np.abs(((p2[1] - p1[1]) / (p2[0] - p1[0] + e)) * (3**0.5) - 1)
                < 1e-5
            ):
                if (
                    p2[1] > p1[1]
                    and p2[1] > (utils.dig(p2[0]))
                    and is_visit[i] != 1
                    and is_visit[j] != 1
                ):
                    min_dis[j] = np.abs(
                        ((p2[1] - p1[1]) / (p2[0] - p1[0] + e)) * (3**0.5) + 1
                    )
        if len(min_dis) != 0:
            min_key = min(min_dis, key=min_dis.get)
            is_visit[i] = 1
            is_visit[min_key] = 1
            input_xy[min_key] = input_xy[i]

    edg1 = []
    for i, p in enumerate(input_xy):
        if np.abs(p[0] - 0.5) < e and p[1] < 0.5 / 3**0.5 + e:
            edg1.append(p)
    edge2 = {}
    for i, p in enumerate(input_xy):
        if p[0] <= 0.5 + e and np.abs(p[1] - utils.vf1(p[0])) < e:
            edge2[i] = p

    for p1 in edg1:
        for i, p2 in edge2.items():
            if (
                np.abs(
                    disP2P(p1[0], p1[1], midx, midy) - disP2P(p2[0], p2[1], midx, midy)
                )
                < e
            ):
                input_xy[i] = p1
    return input_xy


# 创建一个简单的2D点云
def sym4_pcd(shapex, shapey):
    l = 2 * shapex + 1
    w = 2 * shapey + 1
    x = np.linspace(0, 1, l)
    y = np.linspace(0, 1, w)
    X = np.zeros((w, l))
    Y = np.zeros((w, l))
    X[:, :] = x
    Y[:, :] = y.reshape(-1, 1)
    input_xy = np.stack((X.flatten(), Y.flatten()), axis=1)
    for i in range(2):
        input_xy[:, i] = utils.normalize(input_xy[:, i], 0, 1)  # [-1,1]
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


# 从左到右，从下到上
def point_xy(shapex, shapey, orig_size_xy):
    l = 2 * shapex + 1
    w = 2 * shapey + 1
    x = np.linspace(0, 1, l)
    y = np.linspace(0, 1, w)
    X = np.zeros((w, l))
    Y = np.zeros((w, l))

    X[:, :] = x
    Y[:, :] = y.reshape(-1, 1)
    input_xy = np.stack((X.flatten(), Y.flatten()), axis=1)
    # normalize the input_xyz
    for i in range(2):
        input_xy[:, i] = utils.normalize(input_xy[:, i], 0, 1)  # [-1,1]
    # input_xy=square2parallel(input_xy)
    input_xy[:, 0] *= orig_size_xy[0]
    input_xy[:, 1] *= orig_size_xy[1]
    return input_xy
