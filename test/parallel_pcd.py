import matplotlib.pyplot as plt
import numpy as np
import copy


def diag_line(x, y):
    return (y + (3**0.5) * x - 3**0.5) / 2


def vf1(x):
    return -(3**0.5) / 3 * x + (3**0.5) / 3


def vf2(x):
    return ((3**0.5) / 3) * x


def dig(x):
    return -(3**0.5) * x + 3**0.5



def rotate_point_around_pivot(px, py, cx, cy, theta):
    # Convert angle from degrees to radians
    theta = np.radians(theta)

    # Step 1: Translate the point to the origin
    translated_x = px - cx
    translated_y = py - cy

    # Step 2: Apply the rotation matrix
    rotated_x = translated_x * np.cos(theta) - translated_y * np.sin(theta)
    rotated_y = translated_x * np.sin(theta) + translated_y * np.cos(theta)

    # Step 3: Translate the point back
    new_x = rotated_x + cx
    new_y = rotated_y + cy

    return new_x, new_y


def reflect_point_about_line(px, py, a, b, c):
    # Step 1: Translate to make the line pass through the origin
    d = a * px + b * py + c
    translated_x = px
    translated_y = py

    # Step 2: Rotate the line to align with the x-axis
    theta = np.arctan2(b, a)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    rotated_x = cos_theta * translated_x + sin_theta * translated_y
    rotated_y = -sin_theta * translated_x + cos_theta * translated_y

    # Step 3: Reflect about the x-axis
    reflected_x = rotated_x
    reflected_y = -rotated_y

    # Step 4: Rotate back
    final_x = cos_theta * reflected_x - sin_theta * reflected_y
    final_y = sin_theta * reflected_x + cos_theta * reflected_y

    return final_x, final_y


def paralell_pcd(shapex, shapey):
    shear_mat = [
        [1, 0],
        [1 / 2, 3**0.5 / 2],
    ]
    rotate_120 = np.array(
        [
            [-1 / 2, -(3**0.5) / 2],
            [(3**0.5) / 2, -1 / 2],
        ]
    )
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
    old_xy = copy.deepcopy(input_xy)
    is_visit = [0 for i in range(input_xy.shape[0])]
    index_xy = {i: i for i in range(input_xy.shape[0])}
    fir_part = {}
    e = 1e-10
    midx = 0.5
    midy = 0.5 / 3**0.5

    for i, p in enumerate(old_xy):
        if p[1] - 3**0.5 * p[0] < e and p[0] - 0.5 < e and p[1] - vf1(p[0]) < e:
            fir_part[i] = p
    # second

    for i, p in enumerate(old_xy):
        if p[1] - dig(p[0]) <= e and p[0] - 0.5 >= -e and p[1] - vf2(p[0]) <= e:
            newx, newy = rotate_point_around_pivot(p[0], p[1], midx, midy, 240)
            input_xy[i, 0] = newx
            input_xy[i, 1] = newy

    # third part
    for i, p in enumerate(old_xy):
        if p[1] - dig(p[0]) < e and p[1] - vf1(p[0]) > -e and p[1] - vf2(p[0]) > -e:
            newx, newy = rotate_point_around_pivot(p[0], p[1], midx, midy, 120)
            input_xy[i, 0] = newx
            input_xy[i, 1] = newy

    # symmetric
    c = 0
    is_visit = [0 for i in range(input_xy.shape[0])]
    # 对角线对称
    for i, p1 in enumerate(old_xy):
        min_dis = {}
        for j, p2 in enumerate(old_xy):
            if (
                np.abs(diag_line(p1[0], p1[1])) - np.abs(diag_line(p2[0], p2[1])) <= e
                and np.abs(((p2[1] - p1[1]) / (p2[0] - p1[0] + e)) * (3**0.5) - 1)
                < 1e-5
            ):
                if (
                    p2[1] > p1[1]
                    and p2[1] > (dig(p2[0]))
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
            c += 1
            
    edg1=[]
    for i, p in enumerate(input_xy):
        if np.abs(p[0]-0.5)<e and p[1]<0.5/3**0.5+e:
            edg1.append(p)  
    print(len(edg1))
            
    print(f"c = {c}, fr  = {len(fir_part)}")

    return input_xy, fir_part


pcd, fir_part = paralell_pcd(15, 15)
pcd = np.array(pcd)
plt.figure()
fir_part = np.array(list(fir_part.values()))
plt.scatter(fir_part[:, 0], fir_part[:, 1], c="r")
plt.figure()
plt.scatter(pcd[:, 0], pcd[:, 1])
# plt.scatter(pcd[:, 0], pcd[:, 1])
# plt.scatter(x, vf2(x), c="r")
# plt.scatter(x, vf1(x), c="r")
# plt.scatter([0.5] * x.shape[1], x, c="r")
# plt.scatter(x, dig(x), c="r")
plt.show()

print(np.max(pcd, axis=0))
