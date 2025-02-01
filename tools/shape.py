import math
import pickle


import numpy as np
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
import copy
from dolfin import *
import neat
import tools.utils as utils
import tools.read_mesh as rm
from scipy.spatial import Delaunay
from matplotlib.colors import ListedColormap


def triangulation(shapex, shapey):

    num_point_x = 2 * shapex + 1
    num_point_y = 2 * shapey + 1
    # num_squares_x = int((shapex - 1) / 2)
    # num_squares_y = int((shapey - 1) / 2)
    num_squares = int(shapex * shapey)
    Tri = np.zeros((num_squares * 8, 3))
    Index = np.zeros((num_point_y, num_point_x))
    n = 0
    k = 0
    for i in range(num_point_y):
        for j in range(num_point_x):
            Index[i, j] = n
            n += 1
    for ii in range(shapey):
        for jj in range(shapex):
            # i,j is the index of point which is the left top of the square
            i = ii * 2
            j = jj * 2

            # ====================画三角形
            Tri[k, :] = [Index[i, j], Index[i + 1, j], Index[i, j + 1]]
            Tri[k + 1, :] = [Index[i + 1, j], Index[i + 1, j + 1], Index[i, j + 1]]
            Tri[k + 2, :] = [Index[i, j + 1], Index[i + 1, j + 1], Index[i + 1, j + 2]]
            Tri[k + 3, :] = [Index[i, j + 2], Index[i, j + 1], Index[i + 1, j + 2]]
            Tri[k + 4, :] = [Index[i + 1, j], Index[i + 2, j], Index[i + 2, j + 1]]
            Tri[k + 5, :] = [Index[i + 1, j], Index[i + 2, j + 1], Index[i + 1, j + 1]]
            Tri[k + 6, :] = [
                Index[i + 1, j + 1],
                Index[i + 2, j + 1],
                Index[i + 1, j + 2],
            ]
            Tri[k + 7, :] = [
                Index[i + 2, j + 1],
                Index[i + 2, j + 2],
                Index[i + 1, j + 2],
            ]
            k += 8
    return Tri


def find_contour(a, thresh, pcd, shapex, shapey, pcdtype):

    X = pcd[:, 0].reshape(a.shape[0], a.shape[1]).copy()  # 先横向再纵向
    Y = pcd[:, 1].reshape(a.shape[0], a.shape[1]).copy()
    Index = np.zeros((a.shape[0], a.shape[1]))
    Cat = (a > thresh) + 1
    # create index 先横向再纵向
    n = 0  # 点
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            Index[i, j] = n
            n += 1

    new_a = np.copy(a)
    new_a[a > thresh] = 1
    new_a[a < thresh] = -1
    l = new_a[:, 0:-2]
    r = new_a[:, 2:]
    t = new_a[0:-2, :]
    b = new_a[2:, :]
    flag_x = t * b
    flag_y = l * r
    k = 0
    a = a - thresh
    min_r = 0.25
    max_r = 0.75
    is_contour = 0 if pcdtype == "parallel" else 1
    # ii,jj is the index of square
    if pcdtype == "parallel":
        return Index, X, Y, Cat
    for ii in range(shapey):
        for jj in range(shapex):

            # i,j is the index of point which is the left top of the square
            i = ii * 2
            j = jj * 2

            # 计算实际的i，j的x，y
            if flag_y[i, j] == -1:
                rho = np.abs(a[i, j]) / (np.abs(a[i, j]) + np.abs(a[i, j + 2]))
                rho = min(max_r, max(min_r, rho))
                if np.isnan(rho).any():
                    print(
                        "flag_y[i, j]",
                        np.abs(a[i, j]),
                        np.abs(a[i, j]),
                        np.abs(a[i + 2, j]),
                    )

                X[i, j + 1] = (1 - rho) * X[i, j] + rho * X[i, j + 2]
                Y[i, j + 1] = (1 - rho) * Y[i, j] + rho * Y[i, j + 2]
                Cat[i, j + 1] = 0 if is_contour == 1 else Cat[i, j + 1]

            if flag_y[i + 2, j] == -1:
                rho = np.abs(a[i + 2, j]) / (
                    np.abs(a[i + 2, j]) + np.abs(a[i + 2, j + 2])
                )
                rho = min(max_r, max(min_r, rho))

                if np.isnan(rho).any():
                    print(
                        "flag_y[i, j + 2]",
                        abs(a[i, j + 2]),
                        abs(a[i, j + 2]),
                        abs(a[i + 2, j + 2]),
                    )

                X[i + 2, j + 1] = (1 - rho) * X[i + 2, j] + rho * X[i + 2, j + 2]
                Y[i + 2, j + 1] = (1 - rho) * Y[i + 2, j] + rho * Y[i + 2, j + 2]
                Cat[i + 2, j + 1] = 0 if is_contour == 1 else Cat[i + 2, j + 1]

            if flag_x[i, j] == -1:
                rho = np.abs(a[i, j]) / (np.abs(a[i, j]) + np.abs(a[i + 2, j]))
                rho = min(max_r, max(min_r, rho))

                if np.isnan(rho).any():
                    print(" flag_x[i, j]", abs(a[i, j]), abs(a[i, j]), abs(a[i, j + 2]))

                X[i + 1, j] = (1 - rho) * X[i, j] + rho * X[i + 2, j]
                Y[i + 1, j] = (1 - rho) * Y[i, j] + rho * Y[i + 2, j]
                Cat[i + 1, j] = 0 if is_contour == 1 else Cat[i + 1, j]

            if flag_x[i, j + 2] == -1:
                rho = np.abs(a[i, j + 2]) / (
                    np.abs(a[i, j + 2]) + np.abs(a[i + 2, j + 2])
                )
                rho = min(max_r, max(min_r, rho))

                if np.isnan(rho).any():
                    print(
                        "flag_x[i + 2, j]",
                        abs(a[i + 2, j]),
                        abs(a[i + 2, j]),
                        abs(a[i + 2, j + 2]),
                    )
                X[i + 1, j + 2] = (1 - rho) * X[i, j + 2] + rho * X[i + 2, j + 2]
                Y[i + 1, j + 2] = (1 - rho) * Y[i, j + 2] + rho * Y[i + 2, j + 2]
                Cat[i + 1, j + 2] = 0 if is_contour == 1 else Cat[i + 1, j + 2]

            if Cat[i, j + 1] + Cat[i + 2, j + 1] == 0:  # 上下是边界
                X[i + 1, j + 1] = (X[i, j + 1] + X[i + 2, j + 1]) * 0.5
                Y[i + 1, j + 1] = (Y[i, j + 1] + Y[i + 2, j + 1]) * 0.5
                Cat[i + 1, j + 1] = 0 if is_contour == 1 else Cat[i + 1, j + 1]

            if Cat[i + 1, j] + Cat[i + 1, j + 2] == 0:
                X[i + 1, j + 1] = (X[i + 1, j] + X[i + 1, j + 2]) * 0.5
                Y[i + 1, j + 1] = (Y[i + 1, j] + Y[i + 1, j + 2]) * 0.5
                Cat[i + 1, j + 1] = 0 if is_contour == 1 else Cat[i + 1, j + 1]

    if np.isnan(X).any():
        print("x nan")
    if np.isnan(Y).any():
        print("y nan")
    return Index, X, Y, Cat


def get_parallel_filtered_mesh(points):
    # 计算Delaunay三角剖分
    tri = Delaunay(points)

    # 计算每个三角形的面积
    def triangle_area(coords):
        area = 0.5 * np.abs(
            np.dot(coords[:, 0], np.roll(coords[:, 1], 1))
            - np.dot(np.roll(coords[:, 0], 1), coords[:, 1])
        )
        return area

    # 计算最长边和最短边的比值
    def longest_to_shortest_edge_ratio(coords):
        a = np.linalg.norm(coords[1] - coords[0])
        b = np.linalg.norm(coords[2] - coords[1])
        c = np.linalg.norm(coords[2] - coords[0])
        longest_edge = max(a, b, c)
        shortest_edge = min(a, b, c)
        return longest_edge / shortest_edge

    # 设定最大面积和最长边与最短边比值的阈值
    max_area = 0.0006
    max_edge_ratio = 5

    # 计算每个三角形的面积和最长边与最短边的比值
    areas = np.array([triangle_area(points[simplex]) for simplex in tri.simplices])
    # print(areas)
    edge_ratios = np.array(
        [longest_to_shortest_edge_ratio(points[simplex]) for simplex in tri.simplices]
    )
    # 过滤掉面积大于最大面积或最长边与最短边比值大于最大阈值的三角形
    filtered_simplices = tri.simplices[
        (areas < max_area) & (edge_ratios <= max_edge_ratio)
    ]
    return filtered_simplices


def get_tri_cat(Tri, index_x_y_cat):
    cat = []
    for i, tri in enumerate(Tri):
        flag = 0
        for node in tri:
            # Todo 下标回去
            if index_x_y_cat[int(node), -1] == 1:
                flag = 1
                break
                # outside_tri.append(tri)
                # break
        if flag == 1:
            cat.append(1)
        else:
            cat.append(2)
    return cat


def get_outside_Tri(Tri, index_x_y_cat):
    outside_tri = []
    for i, tri in enumerate(Tri):
        flag = 0
        for node in tri:
            # Todo 下标回去
            if index_x_y_cat[int(node), -1] == 1:
                # flag=1
                # break
                outside_tri.append(tri)
                break
        # if flag==0:
        # outside_tri.append(tri)
    return outside_tri


def split_tri(Tri, index_x_y_cat):

    tri_cat = []
    for i, tri in enumerate(Tri):
        flag = 2
        for node in tri:
            # Todo 下标回去
            if index_x_y_cat[int(node), -1] == 1:
                flag = 1
                tri_cat.append(flag)
                break
        if flag == 2:
            tri_cat.append(flag)
    return tri_cat


def load_net(path, config):
    with open(f"{path}", "rb") as f:
        genome = pickle.load(f)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
    return genome, net


def getshape(
    config,
    path,
    thresh,
    pcd,
    Tri,
    shapex,
    shapey,
    fixed_list,
    is_mesh=False,
    go_path=None,
    sympcd=None,
):
    genome, net = load_net(path, config)
    outputs = []
    if (
        config.pcdtype == "sym4"
        or config.pcdtype == "rotate"
        or config.pcdtype == "parallel"
    ):
        for point in sympcd:
            output = net.activate(point)
            outputs.append(output)
    else:
        for point in pcd:
            if config.pcdtype == "sym2":
                point = np.cos(point * math.pi)
            output = net.activate(point)
            outputs.append(output)
    outputs = np.array(outputs)
    outputs = utils.scale(outputs)

    if config.pcdtype == "parallel":
        for idx in fixed_list:
            idx = int(idx)
            if outputs[idx] <= 0.5:
                outputs[idx] = 0
            else:
                outputs[idx] = 1
    outputs = outputs.reshape(2 * shapey + 1, 2 * shapex + 1)
    Index, X, Y, Cat = find_contour(
        outputs, thresh, pcd, shapex, shapey, config.pcdtype
    )
    x_values = X.flatten()
    y_values = Y.flatten()
    cat_values = Cat.flatten()
    index_values = Index.flatten()

    # 假设tri是包含三角形索引的数组
    # 假设a是包含索引、x坐标、y坐标和类别的数组
    index_x_y_cat = np.concatenate(
        (
            index_values.reshape(-1, 1),
            x_values.reshape(-1, 1),
            y_values.reshape(-1, 1),
            cat_values.reshape(-1, 1),
        ),
        axis=1,
    )

    triangles = Tri
    indices = index_x_y_cat[:, 0]
    x_coords = index_x_y_cat[:, 1]
    y_coords = index_x_y_cat[:, 2]

    inside_points = []
    contour_points = []
    for i, c in enumerate(cat_values):
        if c == 1:
            inside_points.append([x_coords[i], y_coords[i]])
        if c == 0:
            contour_points.append([x_coords[i], y_coords[i]])
    inside_points = np.array(inside_points)
    contour_points = np.array(contour_points)

    cat = get_tri_cat(Tri, index_x_y_cat)
    outtri = get_outside_Tri(Tri, index_x_y_cat)
    if go_path == None:
        go_path = path.split(".")[1]
        print(go_path)
    draw_shape(
        genome,
        x_coords,
        y_coords,
        triangles,
        cat,
        index_x_y_cat,
        outtri,
        path,
        go_path,
        config,
        inside_points,
        contour_points,
        is_mesh=is_mesh,
    )


def draw_shape(
    genome,
    x_coords,
    y_coords,
    triangles,
    cat,
    index_x_y_cat,
    outtri,
    g_path,
    to_path,
    config,
    parallel_inside_pcd,
    contour_points,
    is_mesh=False,
):
    f1 = "{:.4f}".format(genome.fitnesses[0])
    f2 = "{:.4f}".format(genome.fitnesses[1])
    plt.figure(figsize=(6, 6))
    if config.pcdtype == "parallel":
        filtered_mesh = get_parallel_filtered_mesh(parallel_inside_pcd)
        mesh = rm.get_parallel_mesh(
            parallel_inside_pcd, filtered_mesh, config.test_mode
        )
    else:
        mesh = rm.getmesh(index_x_y_cat, outtri, config.pcdtype, config.test_mode)
    # if config.pcdtype == "parallel":
    #     plt.xlim(0, 1.5)
    #     plt.ylim(0, 1)
    # else:
    #     plt.xlim(0, 1)
    #     plt.ylim(0, 1)
    plot(mesh)
    plt.savefig(f".{to_path}_{f1}_{f2}_mesh.png", transparent=True, dpi=600)
    plt.gca().set_axis_off()
    plt.close()
    if is_mesh:
        plt.figure(figsize=(6, 6))
        if config.pcdtype == "parallel":
            filtered_mesh = get_parallel_filtered_mesh(parallel_inside_pcd)
            mesh = rm.get_parallel_mesh(
                parallel_inside_pcd, filtered_mesh, config.test_mode
            )
        else:
            mesh = rm.getmesh(index_x_y_cat, outtri, config.pcdtype, config.test_mode)
        # if config.pcdtype == "parallel":
        #     plt.xlim(0, 1.5)
        #     plt.ylim(0, 1)
        # else:
        #     plt.xlim(0, 1)
        #     plt.ylim(0, 1)
        plt.gca().set_axis_off()
        plot(mesh)
        plt.savefig(f".{to_path}_{f1}_{f2}_mesh.png", transparent=True, dpi=600)
        
        plt.close()
    else:
        
        if config.pcdtype == "parallel":
            plt.figure()
            cmap = ListedColormap(["black", "white"])
            filtered_mesh = get_parallel_filtered_mesh(parallel_inside_pcd)
            print(filtered_mesh.shape)
            plt.triplot(
                parallel_inside_pcd[:, 0],
                parallel_inside_pcd[:, 1],
                filtered_mesh,
                "-",
                lw=0.0,
            )
            
            plt.tripcolor(
                parallel_inside_pcd[:, 0],
                parallel_inside_pcd[:, 1],
                filtered_mesh,
                facecolors=[0]*filtered_mesh.shape[0],
                edgecolors="none",
                cmap=cmap,
            )
            plt.gca().set_axis_off()
            plt.gca().set_aspect('equal')
            plt.show()
            plt.savefig(f".{to_path}_{f1}_{f2}.png",bbox_inches='tight', pad_inches=0,transparent=True, dpi=600)
            plt.close()
        else:
            plt.figure(figsize=(6, 6))
            cmap = ListedColormap(['black', "white"])
            plt.triplot(x_coords, y_coords, triangles, "-", lw=0.0)
            plt.tripcolor(
                x_coords,
                y_coords,
                triangles,
                facecolors=cat,
                edgecolors="none",
                cmap=cmap,
            )

            plt.gca().set_axis_off()
            plt.show()
            plt.savefig(f".{to_path}_{f1}_{f2}.png", transparent=True, dpi=600)
            plt.close()

            plt.figure(figsize=(6, 6))
            plt.scatter(
                parallel_inside_pcd[:, 0],
                parallel_inside_pcd[:, 1],
                c=[132 / 255, 124 / 255, 107 / 255],
                edgecolors="black",
                linewidths=1,
                s=20,
            )
            if len(contour_points) != 0:
                plt.scatter(
                    contour_points[:, 0],
                    contour_points[:, 1],
                    c=[255 / 255, 28 / 255, 49 / 255],
                    edgecolors="black",
                    linewidths=1,
                    s=20,
                )
                plt.gca().set_axis_off()
                plt.show()
                plt.savefig(f".{to_path}_{f1}_{f2}_pcd.png", transparent=True, dpi=600)
                plt.close()
