import sys
import pickle
from matplotlib import pyplot as plt
import neat
import tools.utils as utils
import tools.read_mesh as rm
import numpy as np
import tools.shape as shape
from main import (
    point_xy,
    sym4_pcd,
    sym_rotate,
    paralell_pcd,
    paralell_pcd_rotate,
)
import numpy as np
from dolfin import *
import glob
import os
import copy
import numpy as np
import tools.period as fit_period
from tools.shape import *
import tools.utils as utils
from tools.read_mesh import getmesh, get_parallel_mesh
import cv2

config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    "config.ini",
)


orig_size_xy = (1, 1)
# 换密度plot
density = config.density
threshold = 0.5
shapex = orig_size_xy[0] * density
shapey = orig_size_xy[1] * density
pointcloud = None
pcdtype = config.pcdtype
if pcdtype == "parallel":
    pointcloud = paralell_pcd(shapex, shapey)  # point_xy
else:
    pointcloud = point_xy(shapex, shapey, orig_size_xy)

square_pcd = None
fixed_list = None
if pcdtype == "sym4":
    square_pcd = sym4_pcd(shapex, shapey)
elif pcdtype == "rotate":
    square_pcd = sym_rotate(shapex, shapey)
elif pcdtype == "parallel":
    square_pcd = paralell_pcd_rotate(shapex, shapey)

copy_pcd = copy.deepcopy(pointcloud)
Tri = shape.triangulation(shapex, shapey)
fixed_list = []

switch_mode = 1
index_dir = 1057
if switch_mode == 1:
    # the root directory of neural network
    path2 = f"./output/nosym_shear_normalleft_1/output{index_dir}/*.pkl"

    pkl_files = glob.glob(path2, recursive=True)
    for file_path in pkl_files:
        # 从文件名中提取output{k}部分作为新文件名
        base_name = os.path.basename(file_path)
        file_path = f"{file_path}"
        # 读取.pkl文件内容
        utils.plotall(
            config=config,
            g_path=file_path,
            thresh=threshold,
            pcd=pointcloud,
            Tri=Tri,
            shapex=shapex,
            shapey=shapey,
            fixed_list=fixed_list,
            is_mesh=False,
            go_path=None,
            sympcd=square_pcd,
        )


# TODO get single chom
else:
    index_file = 26
    genome_path = f"./output/output{index_dir}/genome{index_file}.pkl"
    with open(f"{genome_path}", "rb") as f:
        genome = pickle.load(f)
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    outputs = []
    if (
        config.pcdtype == "sym4"
        or config.pcdtype == "rotate"
        or config.pcdtype == "parallel"
    ):
        for point in square_pcd:
            output = net.activate(point)
            outputs.append(output)
    else:
        for point in pointcloud:
            if config.pcdtype == "sym2":
                point = np.cos(point * math.pi)
            output = net.activate(point)
            outputs.append(output)
    outputs = np.array(outputs)
    outputs = utils.scale(outputs)

    outputs_square = outputs.reshape(2 * shapex + 1, -1)
    # outputs_square[:, 0] = 0d
    # outputs_square[:, -1] = 0
    # outputs_square[0, :] = 0
    # outputs_square[-1, :] = 0
    # outputs_square[:, 1] = 0
    # outputs_square[:, -2] = 0
    # outputs_square[1, :] = 0
    # outputs_square[-2, :] = 0
    if pcdtype == "parallel" or pcdtype == "nosym":
        for idx in fixed_list:
            idx = int(idx)
            if outputs[idx] <= 0.5:
                outputs[idx] = 0
            else:
                outputs[idx] = 1
    Index, X, Y, Cat = find_contour(
        a=outputs_square, thresh=threshold, pcd=pointcloud, shapex=shapex, shapey=shapey
    )
    x_values = X.flatten()
    y_values = Y.flatten()
    cat_values = Cat.flatten()
    index_values = Index.flatten()
    index_x_y_cat = np.concatenate(
        (
            index_values.reshape(-1, 1),
            x_values.reshape(-1, 1),
            y_values.reshape(-1, 1),
            cat_values.reshape(-1, 1),
        ),
        axis=1,
    )
    inside_points = []
    for i, c in enumerate(cat_values):
        if c == 1:
            inside_points.append(pointcloud[i])
    inside_points = np.array(inside_points)
    # index_x_y_cat[:, 1] = np.round(index_x_y_cat[:, 1], 6)
    # index_x_y_cat[:, 2] = np.round(index_x_y_cat[:, 2], 6)
    # 1,2  1是outside ，2是inside
    outtri = get_outside_Tri(Tri, index_x_y_cat)
    plt.figure()
    plt.gca().set_aspect(1)
    plt.scatter(inside_points[:, 0], inside_points[:, 1])
    plt.savefig("./inside.png")
    plt.close()
    # print(Cat)
    if pcdtype == "parallel":
        filtered_tri = get_parallel_filtered_mesh(inside_points)
        mesh = get_parallel_mesh(inside_points, filtered_tri, config.test_mode)
    else:
        mesh = getmesh(index_x_y_cat, outtri, config.pcdtype, config.test_mode)
    one = Constant(1.0)
    plot(mesh)
    plt.savefig("allmesh.png")
    print(Cat)
    # print(is_c)
    # 创建表达式用于积分，这里用的是常数函数
    area = assemble(one * dx(domain=mesh))
    print("The computed area of the mesh is:", area)
    f1, f2 = fit_period.getfit(mesh, config.pcdtype)
    print(f1, f2)
