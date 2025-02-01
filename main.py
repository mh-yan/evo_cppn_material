import multiprocessing
import os
import pickle
import random
import math
import neat
import numpy as np
import tools.period as fit_period
import tools.fitness_function
from tools.shape import *
import tools.utils as utils
from tools.read_mesh import getmesh, get_parallel_mesh
import tools.handle_constraints as hc
from gen_pcd import *
from multi_task import *


# from homogenization_2d import *
from tools.HomProp2D import *


def get_load_support(pcd):
    load = []
    support = []
    for i, point in enumerate(pcd):
        if point[0] == 3:
            if point[1] <= -0.8:
                load.append(i)

    for i, point in enumerate(pcd):
        if point[0] == -3:
            if point[1] >= 0.8 or point[1] <= -0.8:
                support.append(i)
    return load, support


def has_tri(cat_tri):
    flag1 = 0
    flag2 = 0
    flag3 = 0
    flag4 = 0
    flag = 0
    new_tri = np.copy(Tri)
    pcd = np.copy(pointcloud)
    for i_tri, tri in enumerate(new_tri):
        # print(tri)
        for index_p in tri:
            # print(index_p)
            x = pcd[int(index_p)][0]
            y = pcd[int(index_p)][1]

            if y == 1:
                if cat_tri[i_tri] == 1:
                    flag1 += 1
            if x == 1:
                if cat_tri[i_tri] == 1:
                    flag2 += 1
            if y == -1:
                if cat_tri[i_tri] == 1:
                    flag3 += 1
            if x == -1:
                if cat_tri[i_tri] == 1:
                    flag4 += 1
            if flag1 >= 1 and flag2 >= 1 and flag3 >= 1 and flag4 >= 1:
                flag = 1
                break
        if flag == 1:
            break
    if flag == 1:
        return True
    else:
        return False


def eval_genome(genome, config, gen):
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
    outputs_square = outputs.reshape(2 * config.density + 1, -1)
    Index, X, Y, Cat = find_contour(
        a=outputs_square,
        thresh=0.5,
        pcd=pointcloud,
        shapex=config.density,
        shapey=config.density,
        pcdtype=config.pcdtype,
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
    # 1,2  1是outside ，2是inside
    outtri = get_outside_Tri(Tri, index_x_y_cat)
    # TODO 连通
    outputs_sq = np.copy(Cat)
    handle_contraints = hc.handle_constarints()
    violate_num = 0
    random_number1 = np.random.uniform(2, 3)
    random_number2 = np.random.uniform(2, 3)
    if config.pcdtype == "parallel":
        inside_points = []
        for i, c in enumerate(cat_values):
            if c == 1 or c == 0:
                if c==0:
                    inside_points.append(pointcloud[i])
        inside_points = np.array(inside_points)
        filtered_tri = get_parallel_filtered_mesh(inside_points)
        handle_contraints.cal_violate_num(outputs_sq, filtered_tri, gen)
        violate_num = handle_contraints.violate_num
        mesh = get_parallel_mesh(inside_points, filtered_tri, config.test_mode)
    else:
        outtri = get_outside_Tri(Tri, index_x_y_cat)
        if len(outtri) == 0:
            return (
                [random_number1, random_number2],
                outputs_square,
                0,
                4,
            )
        mesh = getmesh(index_x_y_cat, outtri, config.pcdtype, config.test_mode)
        filtered_tri = mesh.cells()
        handle_contraints.cal_violate_num(outputs_sq, filtered_tri, gen)
        violate_num = handle_contraints.violate_num
    f1 = None
    f2 = None
    solved = False
    if violate_num > 0:
        f1 = random_number1
        f2 = random_number2
    else:
        f1, f2, solved = fit_period.getfit(mesh, config.pcdtype, Tradeoff)
    if not solved:
        violate_num += 1
    is_normal = 0 if violate_num > 0 else 1
    return [f1, f2], outputs_square, is_normal, violate_num


def run_experiment(
    config, task_name, out_path, cur_times, cur_tradeoff, n_generations=100
):
    # 生成对应的点云
    pcdtype = config.pcdtype
    n_generations = config.gen
    density = config.density
    orig_size_xy = (1, 1)
    shapex = orig_size_xy[0] * density
    shapey = orig_size_xy[1] * density
    global square_pcd, pointcloud, Tri, Tradeoff
    print("start generate pcd")
    if pcdtype == "parallel":
        pointcloud = paralell_pcd(shapex, shapey)  # point_xy
    else:
        pointcloud = point_xy(shapex, shapey, orig_size_xy)
    if pcdtype == "sym4":
        square_pcd = sym4_pcd(shapex, shapey)
    elif pcdtype == "rotate":
        square_pcd = sym_rotate(shapex, shapey)
    elif pcdtype == "parallel":
        square_pcd = paralell_pcd_rotate(shapex, shapey)
    Tri = triangulation(shapex, shapey)
    Tradeoff = cur_tradeoff
    print("end generate pcd")

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    p.run(
        fitness_function=pe.evaluate,
        n=n_generations,
        task_name=task_name,
        out_path=out_path,
        cur_times=cur_times,
    )
    print("end!")


# Set the seed

square_pcd = None
pointcloud = None
Tri = None
Tradeoff = None

if __name__ == "__main__":
    utils.clear_folder("./mesh")
    utils.clear_folder("./contour")
    utils.clear_folder("./pcd")
    utils.clear_folder("./output")
    is_collect = 0
    if is_collect == 1:
        ea = exc_all(tasks_parameters, run_experiment)
        ea.run_all()
    else:
        random_seed = 333
        random.seed(random_seed)
        np.random.seed(random_seed)
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path,
        )
        cur_tradeoff = "shear_normalleft"
        run_experiment(
            config=config,
            task_name=config.pcdtype + "_" + cur_tradeoff,
            out_path="./output/",
            cur_times=1,
            cur_tradeoff=cur_tradeoff,
            n_generations=100,
        )
    # run_experiment(config, task_name,out_path, cur_times,n_generations=n_generations)
