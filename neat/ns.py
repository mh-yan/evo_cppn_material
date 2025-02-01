from scipy.spatial.distance import cdist
import numpy as np
import copy
from skimage.metrics import structural_similarity as ssim
from multiprocessing import Pool
import multiprocessing


class archive(object):
    def __init__(self, config) -> None:
        self.data = []
        self.archive_len = config.archive_len
        self.min_novelty = 0

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return self.data.__iter__()

    def __next__(self):
        return self.data.__next__()

    def __getitem__(self, item):
        return self.data[item]

    def update(self, data):

        return

    def store(self, data):
        self.data.append(data)


class novelty_search(object):

    def __init__(self, config) -> None:
        self.threshold = config.novelty_threshold
        self.knn = config.knn
        self.archive = archive(config)
        self.timeout = 0
        self.degrade_rate = config.degrade_rate
        self.upgrade_rate = config.upgrade_rate
        self.isfirst = 1

    def eval_novelty(self, pop_off):
        print("start ns ")
        arch = self.archive.data
        if len(arch) == 0:
            arch = []
        list_pop_off = list(pop_off.values())
        pop_off_novelty = list_pop_off + arch
        # too slow
        dismatrix = self.cal_knn_dis_sim(list_pop_off, pop_off_novelty)

        novelties = [self.cal_novelty(dis, self.knn) for dis in dismatrix]
        if self.isfirst == 1:
            self.isfirst += 1
            self.threshold = np.mean(novelties)
        num_add_gen = 0

        # distribute the novelty
        for i, g in enumerate(list_pop_off):
            g.novelty = novelties[i]
            g.fitnesses.append(g.novelty)
            # update the archive
            if (
                g.novelty < self.threshold
                or len(self.archive) < self.archive.archive_len
            ):
                self.update_archive(g)
                num_add_gen += 1
        self.adjust_threshold(num_add_gen)
        self.update_age()
        print(f"threshold :{self.threshold} and num_add :{num_add_gen}")

    def cal_knn_dis(self, pop_off, pop_off_novelty):
        """
        p1=[[f1,f2]...] from pop_off
        p2=[[f1,f2]...] from pop_off_novelty
        return dismatirx
        """
        # print(pop_off)
        p1 = [[g.fitness[0], g.fitness[1]] for g in pop_off]
        p2 = [[g.fitness[0], g.fitness[1]] for g in pop_off_novelty]
        dismatrix = cdist(p1, p2, "euclidean")
        return dismatrix

    def cal_knn_dis_sim(self, pop_off, pop_off_novelty):
        # 将每个输出转换为二进制矩阵
        # 假设outputs是形状(d, k)的二维数组
        out_shape = [g.outputs.shape for g in pop_off]
        m1 = np.array([np.where(g.outputs > 0.5, 1, 0) for g in pop_off])
        m2 = np.array([np.where(g.outputs > 0.5, 1, 0) for g in pop_off_novelty])
        # 如果outputs是二维的，m1和m2的形状将是(m, d, k)和(n, d, k)
        # 为了计算相似度，我们需要在d维度上也进行比较
        # 增加一个新维度到m2，变为(1, n, d, k)
        m2_expanded = m2[np.newaxis, :, :, :]
        # 计算XOR，逻辑非，然后在d和k维度上求和
        similarity_matrix = np.logical_not(
            np.bitwise_xor(m1[:, np.newaxis], m2_expanded)
        ).sum(axis=(2, 3))
        return similarity_matrix

    def compute_ssim(self, args):
        sub_m1, sub_m2 = args
        return -(ssim(sub_m1, sub_m2, data_range=1) + 1)

    def cal_knn_dis_ssim(self, pop_off, pop_off_novelty):
        """
        ssim range from [0,2]
        """
        m1 = [np.where(g.outputs > 0.5, 1, 0) for g in pop_off]
        m2 = [np.where(g.outputs > 0.5, 1, 0) for g in pop_off_novelty]
        dismatrix = []
        with Pool(processes=multiprocessing.cpu_count()) as pool:
            # Execute the SSIM computation in parallel
            dismatrix = pool.map(
                self.compute_ssim, [(sub_m1, sub_m2) for sub_m1 in m1 for sub_m2 in m2]
            )
            pool.close()
            pool.join()
            pool.terminate()
        dismatrix = np.array(dismatrix)
        return dismatrix.reshape(len(m1), -1)

    def cal_novelty(self, dis, knn):
        novelty = 0
        idx = np.argsort(dis)
        # +1 exclude the element itself
        novelty = np.mean(dis[idx[1 : knn + 1]])
        return novelty

    def update_archive(self, genome):
        self.archive.data.sort(key=lambda g: (g.age, g.novelty))
        if len(self.archive) >= self.archive.archive_len:
            # remove the lowest one
            self.archive.data.pop()
            self.archive.data.append(copy.deepcopy(genome))
        else:
            self.archive.data.append(copy.deepcopy(genome))

    # TODO 自适应threshold
    def adjust_threshold(self, num_add):
        if num_add <= 3:
            self.timeout += 1
            if self.timeout >= 5:
                self.threshold *= self.upgrade_rate
                self.timeout = 0
        if num_add >= 10:
            # TODO 负数的threshold 逻辑有问题
            self.threshold *= self.degrade_rate

    def update_age(self):
        for g in self.archive.data:
            g.age += 1
