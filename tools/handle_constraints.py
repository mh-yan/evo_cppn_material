import numpy as np


class handle_constarints(object):

    def __init__(self):
        self.violate_num = 0

    def cal_violate_num(self, matrix, filtered_simplices, gen):
        vio_num = 0
        if not self.c_is_tri_connect(filtered_simplices):
            vio_num += 1
        if not self.c_is_connect_all(matrix, gen):
            vio_num += 1

        self.violate_num = vio_num

    def c_is_tri_connect(self, filtered_simplices):
        def build_adjacency_matrix(simplices):
            num_simplices = simplices.shape[0]
            adjacency_matrix = np.zeros((num_simplices, num_simplices), dtype=bool)
            # 创建一个边到三角形的字典
            edge_to_tri = {}
            for i, simplex in enumerate(simplices):
                for j in range(3):
                    # 每个dict item 最多两个
                    edge = tuple(sorted((simplex[j], simplex[(j + 1) % 3])))
                    if edge in edge_to_tri:
                        for neighbor in edge_to_tri[edge]:
                            adjacency_matrix[i, neighbor] = True
                            adjacency_matrix[neighbor, i] = True
                        edge_to_tri[edge].append(i)
                    else:
                        edge_to_tri[edge] = [i]
            # print(edge_to_tri)
            return adjacency_matrix

        def is_connected2(adjacency_matrix):
            num_simplices = adjacency_matrix.shape[0]
            if num_simplices == 0:
                return False
            visited = np.zeros(num_simplices, dtype=bool)
            stack = [0]  # 使用栈模拟递归过程
            visited[0] = True
            while stack:
                current = stack.pop()
                for neighbor in range(num_simplices):
                    if adjacency_matrix[current, neighbor] and not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(neighbor)
            return visited.all()

        adjacency_matrix = build_adjacency_matrix(filtered_simplices)
        connected = is_connected2(adjacency_matrix)
        return connected

    def c_is_connect_all(self, matrix2, gen):
        flag = 0
        flag1 = 0
        flag2 = 0
        min_num = 1
        # print(matrix2)
        if gen >= 1:
            min_num = 3
        rows, columns = matrix2.shape
        # 判断上下边
        for j in range(columns):
            if (
                matrix2[0, j] == 1
                and matrix2[rows - 1, j] == 1
                or matrix2[0, j] + matrix2[rows - 1, j] == 1
            ):
                flag1 += 1

        if flag1 >= min_num:
            flag += 1
        # 判断左右边
        for i in range(rows):
            if (
                matrix2[i, 0] == 1
                and matrix2[i, columns - 1] == 1
                or matrix2[i, 0] + matrix2[i, columns - 1] == 1
            ):
                flag2 += 1
        if flag2 >= min_num:
            flag += 1

        if flag == 2:
            return True

        return False
