import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def circle_test_score(inside_points):
    penalty = 0
    if inside_points.shape[0] == 0:
        penalty = -1000
    else:
        for point in inside_points:
            penalty -= 16 * (point[0]) ** 2 + 4 * (point[1]) ** 2 - 1
    return penalty


def element_stiffness(coords):
    # Define the material properties
    E = 1e9  # Young's modulus in Pascals
    nu = 0.3  # Poisson's ratio
    thickness = 0.01  # Thickness in meters

    C = (E / (1 - nu**2)) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])

    # Area of the triangle
    a = coords[0]
    b = coords[1]
    c = coords[2]
    area = 0.5 * np.linalg.det(
        np.array([[1, a[0], a[1]], [1, b[0], b[1]], [1, c[0], c[1]]])
    )

    # B matrix (strain-displacement matrix)
    B = np.zeros((3, 6))
    B[0, [0, 2, 4]] = [b[1] - c[1], c[1] - a[1], a[1] - b[1]]
    B[1, [1, 3, 5]] = [c[0] - b[0], a[0] - c[0], b[0] - a[0]]
    B[2, [0, 1, 2, 3, 4, 5]] = [
        c[0] - b[0],
        b[1] - c[1],
        a[0] - c[0],
        c[1] - a[1],
        b[0] - a[0],
        a[1] - b[1],
    ]
    B /= 2 * area

    # Element stiffness matrix
    K_el = thickness * area * (B.T @ C @ B)
    return K_el


def fea_analysis(nodes, elements, loads, supps, tags):
    # Assembly process
    rows = []
    cols = []
    data = []

    for i, el in enumerate(elements):
        el = [int(i) for i in el]
        el_nodes = nodes[el]
        # Todo 1是内部还是外部
        K_el = (1.0 if tags[i] == 1 else 1e-4) * element_stiffness(el_nodes)
        for i in range(3):
            for j in range(3):
                for k in range(2):
                    for l in range(2):
                        row_index = 2 * el[i] + k
                        col_index = 2 * el[j] + l
                        rows.append(row_index)
                        cols.append(col_index)
                        data.append(K_el[2 * i + k, 2 * j + l])

    # Create the global stiffness matrix in COO format
    K_global = coo_matrix((data, (rows, cols)), shape=(2 * len(nodes), 2 * len(nodes)))

    # Apply boundary conditions
    fixed_dofs = []
    for node, supp in supps.items():
        for i in range(2):
            if supp[i] != 0:
                fixed_dofs.append(node * 2 + i)

    dofs = np.arange(2 * len(nodes))
    free_dofs = np.setdiff1d(dofs, np.array(fixed_dofs))

    # Convert to CSR format for solving
    K_global_csc = K_global.tocsc()
    K_global_csc = K_global_csc[free_dofs, :][:, free_dofs]

    # Apply loads
    F = np.zeros(2 * len(nodes))
    for node, load in loads.items():
        F[2 * node : 2 * node + 2] = load

    # Solve for the displacements
    U_free = spsolve(K_global_csc, F[free_dofs])
    U = np.zeros(2 * len(nodes))
    U[free_dofs] = U_free
    compliance = F.dot(U)
    return -compliance
