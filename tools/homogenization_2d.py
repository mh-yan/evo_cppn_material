# Copyright (C) 2022 Yingqi Jia
# Author(s): Yingqi Jia
# E-mail: yingqij2@illinois.edu
# Date: 12/20/2022

"""
This code is used to compute the homogenized constitutive matrix of a given
primitive cell. For solid elements, the primitive cell is the same as the unit
cell.

Reference:
[1] Vigliotti, A. & Pasini, D. Stiffness and strength of tridimensional
periodic lattices. Computer Methods in Applied Mechanics and Engineering
229–232, 27–43 (2012).
"""

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from scipy.spatial import cKDTree
import dolfinx.io
from mpi4py import MPI


def compare_matrices(array1, array2, precision=12, k=1):
    """
    Find the list "args" such that array1[args] == array2.
    Input:
        array1, array2: Two matrices containing same vectors but with
                        different orders.
        precision: Precision of numbers in array1 and array2. In the parallel
                   computation, small numerical errors might be added to array1
                   and array2, and thus it is necessary to eliminate those
                   errors with the operation of round(precision). This value
                   should be small enougth to eliminate the numerical error.
                   However, information of arrays migth be lost if it is
                   extremely small. For example, array1 = [6.01, 6.02] is the
                   serial data and array2 = [6.010001, 6.020001] is the parlell
                   data. array2.round(8) = [6.010001, 6.020001] can still never
                   matches array1.round(8) = [6.01, 6.02]. array2.round(1)
                   = [6.0, 6.0] and array2 = [6.0, 6.0] will lose the matching
                   information.
        k: Number of returned indices.
    Output:
        args: Relationships of arguments.
    """
    kd_tree = cKDTree(array1.round(precision))
    index = kd_tree.query(array2.round(precision), k=k)[1]
    return index


def find_indices(parent_array, array):
    """
    Find the indices of elements of array in parent_array.
    Requirement: array is the proper subset of parent_array.
    Example:
        a = np.array([1, 3, 7, 5, 4])
        b = np.array([3, 4, 5])
        find_indices(a, b)
        return: [1 4 3]
    """
    sorter = np.argsort(parent_array)
    return sorter[np.searchsorted(parent_array, array, sorter=sorter)]


def transform_mesh(mesh, original_nodes=None, gamma=1, shear=0, rotation=0):
    """Transform a given mesh.

    Args:
        mesh: FEniCSx mesh.
        original_nodes: Original nodes of the FEniCSx mesh.
        gamma: ratio of the vertical and horizontal scaling factors.
        shear: counterclockwise shear angle (in rad).
        rotation: clockwise rotation angle (in rad).

    Returns:
        mesh: FEniCSx mesh with transformed nodes.
        A_mat: Periodicity vectors (assuming the mesh is a parallelogram).
    """
    lambda1 = 1
    lambda2 = gamma * lambda1
    theta1, theta2 = rotation, rotation + np.pi / 2 - shear

    if original_nodes is None:
        original_nodes = mesh.geometry.x.copy()

    nodes = original_nodes.copy()
    nodes[:, 0] *= lambda1
    nodes[:, 1] *= lambda2
    nodes_x = nodes[:, 0] * np.cos(theta1) + nodes[:, 1] * np.cos(theta2)
    nodes_y = nodes[:, 0] * np.sin(theta1) + nodes[:, 1] * np.sin(theta2)
    mesh.geometry.x[:, :2] = np.vstack((nodes_x, nodes_y)).T

    lx, ly = np.ptp(original_nodes, axis=0)[:2]
    A_mat = np.array(
        [
            [lambda1 * lx * np.cos(theta1), lambda1 * lx * np.sin(theta1)],
            [lambda2 * ly * np.cos(theta2), lambda2 * ly * np.sin(theta2)],
        ]
    )
    return mesh, A_mat


def global_stiffness_matrix_fenicsx(mesh, mat_table):
    """Global stiffness matrix derived with FEniCSx."""
    V = VectorFunctionSpace(mesh, ("CG", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    E, nu = mat_table["E0"], mat_table["nu0"]
    lambda_ = E * nu / (1 + nu) / (1 - 2 * nu)  # First Lame constant
    mu = E / (2 * (1 + nu))  # Second Lame constant (shear modulus)
    if mat_table["psflag"] not in ["plane_stress", "plane_strain"]:
        raise ValueError("Unsupported 'psflag'.")
    if mat_table["psflag"] == "plane_stress":
        lambda_ = 2 * lambda_ * mu / (lambda_ + 2 * mu)
    dx = ufl.Measure("dx", metadata={"quadrature_degree": 2})

    def epsilon(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u):
        return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)

    def stiffness(u, v):
        return ufl.inner(sigma(u), epsilon(v)) * dx

    def get_matrix(expr):
        mat_form = form(expr)
        mat = create_matrix(mat_form)

        mat.zeroEntries()
        assemble_matrix(mat, mat_form)
        mat.assemble()

        ai, aj, av = mat.getValuesCSR()
        return sparse.csr_matrix((av, aj, ai))

    return get_matrix(stiffness(u, v))


def topology_matrices(nodes, A_mat=None, eps=1e-4):
    """Find the topology matrices B_0, B_a, and B_eps of the given unit cell.

    Args:
        nodes: A (2, N) array representing the nodal coordinates of the unit cell.
        A_mat: A (2, 2) array of periodicity vectors such that A_mat = [[a1], [a2]].
            If not provided, A_mat will be automatically deduced by assuming
            the unit cell is a rectangle.
    """
    if A_mat is None:
        # Assume the unit cell is a rectangle
        lx, ly = np.ptp(nodes[:2], axis=1)
        a1, a2 = np.array([lx, 0.0]), np.array([0.0, ly])
        _nodes = nodes[:2].copy()
    else:
        # Transform the unit cell from a general parallelogram to a rectangle
        lx, ly = np.linalg.norm(A_mat, axis=1)
        a1, a2 = A_mat
        theta1 = np.arctan2(a1[1], a1[0])
        theta2 = np.arctan2(a2[1], a2[0])
        _nodes_x = np.sin(theta2) * nodes[0] - np.cos(theta2) * nodes[1]
        _nodes_y = -np.sin(theta1) * nodes[0] + np.cos(theta1) * nodes[1]
        _nodes = np.vstack((_nodes_x, _nodes_y)) / np.sin(theta2 - theta1)

    min_x, min_y = np.min(_nodes, axis=1)
    max_x, max_y = np.max(_nodes, axis=1)

    # Find master and slave nodes
    left_nodes = np.argwhere(np.less_equal(_nodes[0], min_x + eps)).flatten()
    bottom_nodes = np.argwhere(np.less_equal(_nodes[1], min_y + eps)).flatten()
    right_nodes = np.argwhere(np.greater_equal(_nodes[0], max_x - eps)).flatten()
    top_nodes = np.argwhere(np.greater_equal(_nodes[1], max_y - eps)).flatten()

    bottom_left_master_node = np.argwhere(
        np.less_equal(_nodes[0], min_x + eps) & np.less_equal(_nodes[1], min_y + eps)
    ).flatten()
    top_left_slave_node = np.argwhere(
        np.less_equal(_nodes[0], min_x + eps) & np.greater_equal(_nodes[1], max_y - eps)
    ).flatten()
    bottom_right_slave_node = np.argwhere(
        np.greater_equal(_nodes[0], max_x - eps) & np.less_equal(_nodes[1], min_y + eps)
    ).flatten()
    top_right_slave_node = np.argwhere(
        np.greater_equal(_nodes[0], max_x - eps)
        & np.greater_equal(_nodes[1], max_y - eps)
    ).flatten()

    left_master_nodes = np.setdiff1d(
        left_nodes, np.hstack([bottom_left_master_node, top_left_slave_node])
    )
    bottom_master_nodes = np.setdiff1d(
        bottom_nodes, np.hstack([bottom_left_master_node, bottom_right_slave_node])
    )
    right_slave_nodes = np.setdiff1d(
        right_nodes, np.hstack([top_right_slave_node, bottom_right_slave_node])
    )
    top_slave_nodes = np.setdiff1d(
        top_nodes, np.hstack([top_left_slave_node, top_right_slave_node])
    )
    assert left_master_nodes.size == right_slave_nodes.size
    assert bottom_master_nodes.size == top_slave_nodes.size

    master_nodes = np.hstack(
        [left_master_nodes, bottom_master_nodes, bottom_left_master_node]
    )
    slave_nodes = np.hstack(
        [
            right_slave_nodes,
            top_slave_nodes,
            top_left_slave_node,
            bottom_right_slave_node,
            top_right_slave_node,
        ]
    )

    # Find interior nodes
    all_nodes = np.arange(nodes.shape[1])
    interior_nodes = np.setdiff1d(all_nodes, np.hstack((master_nodes, slave_nodes)))
    assert master_nodes.size + slave_nodes.size + interior_nodes.size == nodes.shape[1]

    # Compute B_0 matrix
    independent_nodes = np.hstack((master_nodes, interior_nodes))
    rows1, cols1 = independent_nodes, np.arange(independent_nodes.size).astype(int)

    left_coords = _nodes.T[left_master_nodes]
    left_coords[:, 0] += lx
    right_coords = _nodes.T[right_slave_nodes]
    indices = find_indices(independent_nodes, left_master_nodes)
    args = compare_matrices(left_coords, right_coords, precision=6)
    rows2, cols2 = right_slave_nodes, indices[args]

    bottom_coords = _nodes.T[bottom_master_nodes]
    bottom_coords[:, 1] += ly
    top_coords = _nodes.T[top_slave_nodes]
    indices = find_indices(independent_nodes, bottom_master_nodes)
    args = compare_matrices(bottom_coords, top_coords, precision=6)
    rows3, cols3 = top_slave_nodes, indices[args]

    index = find_indices(independent_nodes, bottom_left_master_node)
    rows4 = np.hstack(
        [top_left_slave_node, bottom_right_slave_node, top_right_slave_node]
    )
    cols4 = np.full(rows4.size, index)

    rows = np.hstack([rows1, rows2, rows3, rows4])
    cols = np.hstack([cols1, cols2, cols3, cols4])
    values = np.ones(rows.size)
    B_0 = scipy.sparse.coo_matrix(
        (values, (rows, cols)),
        dtype=int,
        shape=(all_nodes.size, independent_nodes.size),
    )
    assert np.allclose(np.sum(B_0, axis=1), 1)

    B_0 = scipy.sparse.kron(B_0, scipy.sparse.eye(2, dtype=int), format="csc")
    assert B_0.shape == (2 * all_nodes.size, 2 * independent_nodes.size)

    # Compute B_a matrix
    rows = np.hstack([right_slave_nodes, top_slave_nodes])
    cols = np.hstack(
        [np.zeros(right_slave_nodes.size), np.ones(top_slave_nodes.size)]
    ).astype(int)
    if bottom_right_slave_node.size == 1:
        rows = np.append(rows, bottom_right_slave_node)
        cols = np.append(cols, 0)
    if top_left_slave_node.size == 1:
        rows = np.append(rows, top_left_slave_node)
        cols = np.append(cols, 1)
    if top_right_slave_node.size == 1:
        rows = np.append(rows, [top_right_slave_node, top_right_slave_node])
        cols = np.append(cols, [0, 1])
    values = np.ones(rows.size)
    B_a = scipy.sparse.coo_matrix(
        (values, (rows, cols)), dtype=int, shape=(all_nodes.size, 2)
    )
    B_a = scipy.sparse.kron(B_a, np.eye(2, dtype=int))
    assert B_a.shape == (2 * all_nodes.size, 4)

    B_eps = np.array(
        [
            [a1[0], 0.0, a1[1] / 2],
            [0.0, a1[1], a1[0] / 2],
            [a2[0], 0.0, a2[1] / 2],
            [0.0, a2[1], a2[0] / 2],
        ]
    )

    V = np.linalg.det(
        np.array(
            [
                [a1[0], a1[1]],
                [a2[0], a2[1]],
            ]
        )
    )  # Volume
    return B_0, B_a, B_eps, V


def homogenized_constitutive_matrix(
    mat_table,
    mesh_path_name="",
    mesh_file_name="mesh.xdmf",
    mesh=None,
    A_mat=None,
    transform=False,
    gamma=1,
    shear=0,
    rotation=0,
):
    """Compute the homogenized constitutive matrix."""
    if mesh is None:
        with dolfinx.io.XDMFFile(
            MPI.COMM_WORLD, mesh_path_name + mesh_file_name, "r"
        ) as xdmf:
            mesh = xdmf.read_mesh(name="Grid")
    if transform:
        mesh, A_mat = transform_mesh(mesh, gamma=gamma, shear=shear, rotation=rotation)

    K_uc = global_stiffness_matrix_fenicsx(mesh, mat_table)
    nodes = mesh.geometry.x.T

    B_0, B_a, B_eps, V = topology_matrices(nodes, A_mat)
    eps = scipy.sparse.eye(B_0.shape[1]) * 1e-8  # Eliminate the singularity
    lhs = (B_0.T @ K_uc @ B_0).tocsc() + eps
    rhs = (B_0.T @ K_uc @ B_a).tocsc()
    D_0 = -scipy.sparse.linalg.spsolve(lhs, rhs)

    D_a = B_0 @ D_0 + B_a
    K_delta_a = D_a.T @ K_uc @ D_a
    K_eps = B_eps.T @ K_delta_a @ B_eps / V
    return K_eps
