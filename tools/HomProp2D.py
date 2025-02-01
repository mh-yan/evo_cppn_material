import numpy as np
import scipy as sp
from numpy import linalg as LA
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
import cvxopt
import cvxopt.cholmod


def GetHomProp2D_PlaneStress(MetaDesign, E1, nu1, E2, nu2, Amat=np.eye(2)):
    # Get unit cell full stiffness matrix Kuc - assume plane Strain, thickness = 1
    # 1 for stiff material;  0 for soft material
    #     MetaDesign = np.concatenate((MetaDesign,np.flip(MetaDesign,0)),axis=0)
    #     MetaDesign = np.concatenate((MetaDesign,np.flip(MetaDesign,1)),axis=1)

    nelx = MetaDesign.shape[1]
    nely = MetaDesign.shape[0]
    ndof = 2 * (nelx + 1) * (nely + 1)

    KA = np.array(
        [
            [12.0, 3.0, -6.0, -3.0, -6.0, -3.0, 0.0, 3.0],
            [3.0, 12.0, 3.0, 0.0, -3.0, -6.0, -3.0, -6.0],
            [-6.0, 3.0, 12.0, -3.0, 0.0, -3.0, -6.0, 3.0],
            [-3.0, 0.0, -3.0, 12.0, 3.0, -6.0, 3.0, -6.0],
            [-6.0, -3.0, 0.0, 3.0, 12.0, 3.0, -6.0, -3.0],
            [-3.0, -6.0, -3.0, -6.0, 3.0, 12.0, 3.0, 0.0],
            [0.0, -3.0, -6.0, 3.0, -6.0, 3.0, 12.0, -3.0],
            [3.0, -6.0, 3.0, -6.0, -3.0, 0.0, -3.0, 12.0],
        ]
    )
    KB = np.array(
        [
            [-4.0, 3.0, -2.0, 9.0, 2.0, -3.0, 4.0, -9.0],
            [3.0, -4.0, -9.0, 4.0, -3.0, 2.0, 9.0, -2.0],
            [-2.0, -9.0, -4.0, -3.0, 4.0, 9.0, 2.0, 3.0],
            [9.0, 4.0, -3.0, -4.0, -9.0, -2.0, 3.0, 2.0],
            [2.0, -3.0, 4.0, -9.0, -4.0, 3.0, -2.0, 9.0],
            [-3.0, 2.0, 9.0, -2.0, 3.0, -4.0, -9.0, 4.0],
            [4.0, 9.0, 2.0, 3.0, -2.0, -9.0, -4.0, -3.0],
            [-9.0, -2.0, 3.0, 2.0, 9.0, 4.0, -3.0, -4.0],
        ]
    )

    KE1 = E1 / (1 - nu1**2) / 24 * (KA + nu1 * KB)
    KE2 = E2 / (1 - nu2**2) / 24 * (KA + nu2 * KB)

    # FE: Build the index vectors for the for coo matrix format.
    edofMat = np.zeros((nelx * nely, 8), dtype=np.int64)
    for elx in range(nelx):
        for ely in range(nely):
            el = ely + elx * nely
            n1 = (nely + 1) * elx + ely
            n2 = (nely + 1) * (elx + 1) + ely
            edofMat[el, :] = np.array(
                [
                    2 * n1 + 2,
                    2 * n1 + 3,
                    2 * n2 + 2,
                    2 * n2 + 3,
                    2 * n2,
                    2 * n2 + 1,
                    2 * n1,
                    2 * n1 + 1,
                ]
            )

    # Construct the index pointers for the coo format
    iK = np.kron(edofMat, np.ones((8, 1))).flatten()
    jK = np.kron(edofMat, np.ones((1, 8))).flatten()
    sK = ((KE1.flatten()[np.newaxis]).T * MetaDesign.flatten()).flatten("F") + (
        (KE2.flatten()[np.newaxis]).T * (1 - MetaDesign).flatten()
    ).flatten("F")
    Kuc = sp.sparse.coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsr()
    #     Kuc = 0.5 * (Kuc.T+Kuc)
    #     Kuc = cvxopt.spmatrix(sK,iK,jK,(ndof,ndof))

    # Get unit cell periodic topology
    M = np.eye((nelx + 1) * (nely + 1))
    M[0, [nely, (nely + 1) * nelx, (nelx + 1) * (nely + 1) - 1]] = 1
    M[1:nely, range(1 + (nely + 1) * nelx, nely + (nely + 1) * nelx)] = np.eye(nely - 1)
    M[
        np.arange((nely + 1), (nely + 1) * nelx, (nely + 1)),
        np.arange(2 * nely + 1, (nely + 1) * nelx, (nely + 1)),
    ] = 1
    M = M[np.sum(M, axis=0) < 2, :].T
    # Compute homogenized elasticity tensor
    B0 = sp.sparse.kron(M, np.eye(2))
    #     print(B0)
    Bep = np.array(
        [
            [Amat[0, 0], 0.0, Amat[1, 0] / 2],
            [0.0, Amat[1, 0], Amat[0, 0] / 2],
            [Amat[0, 1], 0.0, Amat[1, 1] / 2],
            [0.0, Amat[1, 1], Amat[0, 1] / 2],
        ]
    )
    BaTop = np.zeros(((nelx + 1) * (nely + 1), 2), dtype=np.single)
    BaTop[(nely + 1) * nelx + np.arange(0, nely + 1), 0] = 1
    BaTop[np.arange(nely, (nely + 1) * (nelx + 1), (nely + 1)), 1] = -1
    Ba = np.kron(BaTop, np.eye(2, dtype=float))

    TikReg = sp.sparse.eye(B0.shape[1]) * 1e-8
    F = (Kuc.dot(B0)).T.dot(Ba)
    Kg = (Kuc.dot(B0)).T.dot(B0) + TikReg
    Kg = (0.5 * (Kg.T + Kg)).tocoo()
    #     Kgc, lower = sp.linalg.cho_factor(0.5 * (Kg.T + Kg))
    #     D0 = sp.linalg.cho_solve((Kgc,lower),F)
    #     D0 = np.linalg.solve(0.5*(Kg.T + Kg),F)
    Ksp = cvxopt.spmatrix(Kg.data, Kg.row.astype(np.int64), Kg.col.astype(np.int64))
    Fsp = cvxopt.matrix(F)
    cvxopt.cholmod.linsolve(Ksp, Fsp)
    #     D0 = sp.sparse.linalg.spsolve(0.5*(Kg.T + Kg), F)
    D0 = np.array(Fsp)
    Da = -B0.dot(D0) + Ba
    Kda = (Kuc.dot(Da)).T.dot(Da)
    Chom = (Kda.dot(Bep)).T.dot(Bep) / LA.det(Amat)
    # Modes = Da.dot(Bep)

    # Chris said to replace the output with this:
    Chom = sp.linalg.inv(Chom)
    nueff = -0.5 * (Chom[1, 0] / Chom[0, 0] + Chom[0, 1] / Chom[1, 1])  # Poisson ratio
    Eeff = 0.5 * (1 / Chom[0, 0] + 1 / Chom[1, 1])  # Avg young mod
    #     print("nueff: ", nueff, "Eeff ", Eeff)

    return -nueff, Eeff
    # change this to nueff, Eeff and optimize both (very stiff?, very negative poisson)


# def PeriodicGeometryVoid(MetaDesign):
#     size = MetaDesign.shape
#     match_lr = sum((MetaDesign[0,:]*MetaDesign[-1,:])/size[0])**2
#     match_ud = sum((MetaDesign[:,0]*MetaDesign[:,-1])/size[1])**2
#     return (match_lr+match_ud)/2.0


def PeriodicGeometry(MetaDesign):
    tol = 0.1
    size = MetaDesign.shape
    match_lr = abs(MetaDesign[:, 0] - MetaDesign[:, -1]) < tol
    match_ud = abs(MetaDesign[0, :] - MetaDesign[-1, :]) < tol
    match_lr_novoid = match_lr * (MetaDesign[:, 0] > 0)
    match_ud_novoid = match_ud * (MetaDesign[0, :] > 0)

    # if the material is fully solid
    #     score_lr = sum(match_lr).astype(np.double)/size[0]
    #     score_ud = sum(match_ud).astype(np.double)/size[1]
    # if the material contains void (encoded by "0")
    score_lr = sum(match_lr_novoid).astype(np.double) / size[0]
    score_ud = sum(match_ud_novoid).astype(np.double) / size[1]

    periodicity_score = (score_lr**2 + score_ud**2) / 2.0
    return periodicity_score


# E1 = 1000
# nu1 = 0.33
# E2 = 1
# nu2 = 0.0

# design = np.ones((64,64))
# # design is a (binary) matrix
# Ehom = GetHomProp2D_PlaneStress(design,E1,nu1,E2,nu2)
