from __future__ import print_function
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

# from mshr import *
set_log_level(40)


class PeriodicBoundary(SubDomain):
    def __init__(self, vertices, tolerance=DOLFIN_EPS):
        """vertices stores the coordinates of the 4 unit cell corners"""
        SubDomain.__init__(self, tolerance)
        self.tol = tolerance
        self.vv = vertices
        self.a1 = self.vv[1, :] - self.vv[0, :]  # first vector generating periodicity
        self.a2 = self.vv[3, :] - self.vv[0, :]  # second vector generating periodicity
        self.inside_p = []
        # check if UC vertices form indeed a parallelogram
        assert np.linalg.norm(self.vv[2, :] - self.vv[3, :] - self.a1) <= self.tol
        assert np.linalg.norm(self.vv[2, :] - self.vv[1, :] - self.a2) <= self.tol

    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary AND NOT on one of the
        # bottom-right or top-left vertices

        is_inside = bool(
            (
                near(x[0], self.vv[0, 0] + x[1] * self.a2[0] / self.vv[3, 1], self.tol)
                or near(
                    x[1], self.vv[0, 1] + x[0] * self.a1[1] / self.vv[1, 0], self.tol
                )
            )
            and (
                not (
                    (
                        near(x[0], self.vv[1, 0], self.tol)
                        and near(x[1], self.vv[1, 1], self.tol)
                    )
                    or (
                        near(x[0], self.vv[3, 0], self.tol)
                        and near(x[1], self.vv[3, 1], self.tol)
                    )
                )
            )
            and on_boundary
        )
        if is_inside:
            self.inside_p.append(x)
        return is_inside

    def map(self, x, y):
        if near(x[0], self.vv[2, 0], self.tol) and near(
            x[1], self.vv[2, 1], self.tol
        ):  # if on top-right corner
            y[0] = x[0] - (self.a1[0] + self.a2[0])
            y[1] = x[1] - (self.a1[1] + self.a2[1])
        elif near(
            x[0], self.vv[1, 0] + x[1] * self.a2[0] / self.vv[2, 1], self.tol
        ):  # if on right boundary
            y[0] = x[0] - self.a1[0]
            y[1] = x[1] - self.a1[1]
        else:  # should be on top boundary
            y[0] = x[0] - self.a2[0]
            y[1] = x[1] - self.a2[1]


def getfit(mesh, pcdtype, tradeoff):
    vol = 1
    if pcdtype == "parallel":
        vertices = np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.5, sqrt(3.0) / 2.0], [0.5, sqrt(3.0) / 2.0]]
        )
        vol = 0.5 * sqrt(3.0)
    else:
        vertices = np.array([[0.0, 0.0], [1, 0], [1, 1], [0, 1]])
    subdomains = MeshFunction("size_t", mesh, 1)
    Em = 50e3
    num = 0.2
    Er = 1
    nur = 0.49
    if tradeoff.startswith("tensor"):
        pass
    material_parameters = [(Er, nur)]
    nphases = len(material_parameters)

    def eps(v):
        return sym(grad(v))

    def sigma(v, i, Eps):
        E, nu = material_parameters[i]
        lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
        mu = E / 2 / (1 + nu)
        lmbda = 2 * mu * lmbda / (lmbda + 2 * mu)
        return lmbda * tr(eps(v) + Eps) * Identity(2) + 2 * mu * (eps(v) + Eps)

    Ve = VectorElement("CG", mesh.ufl_cell(), 2)
    Re = VectorElement("R", mesh.ufl_cell(), 0)
    pb = PeriodicBoundary(vertices, tolerance=1e-10)
    W = FunctionSpace(
        mesh,
        MixedElement([Ve, Re]),
        constrained_domain=pb,
    )
    V = FunctionSpace(mesh, Ve)

    v_, lamb_ = TestFunctions(W)
    dv, dlamb = TrialFunctions(W)
    w = Function(W)
    dx = Measure("dx")(subdomain_data=subdomains)

    Eps = Constant(((0, 0), (0, 0)))
    F = sum([inner(sigma(dv, i, Eps), eps(v_)) * dx(i) for i in range(nphases)])
    a, L = lhs(F), rhs(F)
    a += dot(lamb_, dv) * dx + dot(dlamb, v_) * dx

    def macro_strain(i):
        """returns the macroscopic strain for the 3 elementary load cases"""
        Eps_Voigt = np.zeros((3,))
        Eps_Voigt[i] = 1
        return np.array(
            [[Eps_Voigt[0], Eps_Voigt[2] / 2.0], [Eps_Voigt[2] / 2.0, Eps_Voigt[1]]]
        )

    def stress2Voigt(s):
        return as_vector([s[0, 0], s[1, 1], s[0, 1]])

    Chom = np.zeros((3, 3))
    for j, case in enumerate(["Exx", "Eyy", "Exy"]):
        # print("Solving {} case...".format(case))
        Eps.assign(Constant(macro_strain(j)))
        try:
            solve(a == L, w, [], solver_parameters={"linear_solver": "cg"})
        except Exception as e:
            print("error")
            random_number1 = np.random.uniform(2, 3)
            random_number2 = np.random.uniform(2, 3)
            return random_number1, random_number2, False
        (v, lamb) = split(w)
        Sigma = np.zeros((3,))
        for k in range(3):
            Sigma[k] = (
                assemble(
                    sum(
                        [
                            stress2Voigt(sigma(v, i, Eps))[k] * dx(i)
                            for i in range(nphases)
                        ]
                    )
                )
                / vol
            )
        Chom[j, :] = Sigma

    # Chom = (Chom + Chom.T) / 2
    # lmbda_hom = Chom[0, 1]
    # mu_hom = Chom[2, 2]
    # Eeff = mu_hom * (3 * lmbda_hom + 2 * mu_hom) / (lmbda_hom + mu_hom)
    # nueff = lmbda_hom / (lmbda_hom + mu_hom) / 2

    # y = SpatialCoordinate(mesh)
    # plt.figure()
    # p = plot(0.5 * (dot(Eps, y) + v1), mode="displacement", title=case)
    # plt.colorbar(p)
    # plt.show()
    # plt.savefig("deformed.png")
    # eig = np.linalg.eig(Chom)
    # print(f"eig {eig}")
    # print(-(Chom[0,0]+Chom[1,1])/2.,Chom[0,1])
    # return (Chom[0, 0] + Chom[1, 1]) / 2.0, -Chom[0, 1]
    # print(Chom)
    # Chom = sp.linalg.inv(Chom)
    # nueff = -0.5 * (Chom[1, 0] / Chom[0, 0] + Chom[0, 1] / Chom[1, 1])  # Poisson ratio
    # Eeff = 0.5 * (1 / Chom[0, 0] + 1 / Chom[1, 1])  # Avg young mod

    # Eeff = Chom[0, 0]
    # nueff = Chom[0, 1] / Chom[0, 0]
    # Eeff = Chom[0, 2]
    # nueff = Chom[0, 1]

    if tradeoff == "Max:E-Min:nu":
        Chom = sp.linalg.inv(Chom)
        nueff = -0.5 * (
            Chom[1, 0] / Chom[0, 0] + Chom[0, 1] / Chom[1, 1]
        )  # Poisson ratio
        Eeff = -0.5 * (1 / Chom[0, 0] + 1 / Chom[1, 1])  # Avg young mod

    elif tradeoff == "Max:E-Max:nu":
        Chom = sp.linalg.inv(Chom)
        nueff = 0.5 * (
            Chom[1, 0] / Chom[0, 0] + Chom[0, 1] / Chom[1, 1]
        )  # Poisson ratio
        Eeff = -0.5 * (1 / Chom[0, 0] + 1 / Chom[1, 1])  # Avg young mod
    elif tradeoff == "tensor1down":
        Eeff = -Chom[0, 0]
        nueff = Chom[0, 1] / Chom[0, 0]
    elif tradeoff == "tensor1up":
        Eeff = -Chom[0, 0]
        nueff = -Chom[0, 1] / Chom[0, 0]
    elif tradeoff == "tensor1leftdown":
        Eeff = Chom[0, 0]
        nueff = Chom[0, 1] / Chom[0, 0]
    elif tradeoff == "tensor1leftup":
        Eeff = Chom[0, 0]
        nueff = -Chom[0, 1] / Chom[0, 0]
    elif tradeoff == "tensor2down":
        Eeff = -Chom[0, 0]
        nueff = Chom[1, 1]
    elif tradeoff == "tensor2up":
        Eeff = -Chom[0, 0]
        nueff = -Chom[1, 1]
    elif tradeoff == "tensor2leftdown":
        Eeff = Chom[0, 0]
        nueff = Chom[1, 1]
    elif tradeoff == "tensor2leftup":
        Eeff = Chom[0, 0]
        nueff = -Chom[1, 1]
    elif tradeoff == "tensor3down":
        Eeff = -Chom[0, 0]
        nueff = Chom[2, 2]
    elif tradeoff == "tensor3up":
        Eeff = -Chom[0, 0]
        nueff = -Chom[2, 2]
    elif tradeoff == "tensor3leftdown":
        Eeff = Chom[0, 0]
        nueff = Chom[2, 2]
    elif tradeoff == "tensor3leftup":
        Eeff = Chom[0, 0]
        nueff = -Chom[2, 2]
    elif tradeoff == "shear_normalright":
        Eeff = -Chom[0, 2]
        nueff = Chom[0, 1]
    elif tradeoff == "shear_normalleft":
        Eeff = Chom[0, 2]
        nueff = Chom[0, 1]
    elif tradeoff == "shear_bulk":
        Eeff = -0.5 * (Chom[0, 0] + Chom[1, 1] + 2 * Chom[0, 1])
        nueff = Chom[2, 2]
    elif tradeoff == "shear_bulkup":
        Eeff = -0.5 * (Chom[0, 0] + Chom[1, 1] + 2 * Chom[0, 1])
        nueff = -Chom[2, 2]

    return Eeff, nueff, True
