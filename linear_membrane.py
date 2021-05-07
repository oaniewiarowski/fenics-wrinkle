#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 11:18:24 2021

@author: bleyerj
Modified by a niewiarowski

Case 1:
    Topological Dimension: 2
    Geometrical Dimension: 2
    Coordinate System: Cartesian
    Strain Measure: Small strain theory
    Material Law: Linear (SVK)
"""
from fenics_optim import *
from fenics_optim.quadratic_cones import get_slice
import matplotlib.pyplot as plt
import numpy as np
from bm_data import *


def C(lamb, mu):
    return np.array([[lamb+2*mu, lamb, 0],
                     [lamb, lamb+2*mu, 0],
                     [0, 0, mu]])


def C_plane_stress(E, nu):
    return (E/(1-nu**2))*np.array([[1, nu, 0],
                                   [nu, 1, 0],
                                   [0, 0, (1-nu)/2]])



bm = KannoIsotropic()
np.testing.assert_array_almost_equal_nulp(C(bm.lamb_bar, bm.mu), C_plane_stress(bm.E, bm.nu))
#%%

N = 20
ux = 1
uy = 1
mesh = RectangleMesh(Point(0,0), Point(bm.width, bm.height), 2*N, N)#, "crossed")
V = VectorFunctionSpace(mesh, "CG", 2)


def left(x, on_boundary):
    return near(x[0], 0) and on_boundary
def bottom(x, on_boundary):
    return near(x[1], 0) and on_boundary
def top(x, on_boundary):
    return near(x[1], bm.height) and on_boundary
def right(x, on_boundary):
    return near(x[0], bm.width) and on_boundary

bc = [DirichletBC(V, Constant((0, 0)), bottom),
      DirichletBC(V, Constant((ux, uy)), top)]

#%%
C = C_plane_stress(bm.lamb, bm.nu)
Q = np.linalg.cholesky(C)
Qinv = as_matrix(np.linalg.inv(Q))

prob = MosekProblem("No-compression membrane model")
u = prob.add_var(V, bc=bc)


class LinearMembrane(ConvexFunction):
    def conic_repr(self, X):
        y = self.add_var(dim=5, cone=RQuad(5), name="y")
        # elastic strain
        Eel = dot(Qinv.T, get_slice(y, 2))
        self.add_eq_constraint([None, y[1]], b=1)
        # wrinkling strain (up to a negative sign)
        Ew = self.add_var(dim=3, cone=SDP(2))

        self.add_eq_constraint([X, -Eel, Ew])
        self.set_linear_term([None, y[0]])


energy = LinearMembrane(sym(grad(u)), degree=2)
prob.add_convex_term(Constant(bm.t)*energy)


prob.parameters["presolve"] = True
prob.optimize()
#%
plot(u, mode="displacement")
plt.show()

y = prob.get_var("y")
#%%
Veps = VectorFunctionSpace(mesh, "DG", 0, dim=3)
Eel = project(dot(Qinv.T, get_slice(y, 2)), Veps, form_compiler_parameters={"quadrature_degree": energy.degree})
Ew = project(to_vect(sym(grad(u))) - Eel, Veps)

pl = plot(Ew[1])
plt.colorbar(pl)
plt.show()