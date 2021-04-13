#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 11:18:24 2021

@author: bleyerj
"""

from fenics_optim import *
import matplotlib.pyplot as plt

N = 20
mesh = UnitSquareMesh(N, N, "crossed")

V = VectorFunctionSpace(mesh, "CG", 2)

def left(x, on_boundary):
    return near(x[0], 0) and on_boundary
def bottom(x, on_boundary):
    return near(x[1], 0) and on_boundary
def top(x, on_boundary):
    return near(x[1], 1) and on_boundary
def right(x, on_boundary):
    return near(x[0], 1) and on_boundary

bc = [DirichletBC(V, Constant((0., 0.)), left),
      DirichletBC(V, Constant((0.1,0.)), right)]

E = 10
nu = 0.49
mu = E/2/(1+nu)
lamb = E*nu/(1+nu)/(1-2*nu)
C = np.array([[lamb+2*mu, lamb, 0],
              [lamb, lamb+2*mu, 0],
              [0, 0, 2*mu]])
Q = as_matrix(np.linalg.cholesky(C))

prob = MosekProblem("No-compression membrane model")
u = prob.add_var(V, bc=bc)


class LinearMembrane(ConvexFunction):
    def conic_repr(self, X):
        y = self.add_var(dim=5, cone=RQuad(5), name="y")
        # elastic strain
        Eel = dot(Q.T, get_slice(y, 2))
        self.add_eq_constraint([None, y[1]], b=1)
        # wrinkling strain (up to a negative sign)
        Ew = self.add_var(dim=3, cone=SDP(2))

        self.add_eq_constraint([X, -Eel, Ew])
        self.set_linear_term([None, y[0]])


energy = LinearMembrane(sym(grad(u)), degree=2)
prob.add_convex_term(energy)


prob.parameters["presolve"] = True
prob.optimize()

plot(u, mode="displacement")
plt.show()

y = prob.get_var("y")

Veps = VectorFunctionSpace(mesh, "DG", 0, dim=3)
Eel = project(dot(Q.T, get_slice(y, 2)), Veps, form_compiler_parameters={"quadrature_degree": energy.degree})
Ew = project(to_vect(sym(grad(u))) - Eel, Veps)

pl = plot(Ew[1])
plt.colorbar(pl)
plt.show()