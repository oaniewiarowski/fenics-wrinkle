#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 17:02:43 2021

@author: alexanderniewiarowski
"""

from fenics_optim import *
from fenics_optim.quadratic_cones import get_slice
from fenicsmembranes.parametric_membrane import *
import matplotlib.pyplot as plt
import numpy as np
from fenics_wrinkle.bm_data import *
from fenics_wrinkle.materials.INH import INHMembrane
# from fenics_wrinkle.materials.svk import NonLinearMembrane

from fenics_wrinkle.utils import expand_ufl
from fenics_wrinkle.io import WrinklePlotter

bm = KannoIsotropic()
np.testing.assert_array_almost_equal_nulp(C(bm.lamb_bar, bm.mu), C_plane_stress(bm.E, bm.nu))

C = C_plane_stress(bm.lamb, bm.nu)

width = 10
height = 10
H = 5 # vertical displacement of the hypar
N = 40  # mesh resolution
c = 2  #  in-plane stretching
mesh = RectangleMesh(Point(0,0), Point(width, height), N, N, 'crossed')

def NE(x):
    return near(x[0], width) and near(x[1], height)
def SE(x):
    return near(x[0], width) and near(x[1], 0)
def SW(x):
    return near(x[0], 0) and near(x[1], 0)
def NW(x):
    return near(x[0], 0) and near(x[1], height)

def bc(membrane):
    bc = [DirichletBC(membrane.V, Constant((c, -c, 0)), SE, method='pointwise'),
          DirichletBC(membrane.V, Constant((-c, c, 0)), NW, method='pointwise'),
          DirichletBC(membrane.V, Constant((-c, -c, H)), SW, method='pointwise'),
          DirichletBC(membrane.V, Constant((c, c, H)), NE, method='pointwise')]
    return bc


class Geometry:
    def __init__(self):
        V = VectorFunctionSpace(mesh, 'CG', 1, dim=3)
        self.gamma = project(Expression(('x[0]', 'x[1]', 0), degree=1), V)
        self.Gsub1 = project(Constant((1, 0, 0)), V)
        self.Gsub2 = project(Constant((0, 1, 0)), V)

geo = Geometry()

input_dict = {
        'mesh': mesh,
        'resolution': [30,30],
        'geometry': geo,
        'thickness': bm.t,
        'material': 'Incompressible NeoHookean',
        'mu': bm.mu,
        'cylindrical': True,
        'output_file_path': 'hypar',
        'pressure': 0,
        'Boundary Conditions': bc}

mem = membrane = ParametricMembrane((input_dict))
mem.io.write_fields()

prob = MosekProblem("No-compression membrane model")
u__ = prob.add_var(membrane.V, bc=membrane.bc)
prob.var[0] = membrane.u   # replace
u = membrane.u

energy = INHMembrane(u, mem, degree=2)
# prob.add_convex_term(Constant(bm.t)*energy)
prob.add_convex_term(bm.t*bm.mu/2*energy)
io = WrinklePlotter(mem, energy)
mem.io.add_plotter(io.plot, 'output/xdmf_write_interval', 0)   
prob.parameters["presolve"] = True

prob.optimize()
mem.io.write_fields()

#%%
class PressureLoad(ConvexFunction):

    def __init__(self, u, mem, **kwargs):
        self.mem = mem
        ConvexFunction.__init__(self, u, parameters=None, **kwargs)

    def conic_repr(self, u):
        n_star = self.add_var(dim=3, cone=None, name='pressure')
        n = as_vector([n_star[0], n_star[1], n_star[2]])
        g3 = as_vector([-u[2].dx(0) - u[2].dx(0)*u[1].dx(1) + u[1].dx(0)*u[2].dx(1), 
                        u[2].dx(0)*u[0].dx(1) - u[2].dx(1) - u[0].dx(0)*u[2].dx(1),
                        u[0].dx(0) + u[1].dx(1) +u[0].dx(0)*u[1].dx(1) - u[1].dx(0)*u[0].dx(1)])
        b = as_vector([0,0,-1])
        self.add_eq_constraint([-g3, None, None,None, n], b=-b)
        self.set_linear_term([None, None, None, None, -dot(n,u)])
# p = PressureLoad(u, mem)
# prob.add_convex_term(-dot(mem.gsub3,u)*dx(mem.mesh))
# prob.add_convex_term(p)
prob.optimize()
mem.io.write_fields()

#%%
from utils import eigenvalue, eig_vecmat
# E = membrane.E
# E_el = as_tensor([[energy.Eel[0],energy.Eel[2]],
#                   [energy.Eel[2],energy.Eel[1]]])

# # Veps = VectorFunctionSpace(mesh, "DG", 0, dim=3)
# # E_el = project(energy.Eel, Veps, form_compiler_parameters={"quadrature_degree": energy.degree})
# # E_w = project(E - Eel, Veps)
# E_w = E - E_el

E = membrane.E
E_el = 0.5*(energy.C_elastic - membrane.C_0)
E_w = (E - E_el)

def write_eigval(A, l1_name, l2_name):
    l1, l2 = eigenvalue(A)
    
    l1 = project(l1, FunctionSpace(mem.mesh, 'CG',1), form_compiler_parameters={"quadrature_degree": energy.degree})
    l1.rename(l1_name, l1_name)
    
    l2 = project(l2, FunctionSpace(mem.mesh, 'CG',1), form_compiler_parameters={"quadrature_degree": energy.degree})
    l2.rename(l2_name, l2_name)
    
    mem.io.add_extra_output_function(l1)
    mem.io.add_extra_output_function(l2)
    # mem.io.write_fields()
    


def write_eigvec(A, v1_name, v2_name):
    v1, v2 = eig_vecmat(A)
    S = VectorFunctionSpace(mem.mesh, 'CG', 1)
    eig1 = project(v1, S, form_compiler_parameters={"quadrature_degree": energy.degree})
    eig1.rename(v1_name, v1_name)
    eig2 = project(v2, S, form_compiler_parameters={"quadrature_degree": energy.degree})
    eig2.rename(v2_name, v2_name)
    mem.io.add_extra_output_function(eig1)
    mem.io.add_extra_output_function(eig2)
    # mem.io.write_fields()


#%
J = det(energy.C_elastic)/det(mem.C_0)

import ufl
i,j = ufl.indices(2)
gsup = mem.get_metric(mem.gsup1, mem.gsup2)
gsup = inv(energy.C_elastic)
sigma = as_tensor(bm.mu/J*(mem.C_0_sup[i,j] - 1/J**2*gsup[i,j]), [i,j] )

sigma = S = as_tensor(bm.mu*(mem.C_0_sup[i,j] - 1/J*gsup[i,j]), [i,j] )
# sigma = mem.F_n*S*mem.F_n.T
write_eigval(sigma, 'sigma1', 'sigma2')
write_eigvec(sigma, 'sigma1v', 'sigma2v')




write_eigval(E_w, 'Ew1', 'Ew2')
write_eigvec(E_w, 'Ew_v1', 'Ew_v2')

write_eigval(E_el, 'E_el1', 'E_el2')
write_eigvec(E_el, 'E_el_eig1', 'E_el_eig2')



# 2PK S
gsup = inv(energy.C_elastic)
S = as_tensor(bm.mu*(mem.C_0_sup[i,j] - 1/J*gsup[i,j]), [i,j] )
s1, s2 = eigenvalue(S)
sv1, sv2 = eig_vecmat(S)

# drop z comp of Fn
Fn = as_tensor([[mem.gsub1[0], mem.gsub1[1]],
                [mem.gsub2[0], mem.gsub2[1]]]).T

sigma_el = (s1/J)*outer(dot(Fn,sv1), dot(Fn,sv1))

write_eigval(sigma_el, 'sigma1_el', 'sigma2_el')
write_eigvec(sigma_el, 'sigma1v_el', 'sigma2v_el')



mem.io.write_fields()