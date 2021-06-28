#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 22:06:55 2021

@author: alexanderniewiarowski
"""

from fenics_optim import *

from fenicsmembranes.parametric_membrane import *
import matplotlib.pyplot as plt
import numpy as np
from fenics_wrinkle.bm_data import *

from fenics_optim import ConvexFunction

from fenics_wrinkle.io import WrinklePlotter
bm = KannoIsotropic()
# parameters["form_compiler"]["representation"] = 'uflacs'
np.testing.assert_array_almost_equal_nulp(C(bm.lamb_bar, bm.mu), C_plane_stress(bm.E, bm.nu))
C = C_plane_stress(bm.lamb, bm.nu)

N = 20
mesh = RectangleMesh(Point(0,0), Point(bm.width, bm.height), 2*N, N, )

def bottom(x, on_boundary):
    return near(x[1], 0) and on_boundary
def top(x, on_boundary):
    return near(x[1], bm.height) and on_boundary
def left(x, on_boundary):
    return near(x[0], 0) and on_boundary
def right(x, on_boundary):
    return near(x[0], bm.width) and on_boundary
def bnd(x, on_boundary):
    return on_boundary


ux = 2
uy = -2.
u_y = Expression(('x[1]*(-10/100)'), degree=1)


class Geometry:
    def __init__(self):
        self.gamma = project(Expression(('x[0]', 'x[1]', 0), degree=1), VectorFunctionSpace(mesh, 'CG', 2, dim=3))
        self.Gsub1 = project(Constant((1, 0, 0)), VectorFunctionSpace(mesh, 'CG', 2, dim=3))
        self.Gsub2 = project(Constant((0, 1, 0)), VectorFunctionSpace(mesh, 'CG', 2, dim=3))



UX = Constant(0)
UY = Constant(0)
def bc(mem):
    bc = [DirichletBC(mem.V.sub(0), Constant(0), left),
          DirichletBC(mem.V.sub(0), UX, right),
          DirichletBC(mem.V.sub(1), Constant(0), bottom), 
          DirichletBC(mem.V.sub(1), UY, top),
          DirichletBC(mem.V.sub(2),  Constant(0), bnd) ]
    
    
    # c = -5
    # bc = [DirichletBC(mem.V.sub(1), Constant(c), bottom),
    #       DirichletBC(mem.V.sub(0), Constant(c), left),
    #       DirichletBC(mem.V.sub(0), Constant(-c), right),
    #       DirichletBC(mem.V.sub(1), Constant(-c), top) ]
    return bc

input_dict = {
        'mesh': mesh,
        'resolution': [30,30],
        'geometry': Geometry(),
        'thickness': bm.t,
        'material': 'Incompressible NeoHookean',
        'mu': bm.mu,
        'cylindrical': True,
        'output_file_path': 'benchmark_idea_cross_section',
        'pressure': 0,
        'Boundary Conditions': bc}

middle = CompiledSubDomain("near(x[0], 100)")

markers = MeshFunction('size_t', mesh, mesh.geometric_dimension()-1)
markers.set_all(0)

middle.mark(markers, 1)
File('sub.pvd') <<markers
dS = Measure('dS', domain = mesh, subdomain_data = markers)

mem = membrane = ParametricMembrane((input_dict))
mem.io.write_fields()
#
t_j =[]
simplified = []
t_J = []

uys = np.concatenate((np.zeros(1), np.linspace(0,10, 11)))

tranverse = False
for i, uy in enumerate(uys):
    if i == 1:
        UX.assign(1)
    print('UX, UY:', float(UX), float(UY))
    UY.assign(-uy)
    prob = MosekProblem("No-compression membrane model")
    u__ = prob.add_var(mem.V, bc=mem.bc)
    prob.var[0] = mem.u   # replace
    u = mem.u
    from fenics_wrinkle.materials.INH import INHMembrane
    energy = INHMembrane(u, mem, degree=2)
    # energy = NonLinearMembrane(u, mem, degree=2)
    prob.add_convex_term(bm.t*bm.mu/2*energy)
    prob.parameters["presolve"] = True
    prob.optimize()        
    
    
    
    # from utils import eigenvalue, eig_vecmat
    # # E = membrane.E
    # # E_el = as_tensor([[energy.Eel[0],energy.Eel[2]],
    # #                   [energy.Eel[2],energy.Eel[1]]])
    
    # # # Veps = VectorFunctionSpace(mesh, "DG", 0, dim=3)
    # # # E_el = project(energy.Eel, Veps, form_compiler_parameters={"quadrature_degree": energy.degree})
    # # # E_w = project(E - Eel, Veps)
    # # E_w = E - E_el
    
    # E = membrane.E
    # E_el = 0.5*(energy.C_elastic - membrane.C_0)
    # E_w = -(E - E_el)
    
    # def write_eigval(A, l1_name, l2_name):
    #     l1, l2 = eigenvalue(A)
        
    #     l1 = project(l1, FunctionSpace(mem.mesh, 'CG',1), form_compiler_parameters={"quadrature_degree": energy.degree})
    #     l1.rename(l1_name, l1_name)
        
    #     l2 = project(l2, FunctionSpace(mem.mesh, 'CG',1), form_compiler_parameters={"quadrature_degree": energy.degree})
    #     l2.rename(l2_name, l2_name)
        
    #     mem.io.add_extra_output_function(l1)
    #     mem.io.add_extra_output_function(l2)
    #     # mem.io.write_fields()
        
    
    
    # def write_eigvec(A, v1_name, v2_name):
    #     v1, v2 = eig_vecmat(A)
    #     S = VectorFunctionSpace(mem.mesh, 'CG', 1)
    #     eig1 = project(v1, S, form_compiler_parameters={"quadrature_degree": energy.degree})
    #     eig1.rename(v1_name, v1_name)
    #     eig2 = project(v2, S, form_compiler_parameters={"quadrature_degree": energy.degree})
    #     eig2.rename(v2_name, v2_name)
    #     mem.io.add_extra_output_function(eig1)
    #     mem.io.add_extra_output_function(eig2)
    #     # mem.io.write_fields()
    
    
  #%%  
    # write_eigval(E_w, 'Ew1', 'Ew2')
    # write_eigvec(E_w, 'Ew_v1', 'Ew_v2')
    
    # write_eigval(E_el, 'E_el1', 'E_el2')
    # write_eigvec(E_el, 'E_el_eig1', 'E_el_eig2')

    io = WrinklePlotter(mem, energy)
    mem.io.add_plotter(io.plot, 'output/xdmf_write_interval', 0)
    mem.io.write_fields()
    #%
    mf = MeshFunction('size_t', mesh, 2, mesh.domains())
    t_ji = bm.t*sqrt((det(mem.C_0)('-')/det(mem.C_n)('-')))*mem.j_a('-')*dS(1) + Constant(0)*dx(domain=mesh, subdomain_data=mf)
    simplified_i = bm.t*det(mem.C_n)('-')*dS(1) + Constant(0)*dx(domain=mesh, subdomain_data=mf)
    t_Ji = bm.t*sqrt((det(mem.C_0)('-')/det(mem.C_n)('-')))*dS(1) + Constant(0)*dx(domain=mesh, subdomain_data=mf)
    # t_i = bm.t*mem.lambda3('-')*dS(1)
    t_j.append(assemble(t_ji))
    simplified.append(assemble(simplified_i))
    t_J.append(assemble(t_Ji))
    # L3 = dot(grad(f)('+'), n('+'))*dS 

    # print(t, t/(bm.height))
    
#%%   
fig, ax = plt.subplots()
ax.plot(uys, t_j, '-', label='cross sectional area in initial config:' + r'$\int h J_a dS = \int H dS$')
# ax.plot(us, simplified, '.', label='simplifed')
ax.plot(uys, t_J, '--', label='cross sectional area in deformed config \n w/o correcting for change in height:'+ r'$\int h dS $')
alpha = (bm.height-uys)/bm.height
ax.plot(uys, t_J*alpha, '--', label='corrected cross sectional area:'+ r'$ \alpha \int h dS,  \alpha=(height-\Delta u_y)/height$')
ax.set_xlabel('tranverse displacement')
ax.set_ylabel('cross section area')
plt.legend()