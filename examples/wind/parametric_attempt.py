#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 23:41:53 2021

@author: alexanderniewiarowski
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 12:15:16 2021

@author: alexanderniewiarowski
"""
from dolfin import *
import dolfin as df
import fenics_optim as fo
from fenicsmembranes.parametric_membrane import ParametricMembrane

from fenics_wrinkle.materials.INH import INHMembrane
from fenics_wrinkle.materials.svk import SVKMembrane
from fenics_wrinkle.utils import eigenvalue
from fenics_wrinkle.io import WrinklePlotter
from fenics_wrinkle.bm_data import Mosler
from fenics_wrinkle.utils import expand_ufl, eigenvalue
import numpy as np
import sympy as sp
import sympy.printing.ccode as ccode
import scipy.interpolate
import json
from matplotlib import rc_file
rc_file('../submission/journal_rc_file.rc')
import matplotlib.pyplot as plt



topological_dim = 2
geometrical_dim = 3

COORDS = np.loadtxt("nodes.txt", dtype='float')
CON = np.loadtxt("elements.txt",dtype='uint') - 1  # from MATLAB, start from 0

num_local_vertices = COORDS.shape[0]
num_global_vertices = num_local_vertices  # True if run in serial
num_local_cells = CON.shape[0]
num_global_cells = num_local_cells

# Create mesh object and open editor
mesh = df.Mesh()
editor = df.MeshEditor()
editor.open(mesh, "triangle", topological_dim, geometrical_dim)
editor.init_vertices_global(num_local_vertices, num_global_vertices)
editor.init_cells_global(num_local_cells, num_global_cells)

# Add verticess
for i, coord in enumerate(COORDS):
    editor.add_vertex(i, coord)

# Add cells
for i, cell in enumerate(CON):
    editor.add_cell(i, cell)

# Close editor
editor.close()


f = df.File('mesh.pvd')
f << mesh


# %% Original pressures for comparison
W = FunctionSpace(mesh, 'DG', 0)
ret_dofmap = W.dofmap()

p = Function(W)

PRESSURES = np.loadtxt("pressures.txt", dtype='float')
assert(len(PRESSURES) == num_global_cells)

temparray = np.zeros(num_global_cells)
for c, mesh_cell in enumerate(cells(mesh)):

    temparray[ret_dofmap.cell_dofs(mesh_cell.index())] = PRESSURES[c]

p.vector()[:] = temparray

f = df.File('PRESSURES.pvd')
f << p


#%% build list of cell midpoints
R = 10
MID_PTS = []
for c, mesh_cell in enumerate(cells(mesh)):
    MID_PTS.append([v for v in mesh_cell.midpoint()])
MID_PTS = np.array(MID_PTS)

# Find intersection of sphere with radius 10
NEW_MID_PTS = np.zeros((num_global_cells, 3))
for i, mp in enumerate(MID_PTS):
    scale_factor = R/np.sqrt(mp.dot(mp))
    corrected = mp*scale_factor
    NEW_MID_PTS[i] = corrected
    print(np.sqrt(corrected.dot(corrected)))
    
    #%%
# Build inverse mapping to parametric sphere
PRM_COORDS = np.zeros((num_global_cells, 2))
for i, mp in enumerate(NEW_MID_PTS):
    X = mp[0]
    Y = mp[1]
    Z = mp[2]
    eta = np.arccos(Z/R)
    # xi = np.arcsin(Y/(R*np.sin(eta)))

    xi = np.arccos(X/(R*np.sin(eta)))
    # print(xi, eta)
    PRM_COORDS[i] = [-xi, eta]

#%%

import fenics_wrinkle.geometry.sphere as sphere
geo = sphere.ParametricSphere(R)



def pinnedBCMembrane(membrane):
    bc = []
    # bot = CompiledSubDomain("(near(x[1], 0) && on_boundary)")
    
    # top = CompiledSubDomain("(near(x[1], pi) && on_boundary)", pi=pi)
    
    left = CompiledSubDomain("(near(x[0], -pi) && on_boundary)", pi=pi)
    
    right = CompiledSubDomain("(near(x[0], 0) && on_boundary)", pi=pi)
    bnd = CompiledSubDomain("on_boundary")
    bc.append(DirichletBC(membrane.V, Constant((0,0,0)), bnd))
    # bc.append(DirichletBC(membrane.V, Constant((0,0,0)), left))
    # bc.append(DirichletBC(membrane.V, Constant((0,0,0)), right))
    # bc.append(DirichletBC(membrane.V.sub(0), Constant((0)), left))
    # bc.append(DirichletBC(membrane.V.sub(2), Constant((0)), left))
    # bc.append(DirichletBC(membrane.V.sub(0), Constant((0)), right))
    # bc.append(DirichletBC(membrane.V.sub(2), Constant((0)), right))
    return bc
    
prm_mesh = df.RectangleMesh(Point(-pi,0), Point(0, pi), 15, 15)



V = FunctionSpace(prm_mesh, 'CG', 1)
t = 0.005
E = 0.5E9

nu = 0
mu = E/2/(1+nu)
lamb = E*nu/(1+nu)/(1-2*nu)
# lamb_bar = 2*lamb*mu/(lamb+2*mu)
p = 200
    
class PressureInterpolator(df.UserExpression):
    def __init__(self, XY, P, **kwargs):
        super().__init__(**kwargs)
        self.degree = 5
        # XY = list(zip(X.ravel(), Y.ravel()))
        self.f = scipy.interpolate.CloughTocher2DInterpolator(XY, P.ravel(), fill_value=0)
        # self.f = scipy.interpolate.interp2d(XY[:,0], XY[:,1], P, kind='linear')
    def eval(self, values, x):
        values[0] = self.f(x[0], x[1])

    def value_shape(self):
        return ()
    

p_prn = PressureInterpolator(PRM_COORDS, PRESSURES)

input_dict = {
        'mesh': prm_mesh,
        'geometry': geo,
        'thickness': t,
        'material': 'Incompressible NeoHookean',
        'mu': mu,
        'lmbda': lamb,
        'cylindrical': True,

        'output_file_path': 'parametric_wind',
        'pressure': p,
        'Boundary Conditions': pinnedBCMembrane,
        # 'pbc': pbc,
        'inflation_solver': 'Custom Newton'}

mem = ParametricMembrane(input_dict)
mem.p_ext.assign(project(p_prn, V))
mem.io.write_fields()
mem.inflate(200)

mem.u.vector()[:] = 0

#%%
def linear_volume_potential_split(mem):
    import ufl
    from ufl.algorithms import expand_derivatives
    """

    u_bar.(g1 x g2_bar) + u_bar.(g1_bar x g2) +\
    u.g3_bar + X.(g1 x g2_bar) + X.(g1_bar x g2)

    """

    g1_bar = project(mem.gsub1, mem.V)
    g2_bar = project(mem.gsub2, mem.V)
    g3_bar = project(mem.gsub3, mem.V)
    u_bar = df.Function(mem.V)
    u_bar.assign(mem.u)
    u = mem.u

    # This generates a list of linear terms returned by expand_ufl
    # Currently, expand_ufl doesn't support more complex expressions and
    # it is not possible to accomplish this with one call to expand_ufl
    # Vg1u + Vg2u + Vu
    dV_LINu = expand_ufl(dot(u.dx(0), cross(g2_bar, u_bar))) +\
              expand_ufl(dot(u.dx(1), cross(u_bar, g1_bar))) +\
              expand_ufl(dot(u, g3_bar))
    #  Vg1u_const + Vg2u_const
    dV_LINu_const = expand_ufl(dot(mem.Gsub1, cross(g2_bar, u_bar))) +\
                    expand_ufl(dot(mem.Gsub2, cross(u_bar, g1_bar)))


    # Vg1X + Vg2X
    dV_LINX = expand_ufl(dot(u.dx(0), cross(g2_bar,mem.gamma))) +\
                  expand_ufl(dot(u.dx(1), cross(mem.gamma, g1_bar))) 
                   
                   
      # Vg1X + Vg2X
    dV_LINX_const = expand_ufl(dot(mem.Gsub1, cross(g2_bar, mem.gamma))) +\
                    expand_ufl(dot(mem.Gsub2, cross(mem.gamma, g1_bar))) +\
                    expand_ufl(dot(mem.gamma, g3_bar))
    return dV_LINX + dV_LINu, dV_LINX_const + dV_LINu_const





#%%
def linear_volume_potential(mem, p):
    """

    u_bar.(g1 x g2_bar) + u_bar.(g1_bar x g2) +\
    u.g3_bar + X.(g1 x g2_bar) + X.(g1_bar x g2)

    """
    g1_bar = project(mem.gsub1, mem.V)
    g2_bar = project(mem.gsub2, mem.V)
    g3_bar = project(mem.gsub3, mem.V)
    u_bar = df.Function(mem.V)
    u_bar.assign(mem.u)

    u = mem.u

    # This generates a list of linear terms returned by expand_ufl
    # Currently, expand_ufl doesn't support more complex expressions and
    # it is not possible to accomplish this with one call to expand_ufl
    dV_LIN = expand_ufl(dot(u_bar, cross(mem.Gsub1 + u.dx(0), g2_bar))) +\
              expand_ufl(dot(u_bar, cross(g1_bar, mem.Gsub2 + u.dx(1)))) +\
              expand_ufl(dot(u, g3_bar)) +\
              expand_ufl(dot(mem.gamma, cross(mem.Gsub1 + u.dx(0), g2_bar))) +\
              expand_ufl(dot(mem.gamma, cross(g1_bar, mem.Gsub2 + u.dx(1))))

    return dV_LIN

#%%
def mosek_inflate(self, p, i=0, max_iter=100, tol=1e-4):
    u = self.u
    E_list = []
    E = 1
    itr = 0
    # l1_old = np.mean(df.project(self.lambda1, self.Vs).compute_vertex_values(self.mesh))
    u_old = df.Function(self.V)
    u_old.assign(self.u)
    # for j in range(itr):
    with df.Timer(f"Mosek Inflate Interval {i}, Iteration {itr}"):
        while E > tol:
            if itr > max_iter:
                break

            prob = fo.MosekProblem("No-compression membrane model")
            _ = prob.add_var(self.V, bc=self.bc)
            prob.var[0] = u

            # self.energy = energy = INHMembrane(u, self, degree=2)
            # prob.add_convex_term(self.thickness*self.material.mu/2*self.J_A*energy)
            
            self.energy = energy = SVKMembrane(u, self, lamb, mu, degree=2)
            prob.add_convex_term(Constant(t)*self.J_A*energy) # membrane.J_A

            # U_air_list = linear_volume_potential(self, p)
            # print(type(U_air_list[0]))
            # for dU in U_air_list:
            #     prob.add_obj_func(-Constant(p/3)*dU*dx(self.mesh))
                
                
            vol_lin_u, vol_lin_const = linear_volume_potential_split(self)               
            for dv in vol_lin_u:
                prob.add_obj_func(-Constant(p/3)*dv*dx(self.mesh))
            for dv in vol_lin_const:
                prob.add_obj_func(-Constant(p/3)*dv*dx(self.mesh))
            g3_bar = project(mem.gsub3, mem.V)
            prob.add_obj_func(dot(self.p_ext*g3_bar,u)*dx(self.mesh))
            io = WrinklePlotter(self, energy)
            self.io.add_plotter(io.plot, 'output/xdmf_write_interval', 0)
            prob.parameters["presolve"] = True
            prob.optimize()
            l1 = np.mean(df.project(self.lambda1, self.Vs).compute_vertex_values(self.mesh))
            # E = l1 - l1_old
            E = df.errornorm(u, u_old)
            E_list.append(E)
            print(f'E={E},  l1={l1}, itr={itr}')
            itr += 1
            # l1_old = l1
            u_old.assign(u)

            self.io.write_fields()
    return E_list, itr
