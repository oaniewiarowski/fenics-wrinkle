#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 23:41:53 2021

@author: alexanderniewiarowski
"""

import dolfin as df
from dolfin import Constant, project, pi, dx, dot
import fenics_optim as fo
from fenicsmembranes.parametric_membrane import ParametricMembrane
# from fenics_wrinkle.materials.INH import INHMembrane
from fenics_wrinkle.materials.svk import SVKMembrane
from fenics_wrinkle.io import WrinklePlotter
from fenics_wrinkle.pressure import linear_volume_potential_split
import numpy as np
import scipy.interpolate


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
W = df.FunctionSpace(mesh, 'DG', 0)
ret_dofmap = W.dofmap()
p = df.Function(W)

PRESSURES = np.loadtxt("pressures.txt", dtype='float')
assert(len(PRESSURES) == num_global_cells)

temparray = np.zeros(num_global_cells)
for c, mesh_cell in enumerate(df.cells(mesh)):
    temparray[ret_dofmap.cell_dofs(mesh_cell.index())] = PRESSURES[c]

p.vector()[:] = temparray

f = df.File('PRESSURES.pvd')
f << p

# %% build list of cell midpoints
R = 10
MID_PTS = []
for c, mesh_cell in enumerate(df.cells(mesh)):
    MID_PTS.append([v for v in mesh_cell.midpoint()])
MID_PTS = np.array(MID_PTS)

# Find intersection of sphere with radius 10
NEW_MID_PTS = np.zeros((num_global_cells, 3))
for i, mp in enumerate(MID_PTS):
    scale_factor = R/np.sqrt(mp.dot(mp))
    corrected = mp*scale_factor
    NEW_MID_PTS[i] = corrected
    print(np.sqrt(corrected.dot(corrected)))

# %%
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
    # x = -xi; y = eta
    # a = -pi/2; b =- pi/2
    # PRM_COORDS[i] = [a+b-y, x-a+b]




#%%

PRM_COORDS = np.zeros((num_global_cells, 2))
for i, mp in enumerate(NEW_MID_PTS):
    X = mp[0]
    Y = mp[2]
    Z = mp[1]
    t = -1/(Z/R+1)
    PRM_COORDS[i] = [X/R*t, Y/R*t]

# %%
# import fenics_wrinkle.geometry.sphere as sphere
# geo = sphere.ParametricSphere(R)


def pinnedBCMembrane(membrane):
    bc = []
    bnd = df.CompiledSubDomain("on_boundary")
    bc.append(df.DirichletBC(membrane.V, Constant((0,0,0)), bnd))
    return bc
N = 60
prm_mesh = df.RectangleMesh(df.Point(-pi, 0), df.Point(0, pi), N, N)




#%%

import sympy as sp

class ParametricSphere():
    def __init__(self, radius):

        self.r = radius

        xi_1, xi_2, r = sp.symbols('x[0], x[1], r')

        h = xi_1**2 + xi_2**2
        X = r*2*xi_1/(h + 1)
        Y = r*2*xi_2/(h + 1)
        Z = -r*(h - 1)/(h + 1)

        gamma_sp = [X, Y, Z]

        ccode = lambda z: sp.printing.ccode(z)

        gamma = df.Expression([ccode(val) for val in gamma_sp],
                           r=radius,
                           degree=4)
        
        # G_1 = ∂X/xi^1
        Gsub1 = df.Expression([ccode(val.diff(xi_1)) for val in gamma_sp],
                           r=radius,
                           degree=4)

        # G_2 = ∂X/xi^2
        Gsub2 = df.Expression([ccode(val.diff(xi_2)) for val in gamma_sp],
                           r=radius,
                           degree=4)

        self.gamma = gamma
        self.Gsub1 = Gsub1
        self.Gsub2 = Gsub2

        
def pinnedBCMembrane(membrane):
    bc =[]
    bnd = df.CompiledSubDomain("on_boundary")
    bc.append(df.DirichletBC(membrane.V, Constant((0, 0, 0)), bnd))
    return bc

from mshr import *

domain = Circle(df.Point(0, 0), 1)
prm_mesh = generate_mesh(domain, 40)
geo = ParametricSphere(10)


V = df.FunctionSpace(prm_mesh, 'CG', 1)
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
        # x_ = x[0]
        # y = x[1]
        # a = -pi/2; b = pi/2
        # p0, p1 = [a+b-y, x_-a+b]
        #  self.f(p0, p1) # 
        values[0] =self.f(x[0], x[1])

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
        'inflation_solver': 'Custom Newton'}

mem = ParametricMembrane(input_dict)
mem.p_ext.assign(project(p_prn, V))

mem.io.write_fields()
mem.inflate(200)

mem.u.vector()[:] = 0


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


            self.energy = energy = SVKMembrane(u, self, lamb, mu, degree=2)
            prob.add_convex_term(Constant(t)*self.J_A*energy) # membrane.J_A



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




#%%


class LogSolver:
    def __init__(self, mem):
        self.mem =mem

    def solve(self, p, i=0, max_iter=100, tol=1e-4):
        mem = self.mem
        R = df.FunctionSpace(mem.mesh, "Real", 0)
        R3 = df.VectorFunctionSpace(mem.mesh, "Real", 0, dim=3)
        u = mem.u
        
        self.E_list = E_list = []
        E = 1
        itr = 0
        self.VOL = VOL = []
    
        u_old = df.Function(mem.V)
        u_old.assign(mem.u)
        # for j in range(itr):
        with df.Timer(f"Mosek Inflate Interval {i}, Iteration {itr}"):
            while E > tol:
                if itr > max_iter:
                    break
    
                prob = fo.MosekProblem("log-term")
                y, u_ = prob.add_var([R3, mem.V], cone=[fo.Exp(3), None], bc=[None, mem.bc], name=['y','_u'])
                prob.var[1] = mem.u 
                u = mem.u
                vol_lin_u, vol_lin_const = linear_volume_potential_split(mem)
    
                prob.add_eq_constraint(R, A=lambda mu: [mu * y[1] * dx(mem.mesh)], b=1, name='one')
                prob.add_obj_func([-(3*mem.gas.constant) * y[2] * dx(mem.mesh)])
    
                # y_0 = \int V_hat
                prob.add_eq_constraint(R, A=lambda mu: [mu * y[0] * dx(mem.mesh),
                                                    -mu*(Constant(1/3)*sum(vol_lin_u)*dx(mem.mesh))],
                                b=lambda mu: mu*(Constant(1/3)*sum(vol_lin_const)*dx(mem.mesh)), name='lamb')
    
    
                g3_bar = project(mem.gsub3, mem.V)
                prob.add_obj_func([None, dot(mem.p_ext*g3_bar,u)*dx(mem.mesh)])
    
                mem.energy = energy = SVKMembrane(u, mem, lamb, mu, degree=2)
                prob.add_convex_term(Constant(t)*mem.J_A*energy)
    
                io = WrinklePlotter(mem, energy)
                mem.io.add_plotter(io.plot, 'output/xdmf_write_interval', 0)
                prob.parameters["presolve"] = True
                prob.optimize()
                l1 = np.mean(df.project(mem.lambda1, mem.Vs).compute_vertex_values(mem.mesh))
                # E = l1 - l1_old
                E = df.errornorm(u, u_old)
                E_list.append(E)
                VOL.append(df.assemble(Constant(1/3)*(sum(vol_lin_u)+sum(vol_lin_const))*dx(mem.mesh))/3)
                print(f'E={E},  l1={l1}, itr={itr}')
                itr += 1
                # l1_old = l1
                u_old.assign(u)
    
                mem.io.write_fields()
        # return E_list, itr

solver = LogSolver(mem)

solver.solve(200, max_iter=10)