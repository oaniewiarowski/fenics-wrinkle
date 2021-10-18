#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 17:02:43 2021

@author: alexanderniewiarowski
"""

import fenics_optim as fo
import dolfin as df
from dolfin import project, DirichletBC, near, Constant
from fenicsmembranes.parametric_membrane import ParametricMembrane
from fenics_wrinkle.bm_data import KannoIsotropic
from fenics_wrinkle.materials.INH import INHMembrane
from fenics_wrinkle.io import WrinklePlotter
import mshr

bm = KannoIsotropic()

QUAD_DEGREE = 2
width = w = 10/df.sqrt(2)
height = h = 10/df.sqrt(2)
H = 5/df.sqrt(2)  # vertical displacement of the hypar
N = 30/2  # mesh resolution
c = 1/df.sqrt(2)  # in-plane stretching
POINT = False

if POINT:
    mesh = df.RectangleMesh(df.Point(0, 0), df.Point(width, height), N, N, 'crossed')
    
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

#%%


b = .25/df.sqrt(2)
bb = b/df.sqrt(2)
d = df.sqrt(w**2 + h**2)
pts = [[b, 0],
       [w-b, 0],
       [w,b],
       [w,h-b],
       [w-b,h],
       [b,h],
       [0,h-b],
       [0,b]]
domain = mshr.Polygon([df.Point(pt) for pt in pts])
mesh = mshr.generate_mesh(domain,N)
mesh.translate(df.Point((-w/2, -h/2)))
mesh.rotate(-45, 2, df.Point(0,0))

eps=1e-6
extent = (d-2*bb)/2
bot = df.CompiledSubDomain("(near(x[1], -ymax, eps) && on_boundary)", ymax=extent, eps=eps)
top = df.CompiledSubDomain("(near(x[1], ymax, eps) && on_boundary)", ymax=extent, eps=eps)
left = df.CompiledSubDomain("(near(x[0], -xmax, eps) && on_boundary)", xmax=extent, eps=eps)
right = df.CompiledSubDomain("(near(x[0], xmax, eps) && on_boundary)", xmax=extent, eps=eps)


def bc(membrane):
    bc = [DirichletBC(membrane.V, Constant((c, 0, 0)), right),
          DirichletBC(membrane.V, Constant((-c, 0, 0)), left),
          DirichletBC(membrane.V, Constant((0, -c, H)), bot),
          DirichletBC(membrane.V, Constant((0, c, H)), top)]
    return bc



mesh_func = df.MeshFunction("size_t", mesh, mesh.topology().dim()-1)

bot.mark(mesh_func, 1)
top.mark(mesh_func, 2)
left.mark(mesh_func, 3)
right.mark(mesh_func, 4)

df.File(f'results/mesh_boundaries.pvd') << mesh_func

#%%
class Geometry:
    def __init__(self):
        V = df.VectorFunctionSpace(mesh, 'CG', 1, dim=3)
        self.gamma = project(df.Expression(('x[0]', 'x[1]', 0), degree=1), V)
        self.Gsub1 = project(Constant((1, 0, 0)), V)
        self.Gsub2 = project(Constant((0, 1, 0)), V)


geo = Geometry()

input_dict = {
        'mesh': mesh,
        'geometry': geo,
        'thickness': bm.t,
        'material': 'Incompressible NeoHookean',
        'mu': bm.mu,
        'cylindrical': False,
        'output_file_path': f'results/hypar',
        'pressure': 0,
        'Boundary Conditions': bc}

mem = membrane = ParametricMembrane((input_dict))
mem.io.write_fields()

prob = fo.MosekProblem("No-compression membrane model")
u__ = prob.add_var(membrane.V, bc=membrane.bc)
prob.var[0] = membrane.u   # replace
u = membrane.u

energy = INHMembrane(u, mem, degree=QUAD_DEGREE)
prob.add_convex_term(bm.t*bm.mu/2*energy)
io = WrinklePlotter(mem, energy)
mem.io.add_plotter(io.plot, 'output/xdmf_write_interval', 0)
prob.parameters["presolve"] = True

prob.optimize()

#%%
for eps in [.005, .01, .02, .03, .04, 0.05,  .1]:
    io = WrinklePlotter(mem, energy)
    io.s1_max = Constant(420)
    io.thresh = Constant(eps)
    mem.io.add_plotter(io.plot, 'output/xdmf_write_interval', 0)
    mem.io.write_fields()


