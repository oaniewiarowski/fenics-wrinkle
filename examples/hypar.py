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

bm = KannoIsotropic()

QUAD_DEGREE = 3
width = w = 10/df.sqrt(2)
height = h = 10/df.sqrt(2)
H = 5/df.sqrt(2)  # vertical displacement of the hypar
N = 30  # mesh resolution
c = 1/df.sqrt(2)  # in-plane stretching
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
import mshr

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
mesh.rotate(-45, 2, df.Point(b/2,b/2))
mesh.translate(df.Point((-b/2, d/2-bb-b/2)))

eps=1e-6
bot = df.CompiledSubDomain("(near(x[1], 0, eps) && on_boundary)", eps=eps)
top = df.CompiledSubDomain("(near(x[1], ymax, eps) && on_boundary)", ymax=(d-2*bb), eps=eps)
left = df.CompiledSubDomain("(near(x[0], 0, eps) && on_boundary)", eps=eps)
right = df.CompiledSubDomain("(near(x[0], xmax, eps) && on_boundary)", xmax=(d-2*bb), eps=eps)


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

df.File('output/mesh_func.pvd') << mesh_func
# mem.ds = df.Measure('ds', domain=mem.mesh, subdomain_data=mesh_func)


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
        'output_file_path': 'output/hypar',
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


io = WrinklePlotter(mem, energy)
io.s1_max =Constant(420)
mem.io.add_plotter(io.plot, 'output/xdmf_write_interval', 0)
mem.io.write_fields()
