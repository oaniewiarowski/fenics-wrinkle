#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 22:06:55 2021

@author: alexanderniewiarowski
"""

import dolfin as df
from dolfin import Constant, near, project, det, sqrt, assemble
import fenics_optim as fo
from fenics_wrinkle.parametric_membrane import ParametricMembrane
import matplotlib.pyplot as plt
import numpy as np
from fenics_wrinkle.bm_data import KannoIsotropic
from fenics_wrinkle.materials.INH import INHMembrane
from fenics_wrinkle.io import WrinklePlotter
from matplotlib import rc_file
import os
import json
rc_file('../submission/journal_rc_file.rc')

bm = KannoIsotropic()
t = bm.t

N = 15
mesh = df.RectangleMesh(df.Point(0, 0), df.Point(bm.width, bm.height), 2*N, N)

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
u_y = df.Expression(('x[1]*(-10/100)'), degree=1)


class Geometry:
    def __init__(self):
        self.gamma = project(df.Expression(('x[0]', 'x[1]', 0), degree=1),
                             df.VectorFunctionSpace(mesh, 'CG', 1, dim=3))
        self.Gsub1 = project(Constant((1, 0, 0)),
                             df.VectorFunctionSpace(mesh, 'CG', 1, dim=3))
        self.Gsub2 = project(Constant((0, 1, 0)),
                             df.VectorFunctionSpace(mesh, 'CG', 1, dim=3))

UX = Constant(0)
UY = Constant(0)

def bc(mem):
    bc = [df.DirichletBC(mem.V.sub(0), Constant(0), left),
          df.DirichletBC(mem.V.sub(0), UX, right),
          df.DirichletBC(mem.V.sub(1), Constant(0), bottom),
          df.DirichletBC(mem.V.sub(1), UY, top),
          df.DirichletBC(mem.V.sub(2), Constant(0), bnd)]
    return bc


T_0 = []
T_REF = []
T_ELASTIC = []

WIDTH = []
ELASTIC_WIDTH = []
h_1 = bm.t*(200/203)
h_crit = 0.5*(bm.t+h_1)
u_y_crit = bm.height - 500/203/h_crit
uys = np.concatenate((np.zeros(1),
                      np.linspace(0, 1, 11),
                      np.array([1.5, 2]),
                      np.array([u_y_crit])))
uys.sort()
# uys = np.concatenate((np.zeros(1), np.linspace(0, .85, 4), np.array([1,2,3])))


input_dict = {'mesh': mesh,
              'geometry': Geometry(),
              'thickness': bm.t,
              'output_file_path': 'results/planar_strip',
              'pressure': 0,
              'Boundary Conditions': bc}

middle = df.CompiledSubDomain("near(x[0], 100)")
markers = df.MeshFunction('size_t', mesh, mesh.geometric_dimension()-1)
markers.set_all(0)

middle.mark(markers, 1)
df.File('sub.pvd') << markers
dS = df.Measure('dS', domain=mesh, subdomain_data=markers)
DS = assemble(Constant(1)*dS(1))

mem = membrane = ParametricMembrane(input_dict)
width_n = df.dot(df.dot(mem.F, mem.Gsub2), mem.Gsub2)('-')*dS(1)


for i, uy in enumerate(uys):
    if i == 0:
        print("slack")
    if i == 1:
        UX.assign(3)

    print('UX, UY:', float(UX), float(UY))
    UY.assign(-uy)
    prob = fo.MosekProblem("No-compression membrane model")
    u__ = prob.add_var(mem.V, bc=mem.bc)
    prob.var[0] = mem.u   # replace
    u = mem.u

    energy = INHMembrane(u, mem, bm.mu, degree=3)
    prob.add_convex_term(bm.t*bm.mu/2*energy)
    prob.parameters["presolve"] = True
    prob.optimize()

    io = WrinklePlotter(mem, energy)
    mem.io.add_plotter(io.plot, 'output/xdmf_write_interval', 0)
    mem.io.write_fields()

    dS_el = df.Measure('dS', domain=mesh, subdomain_data=markers, metadata=io.fcp)

    # initial thickness
    t_0 = t*(sqrt(det(mem.C_0)/det(mem.C_n))*mem.j_a)('-')*dS(1)
    T_0.append(df.assemble(t_0)/DS)

    # elastic thickness
    t_elastic = df.project(t*sqrt((det(mem.C_0)/det(energy.C_n_el))),
                           io.Vs, form_compiler_parameters=io.fcp)
    T_ELASTIC.append(assemble(t_elastic*dS_el(1),
                              form_compiler_parameters=io.fcp)/DS)

    # reference thickness
    t_ref_i = t*sqrt((det(mem.C_0)('-')/det(mem.C_n)('-')))*dS(1)
    T_REF.append(assemble(t_ref_i)/DS)

    # reference width
    WIDTH.append(assemble(width_n))

    # elastic width (fictious)
    j_hat = df.as_vector([0, 1])
    elastic_length = df.project(sqrt(df.dot(df.dot(energy.C_n_el, j_hat), j_hat)),
                                io.Vs, form_compiler_parameters=io.fcp)
    ELASTIC_WIDTH.append(assemble(elastic_length('-')*dS(1),
                                  form_compiler_parameters=io.fcp))

results = {"T_0": T_0,
           "T_REF": T_REF,
           "T_ELASTIC": T_ELASTIC,
           "WIDTH": WIDTH,
           "ELASTIC_WIDTH": ELASTIC_WIDTH,
           "uys": list(uys),
           "DS": DS,
           }
with open("results/thicknesses.json", "w") as outfile:
    json.dump(results, outfile)

# %%


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_json("./results/thicknesses.json")
    DS = 100
    T_0 = df.T_0
    T_REF = df.T_REF
    T_ELASTIC = df.T_ELASTIC
    WIDTH = df.WIDTH
    ELASTIC_WIDTH = df.ELASTIC_WIDTH
    uys = df.uys
    out_path = '../submission/figures/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # csv file saved in Paraview. File > Save Data
    fname = 'results/planar_strip'

    fig, axs = plt.subplots(3, 1, figsize=[6.5, 6])
    lw = 1
    # thickness plot
    ax = axs[0]
    ax.plot(uys, np.array(T_0), '-', lw=lw, label=r'$H$')
    ax.plot(uys, np.array(T_REF), 'x-.', ms=7, lw=lw, label=r'$h_{ref}$')
    ax.plot(uys, np.array(T_ELASTIC), '.--', ms=7, lw=lw, label=r'$h_{elastic}$')

    ax.text(uys[0]-.05, T_ELASTIC[0], '0', horizontalalignment='center', verticalalignment='top')
    ax.text(uys[1]-.05, T_ELASTIC[1], '1', horizontalalignment='center', verticalalignment='top')

    ax.axhline(h_1, label=r'$h_1$', linestyle='dashdot',lw=1, c='k', zorder=1)
    ax.axhline(h_crit, label=r'$h_{crit}$', linestyle='dotted', lw=1, c='k', zorder=1)
    ax.axvline(u_y_crit, label=r'$u_{y,crit}$', linestyle='--', lw=1, c='k', zorder=1)
    ax.set_ylabel('Thickness ' + r'$ (mm)$')
    ax.legend(loc='best', bbox_to_anchor=(0.05, 0.2, 0.5, 0.5))
    ax.grid(True)

    # Width plot
    ax = axs[1]
    ax.plot(uys, DS+uys*0, '-', lw=lw, label=r'$w_0$')
    ax.plot(uys, np.array(WIDTH), 'x-.', ms=7, lw=lw, label=r'$w_{ref}$')
    ax.plot(uys, np.array(ELASTIC_WIDTH), '.--', ms=7, lw=lw, label=r'$w_{elastic}$')

    ax.text(uys[0]-.05, WIDTH[0]-0.2, '0, 1', horizontalalignment='left', verticalalignment='top')
    # ax.text(uys[1]+.05, WIDTH[1], '1', horizontalalignment='center', verticalalignment='top')
    ax.axvline(u_y_crit, label=r'$u_{y,crit}$', linestyle='--', lw=1, c='k', zorder=1)
    ax.set_ylabel('Width ' + r'$ (mm)$')
    ax.legend()
    ax.grid(True)

    # Area plot
    ax = axs[2]
    ax.plot(uys, np.array(T_0)*DS, '-', lw=lw, label=r'$A_0$')
    ax.plot(uys, np.array(WIDTH)*np.array(T_REF),
            'x-.', ms=7, lw=lw, label=r'$A_{ref}$')
    ax.plot(uys, np.array(ELASTIC_WIDTH)*np.array(T_ELASTIC),
            '.--', ms=7, lw=lw, label=r'$A_{elastic}$')
    ax.text(uys[0]-.05, T_0[0]*DS, '0', horizontalalignment='center', verticalalignment='top')
    ax.text(uys[1]-.05, T_REF[1]*DS, '1', horizontalalignment='center', verticalalignment='top')
    ax.axvline(u_y_crit, label=r'$u_{y,crit}$', linestyle='--', lw=1, c='k', zorder=1)
    ax.set_xlabel('Tranverse displacement ' + r'$u_y$' + ' ' + r'$(mm)$')
    ax.set_ylabel('Area ' + r'$ (mm^2)$')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(out_path+'planar_strip_3in1.pdf', dpi=600)


