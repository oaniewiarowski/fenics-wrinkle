#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 22:06:55 2021

@author: alexanderniewiarowski
"""
import dolfin as df
from dolfin import Constant, near, project, det, sqrt, dx
import fenics_optim as fo
from fenicsmembranes.parametric_membrane import ParametricMembrane
import matplotlib.pyplot as plt
import numpy as np
from fenics_wrinkle.bm_data import KannoIsotropic
from fenics_wrinkle.materials.INH import INHMembrane

from fenics_wrinkle.io import WrinklePlotter
bm = KannoIsotropic()

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
        self.gamma = project(df.Expression(('x[0]', 'x[1]', 0), degree=1), df.VectorFunctionSpace(mesh, 'CG', 1, dim=3))
        self.Gsub1 = project(Constant((1, 0, 0)), df.VectorFunctionSpace(mesh, 'CG', 1, dim=3))
        self.Gsub2 = project(Constant((0, 1, 0)), df.VectorFunctionSpace(mesh, 'CG', 1, dim=3))


UX = Constant(0)
UY = Constant(0)

def bc(mem):
    bc = [df.DirichletBC(mem.V.sub(0), Constant(0), left),
          df.DirichletBC(mem.V.sub(0), UX, right),
          df.DirichletBC(mem.V.sub(1), Constant(0), bottom),
          df.DirichletBC(mem.V.sub(1), UY, top),
          df.DirichletBC(mem.V.sub(2),  Constant(0), bnd)]

    # c = -5
    # bc = [DirichletBC(mem.V.sub(1), Constant(c), bottom),
    #       DirichletBC(mem.V.sub(0), Constant(c), left),
    #       DirichletBC(mem.V.sub(0), Constant(-c), right),
    #       DirichletBC(mem.V.sub(1), Constant(-c), top) ]
    return bc



def bc_uniaxial(mem):
    bc = [df.DirichletBC(mem.V, Constant((0,0,0)), left),
          df.DirichletBC(mem.V, Constant((UX,0,0)), right),


          df.DirichletBC(mem.V.sub(2),  Constant(0), bnd)]

    # c = -5
    # bc = [DirichletBC(mem.V.sub(1), Constant(c), bottom),
    #       DirichletBC(mem.V.sub(0), Constant(c), left),
    #       DirichletBC(mem.V.sub(0), Constant(-c), right),
    #       DirichletBC(mem.V.sub(1), Constant(-c), top) ]
    return bc


input_dict = {'mesh': mesh,
              'geometry': Geometry(),
              'thickness': bm.t,
              'material': 'Incompressible NeoHookean',
              'mu': bm.mu,
              'cylindrical': True,
              'output_file_path': 'results/planar_strip',
              'pressure': 0,
              'Boundary Conditions': bc}

middle = df.CompiledSubDomain("near(x[0], 100)")
markers = df.MeshFunction('size_t', mesh, mesh.geometric_dimension()-1)
markers.set_all(0)

middle.mark(markers, 1)
df.File('sub.pvd') << markers
dS = df.Measure('dS', domain=mesh, subdomain_data=markers)

mem = membrane = ParametricMembrane(input_dict)

t_j = []
t_J = []
t_el = []
DS = []
Ja = []
Ja_el = []
LENGTH = []
ELASTIC_LENGTH = []

uys = np.concatenate((np.zeros(2), np.linspace(0, 1, 11), np.array([1.5,2])))
uys = np.concatenate((np.zeros(1), np.linspace(0, .85, 4), np.array([1,2,3])))
for i, uy in enumerate(uys):
    if i == 0:
        print("slack")
    if i == 1:
        # mem.bc = bc_uniaxial(mem)
        UX.assign(3)
    # if i == 2:
    #     # release top
    #     mem.bc = bc(mem)
    # if i > 2:
    #     mem.bc = bc(mem)



    print('UX, UY:', float(UX), float(UY))
    UY.assign(-uy)
    prob = fo.MosekProblem("No-compression membrane model")
    u__ = prob.add_var(mem.V, bc=mem.bc)
    prob.var[0] = mem.u   # replace
    u = mem.u

    energy = INHMembrane(u, mem, degree=5)

    prob.add_convex_term(bm.t*bm.mu/2*energy)
    prob.parameters["presolve"] = True
    prob.optimize()

    io = WrinklePlotter(mem, energy)
    mem.io.add_plotter(io.plot, 'output/xdmf_write_interval', 0)
    mem.io.write_fields()

    mf = df.MeshFunction('size_t', mesh, 2, mesh.domains())
    zero = Constant(0)*dx(domain=mesh, subdomain_data=mf)

    dS_el = df.Measure('dS', domain=mesh, subdomain_data=markers, metadata=io.fcp)
    t_elastic = df.project(bm.t*sqrt((det(mem.C_0)/det(energy.C_n_el))),io.Vs, form_compiler_parameters=io.fcp)
    t_el.append(df.assemble(t_elastic*dS_el(1), form_compiler_parameters=io.fcp))

    t_ji = bm.t*sqrt((det(mem.C_0)('-')/det(mem.C_n)('-')))*mem.j_a('-')*dS(1) + zero
    t_j.append(df.assemble(t_ji))

    # t J
    t_Ji = bm.t*sqrt((det(mem.C_0)('-')/det(mem.C_n)('-')))*dS(1) + zero
    t_J.append(df.assemble(t_Ji))

    DS.append(df.assemble(Constant(1)*dS(1)))
    Ja.append(df.assemble(mem.j_a('-')*dS(1)))
    ja_el = df.project(sqrt(det(energy.C_n_el)), io.Vs, form_compiler_parameters=io.fcp)
    Ja_el.append(df.assemble(ja_el('-')*dS(1), form_compiler_parameters=io.fcp))
    
    LENGTH.append(df.assemble(df.dot(df.dot(mem.F, mem.Gsub2),mem.Gsub2)('-')*dS(1)))
    elastic_length = df.project(sqrt(df.dot(df.dot(energy.C_n_el, df.as_vector([0,1])),df.as_vector([0,1]))), io.Vs, form_compiler_parameters=io.fcp)
    ELASTIC_LENGTH.append(df.assemble(elastic_length('-')*dS(1), form_compiler_parameters=io.fcp))
#%%


from matplotlib import rc_file
import os
rc_file('../submission/journal_rc_file.rc')
#plt.style.use('seaborn-whitegrid')

out_path = '../submission/figures/'
if not os.path.exists(out_path):
    os.makedirs(out_path)


# For convenience
labels = {
        'KS': r'$J = KS\left( (\lambda_1 - \bar{\lambda})^2\right)$',
        'abs': r'$J = \int ( \lambda_1 - \bar{\lambda} )^2 d\xi$',
        'lsq': r'$\hat{J} = \int ( \lambda_1 - \bar{\lambda} )^2 d\xi$'
        }
#%%
if __name__ == "__main__":

    # csv file saved in Paraview. File > Save Data
    fname = 'results/planar_strip'

    fig, ax = plt.subplots(figsize=[6.5,3])
    alpha = (100-uys)/100
    ax.plot(uys, t_j, '-',
            label='Initial cross section') # + r'$\int h J_a dS = \int H dS$')

    ax.plot(uys, np.array(LENGTH)*t_J/100, 'x-.', ms=10,
            label='Reference cross section') # + r'$\int H\sqrt{\frac{C_0}{C_n}} dS $')
    # j =0
    # for x, y in zip(uys, t_J):
    #     ax.text(x+j/100, y, str(j), color="orange", fontsize=12)
    #     j+=1
    ax.plot(uys, np.array(ELASTIC_LENGTH)*t_el/100, '.--', ms=10,
            label='Elastic cross section')# + r'$\int H\sqrt{\frac{C_0}{C_n^e}} dS$')
    # j =0
    # for x, y in zip(uys, t_el):
    #     ax.text(x+j/50, y, str(j), color="green", fontsize=12)
    #     j+=1

    ax.text(uys[0]-.05, t_el[0], 'A', horizontalalignment='center', verticalalignment='top')
    ax.text(uys[1]-.05, t_el[1], 'B', horizontalalignment='center', verticalalignment='top')
    # ax.text(uys[2]-.05, t_el[2], 'C', horizontalalignment='center', verticalalignment='top')
    # ax.text(uys[3]-.05, t_el[3], 'D', horizontalalignment='center', verticalalignment='top')


    # ax.text(uys[0]+.05, t_J[0], 'A', horizontalalignment='center', verticalalignment='top')
    # ax.text(uys[1]+.05, t_J[1], 'B', horizontalalignment='center', verticalalignment='top')
    # ax.text(uys[2]+.05, t_J[2], 'C', horizontalalignment='center', verticalalignment='top')
    # ax.text(uys[3]+.05, t_J[3], 'D', horizontalalignment='center', verticalalignment='top')
    ax.set_xlabel('Tranverse displacement '+r'$\Delta u_y$')
    ax.set_ylabel('Cross sectional area '+ r'$ (mm^2)$')
    ax.legend()

    ax.grid(True)

    plt.tight_layout()
    plt.savefig(out_path+f'planar_strip_thickness_corrected.pdf', dpi=600)

# alpha = (bm.height - uys)/bm.height
# ax.plot(uys, t_J*alpha, 'o-',
#         label='corrected cross sectional area:' + r'$ \alpha \int h dS, \alpha=(height-\Delta u_y)/height$')

    import json
    results = {"intial": t_j,
               "ref": t_J,
               "elastic": t_el
               }
    with open("results/thicknesses_corrected.json", "w") as outfile:
        json.dump(results, outfile)
