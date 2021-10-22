#!/usr/bin/env python
# coding: utf-8

import dolfin as df
from dolfin import Constant
# importing fenics_optim after ParametricMembrane causes plotter to hang
import fenics_optim as fo
from fenics_wrinkle.parametric_membrane import ParametricMembrane

from fenics_wrinkle.materials.INH import INHMembrane
from fenics_wrinkle.materials.svk import SVKMembrane
from fenics_wrinkle.utils import eigenvalue
from fenics_wrinkle.io import WrinklePlotter

import numpy as np
import sympy as sp
import sympy.printing.ccode as ccode
import scipy.interpolate
import json
from matplotlib import rc_file
rc_file('../submission/journal_rc_file.rc')
import matplotlib.pyplot as plt

width = 200
height = 100
t = 0.025

E_ = 3500
nu = 0.499999
UX = df.Constant(0)
UY = df.Constant(0)

mu = E_/2/(1+nu)
lamb = E_*nu/(1+nu)/(1-2*nu)
# lamb = lamb_bar = 2*lamb*mu/ (lamb + 2*mu)


MAT = 'svk'
# MAT = 'inh'  # uncomment to run with inh

bottom = df.CompiledSubDomain("(near(x[1], -h/2) && on_boundary)", h=height)
top = df.CompiledSubDomain("(near(x[1], h/2) && on_boundary)", h=height)


class Geometry:
    def __init__(self):
        xi_1, xi_2, w, h = sp.symbols('x[0], x[1], w, h')
        gamma_sp = [xi_1, xi_2, 0*xi_1]

        self.gamma = df.Expression([ccode(val) for val in gamma_sp],
                                   w=width,
                                   h=height,
                                   degree=4)

        # Get the covariant tangent basis
        # G_1 = ∂X/xi^1 = [1, 0, 0]
        self.Gsub1 = df.Expression([ccode(val.diff(xi_1)) for val in gamma_sp],
                                   w=width,
                                   h=height,
                                   degree=4)

        # G_2 = ∂X/xi^2 = [0, 1, 0]
        self.Gsub2 = df.Expression([ccode(val.diff(xi_2)) for val in gamma_sp],
                                   w=width,
                                   h=height,
                                   degree=4)


def bcs(self):
    bc = [df.DirichletBC(self.V, Constant((0, 0, 0)), bottom),
          df.DirichletBC(self.V, Constant((UX, UY, 0)), top)]
    return bc


def shear_test(ux, uy, N, degree=2, diag='right', mat='svk'):
    UX.assign(ux)
    UY.assign(uy)
    geo = Geometry()
    mesh = df.RectangleMesh(df.Point(-width/2, -height/2),
                            df.Point(width/2, height/2),
                            2*N, N, diag)

    input_dict = {
            'mesh': mesh,
            'geometry': geo,
            'thickness': t,
            'output_file_path': f'results/convergence/shear_{ux}_{uy}_{2*N}x{N}',
            'pressure': 0,
            'Boundary Conditions': bcs}

    membrane = ParametricMembrane((input_dict))

    prob = fo.MosekProblem("No-compression membrane model")
    _ = prob.add_var(membrane.V, bc=membrane.bc)
    prob.var[0] = membrane.u   # replace
    u = membrane.u

    if mat == 'svk':
        energy = SVKMembrane(u, membrane, lamb, mu, degree=degree)
        prob.add_convex_term(Constant(t)*energy)
    if mat == 'inh':
        energy = INHMembrane(u, membrane, mu, degree=degree)
        prob.add_convex_term(Constant(t*mu/2)*membrane.J_A*energy)

    io = WrinklePlotter(membrane, energy)
    membrane.io.add_plotter(io.plot, 'output/xdmf_write_interval', 0)
    prob.parameters["presolve"] = True
    prob.optimize()
    membrane.io.write_fields()
    return membrane, energy
# %%

class Mansfield(df.UserExpression):
    def __init__(self, X, Y, E, **kwargs):
        super().__init__(**kwargs)
        self.degree = 5
        XY = list(zip(X.ravel(), Y.ravel()))
        self.f = scipy.interpolate.LinearNDInterpolator(XY, E.ravel(), fill_value=0)
        del X, Y, E

    def eval(self, values, x):
        values[0] = self.f(x[0], x[1])

    def value_shape(self):
        return ()


def convergence_test(u, v, path, num_levels=5, max_degree=3, diag='right'):
    X = np.load(f'mansfield/{path}/E_x.npy',
                allow_pickle=True).astype('float64')
    Y = np.load(f'mansfield/{path}/E_y.npy',
                allow_pickle=True).astype('float64')
    Emf = np.load(f'mansfield/{path}/E_vals.npy',
                  allow_pickle=True).astype('float64')

    mf = Mansfield(X, Y, Emf)

    h = {}  # discretization
    n_list = {}
    E = {}  # error measure

    # Iterate over quad degrees and mesh refinement levels
    degrees = range(2, max_degree + 1)
    for degree in degrees:
        n = 1  # coarsest mesh division
        h[degree] = []
        n_list[degree] = []
        E[degree] = []
        for i in range(num_levels):
            n_list[degree].append(n)
            with df.Timer(f"Shear test {u}, {v}, {n}, {degree}"):
                membrane, energy = shear_test(u, v, n, degree=degree, diag=diag, mat=MAT)
            h[degree].append(membrane.mesh.hmax())
            V1 = df.FunctionSpace(membrane.mesh, 'CG', 1)
            fcp = {"quadrature_degree": degree}
            E_el, _ = eigenvalue(energy.E_el)
            E1 = df.project(E_el, V1, form_compiler_parameters=fcp)

            L2_error = df.errornorm(mf, E1, norm_type='L2', degree_rise=3, mesh=membrane.mesh)
            E[degree].append(L2_error)
            print('(%d x %d) P%d mesh,  L2 error = %g' %
                  (2*n, n, degree, L2_error))
            n *= 2
    return h, n_list, E


results = {0: {'u': 0.6,
               'v': 0.2},
           1: {'u': 0.06,
               'v': 0.02}
           }
diag = 'right'
num_levels = 7
max_degree = 4
for i in range(0, len(results)):
    u = results[i]['u']
    v = results[i]['v']
    path = f'2000_u{u}_v{v}'
    h, n, E = convergence_test(u, v, path,
                               num_levels=num_levels,
                               max_degree=max_degree,
                               diag=diag)
    results[i]['h'] = h
    results[i]['n'] = n
    results[i]['E'] = E


with open(f'{MAT}_convergence_{diag}_{num_levels}.json', 'w') as fp:
    json.dump(results, fp,  indent=4)

df.list_timings(df.TimingClear.keep,
                [df.TimingType.wall, df.TimingType.system])
# %%
from mpltools import annotation

def plot_convergence(results):
    sizes = {'small': [3.25, 3],
             'large': [6, 3.25]}
    markers = ['o', 'x', '+']
    ls = ['dotted', 'dashed']
    for size in sizes.keys():
        fig, ax = plt.subplots(figsize=sizes[size])

        # colors = ['k', 'r']
        for c, case in enumerate(range(0, len(results))):
            for d, degree in enumerate(results[case]['E'].keys()):
                h = 1/np.array(results[case]['h'][degree])
                E = results[case]['E'][degree]
                ax.loglog(h, E,
                          linestyle=ls[c],
                          marker=markers[d],
                          ms=7,
                          lw=1,
                          label=f"$Q{degree}, u, v=({results[case]['u']},{results[case]['v']})$")

        annotation.slope_marker((.04,  .015), (-1, 1), invert=True, ax=ax)
        ax.legend(ncol=2)

        ax.set_xlabel(r'$log_{10}(1/h)$')
        ax.set_ylabel(r'$log_{10}(L^2 error) $')
        ax.set_xlabel(r'$1/h$')
        ax.set_ylabel(r'$L^2 error $')
        # ax.set_xlim(5e-3, 1)

        # from matplotlib.ticker import MaxNLocator
        # my_locator = MaxNLocator(4)
        # ax.xaxis.set_major_locator(my_locator)
        # ax.xaxis.set_minor_formatter(plt.NullFormatter())

        plt.tight_layout()

        import os
        if not os.path.exists('submission/figures/'):
            os.makedirs('submission/figures/')

        # plt.savefig(f'../submission/figures/shear_test_convergence_crossed_{size}.pdf', dpi=600)
        plt.savefig(f'../submission/figures/shear_convergence_{diag}_{num_levels}_{size}.pdf', dpi=600)
# res = {0: results[1]}
plot_convergence(results)


# %% Combined INH SVK
with open(f'svk_convergence_{diag}_{num_levels}.json', 'r') as fp:
    results_SVK = json.load(fp)
with open(f'inh_convergence_{diag}_{num_levels}.json', 'r') as fp:
    results_INH = json.load(fp)


from mpltools import annotation
def plot_convergence(results):
    sizes = {'small': [3.25, 3],
             'large': [6, 3.25]}
    markers = ['o', 'x', '+']
    ls = ['dotted', 'dashed']

    for size in sizes.keys():
        fig, ax = plt.subplots(figsize=sizes[size])

        # colors = ['k', 'r']
        RESULTS = [results_SVK, results_INH]
        for m, mat in enumerate(['SVK ', 'INH ']):
            results = RESULTS[m]
            for c, case in enumerate(range(0, len(results))):
                case = str(case)
                for d, degree in enumerate(results[case]['E'].keys()):
                    if int(degree) > 2:
                        break
                    h = 1/np.array(results[case]['h'][degree])
                    E = results[case]['E'][degree]
                    ax.loglog(h, E,
                              linestyle=ls[c],
                              marker=markers[m],
                              ms=7,
                              lw=1,
                              label= f'{mat}' + f"$  Q_{degree}, u, v=({results[case]['u']},{results[case]['v']})$")


        annotation.slope_marker((.04,  .015), (-1, 1), invert=True, ax=ax)
        ax.legend(ncol=2)

        ax.set_xlabel(r'$log_{10}(1/h)$')
        ax.set_ylabel(r'$log_{10}(L^2 error) $')
        ax.set_xlabel(r'$1/h$')
        ax.set_ylabel(r'$L^2 error $')
        # ax.set_xlim(5e-3, 1)

        # from matplotlib.ticker import MaxNLocator
        # my_locator = MaxNLocator(4)
        # ax.xaxis.set_major_locator(my_locator)
        # ax.xaxis.set_minor_formatter(plt.NullFormatter())

        plt.tight_layout()

        import os
        if not os.path.exists('submission/figures/'):
            os.makedirs('submission/figures/')

        # plt.savefig(f'../submission/figures/shear_test_convergence_crossed_{size}.pdf', dpi=600)
        plt.savefig(f'../submission/figures/shear_convergence_{diag}_{num_levels}_{size}.pdf', dpi=600)
# res = {0: results[1]}

diag = 'right'
num_levels = 7
max_degree = 4
plot_convergence(None)
