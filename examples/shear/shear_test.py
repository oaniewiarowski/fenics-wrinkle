#!/usr/bin/env python
# coding: utf-8

import dolfin as df
from dolfin import DirichletBC, Expression, Constant
import ufl
# importing fenics_optim after ParametricMembrane causes plotter to hang
import fenics_optim as fo
from fenicsmembranes.parametric_membrane import ParametricMembrane

from fenics_wrinkle.materials.INH import INHMembrane
from fenics_wrinkle.materials.svk import SVKMembrane
from fenics_wrinkle.utils import eigenvalue
from fenics_wrinkle.io import WrinklePlotter
from fenics_wrinkle.bm_data import Mosler

import numpy as np
import sympy as sp
import sympy.printing.ccode as ccode
import scipy.interpolate

from matplotlib import rc_file
rc_file('../submission/journal_rc_file.rc')
import matplotlib.pyplot as plt

width = 200
height = 100
t = 0.025

E_ = 3500
nu = 0.31
UX = df.Constant(0)
UY = df.Constant(0)

mu = E_/2/(1+nu)
lamb = E_*nu/(1+nu)/(1-2*nu)

bm = Mosler()
mu = bm.mu
t = bm.t

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


def shear_test(ux, uy, N, degree=2):
    UX.assign(ux)
    UY.assign(uy)
    geo = Geometry()
    mesh = df.RectangleMesh(df.Point(-width/2, -height/2),
                            df.Point(width/2, height/2),
                            2*N, N, 'left')

    input_dict = {
            'mesh': mesh,
            'geometry': geo,
            'thickness': t,
            'material': 'Incompressible NeoHookean',
            'mu': mu,
            'cylindrical': True,
            'output_file_path': 'conic_INH',
            'pressure': 0,
            'Boundary Conditions': bcs}

    membrane = ParametricMembrane((input_dict))

    prob = fo.MosekProblem("No-compression membrane model")
    _ = prob.add_var(membrane.V, bc=membrane.bc)
    prob.var[0] = membrane.u   # replace
    u = membrane.u

    energy = INHMembrane(u, membrane, degree=degree)
    prob.add_convex_term(Constant(t*mu/2)*membrane.J_A*energy)
    # energy = SVKMembrane(u, mem, lamb, mu, degree=2)
    # prob.add_convex_term(t*mem.J_A*energy)

    io = WrinklePlotter(membrane, energy)
    membrane.io.add_plotter(io.plot, 'output/xdmf_write_interval', 0)
    prob.parameters["presolve"] = True
    prob.optimize()
    membrane.io.write_fields()
    return membrane, energy


class Mansfield(df.UserExpression):
    def __init__(self, X, Y, E, degree=3, **kwargs):
        super().__init__(**kwargs)
        self.degree = 5
        XY = list(zip(X.ravel(), Y.ravel()))
        self.f = scipy.interpolate.LinearNDInterpolator(XY, E.ravel(),
                                                        fill_value=0)
        del X, Y, E

    def eval(self, values, x):
        values[0] = self.f(x[0], x[1])

    def value_shape(self):
        return ()

def convergence_test(u, v, path, num_levels=5, max_degree=3):
    X = np.load(f'mansfield/{path}/E_x.npy',
                allow_pickle=True).astype('float64')
    Y = np.load(f'mansfield/{path}/E_y.npy',
                allow_pickle=True).astype('float64')
    Emf = np.load(f'mansfield/{path}/E_vals.npy',
                  allow_pickle=True).astype('float64')

    mf = Mansfield(X, Y, Emf)

    h = {}  # discretization parameter: h[degree][level]
    E = {}  # error measure(s): E[degree][level]

    # Iterate over quad degrees and mesh refinement levels
    degrees = range(2, max_degree + 1)
    for degree in degrees:
        n = 1  # coarsest mesh division
        h[degree] = []
        E[degree] = []
        for i in range(num_levels):
            h[degree].append(1.0 / n)
            membrane, energy = shear_test(u, v, n, degree=degree)
            V = df.FunctionSpace(membrane.mesh, 'CG', 1)
            fcp = {"quadrature_degree": degree}
            E1, _ = eigenvalue(energy.E_el)
            E1 = df.project(E1, V, form_compiler_parameters=fcp)
            L2_error = df.errornorm(mf, E1, norm_type='L2', degree_rise=3)
            E[degree].append(L2_error)
            print('(%d x %d) P%d mesh,  L2 error = %g' %
              (2*n, n, degree, L2_error))
            n *= 2
    return h, E



h, E = convergence_test(0.06, 0.02, '2000_u0.06_v0.02', num_levels=6, max_degree=4)
#%%
import json
from mpltools import annotation
def plot_convergence(h, E):

    fig, ax = plt.subplots(figsize=[3.25, 3])
    for degree in h.keys():
        ax.loglog(h[degree], E[degree], '.-', label=f'degree={degree}')

    annotation.slope_marker((.4,  .004), (1, 1), invert=False, ax=ax)
    ax.legend()
    ax.set_xlabel(r'$log_{10}(1/h)$')
    ax.set_ylabel(r'$log_{10}(L^2 error) $')

    # from matplotlib.ticker import MaxNLocator
    # my_locator = MaxNLocator(4)
    # ax.xaxis.set_major_locator(my_locator)
    # ax.xaxis.set_minor_formatter(plt.NullFormatter())

    plt.tight_layout()


    import os
    if not os.path.exists('submission/figures/'):
        os.makedirs('submission/figures/')

    plt.savefig('./submission/figures/shear_test_convergence.pdf', dpi=600)

plot_convergence(h, E)
