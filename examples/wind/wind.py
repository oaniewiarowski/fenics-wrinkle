#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 23:41:53 2021

@author: alexanderniewiarowski
"""
import os
import sys
import dolfin as df
from dolfin import Constant, project, dot
import ufl
import fenics_optim as fo
from fenics_wrinkle.parametric_membrane import ParametricMembrane
from fenics_wrinkle.materials.svk import SVKMembrane
from fenics_wrinkle.io import WrinklePlotter
from fenics_wrinkle.pressure import linear_volume_potential_split
from fenics_wrinkle.pressure import linear_volume_potential
from fenics_wrinkle.utils import eigenvalue
import numpy as np
import sympy as sp
import sympy.printing.ccode as ccode
import scipy.interpolate
from matplotlib import rc_file
rc_file('../submission/journal_rc_file.rc')
import matplotlib.pyplot as plt


# To add small radial displacement to simulate hoop prestress
HOOP_PRESTRESS = True
LOAD = 'symmetric'

R_f = 10  # m
t = 0.005  # 5 mm
p = 200  # 0.2 kPa
E = 0.5E9  # 0.5 GPa
nu = 0

mu = E/2/(1+nu)
lamb = E*nu/(1+nu)/(1-2*nu)
sigma_0 = p*R_f/2/(t)
R = E/(E+sigma_0)*R_f if HOOP_PRESTRESS else R_f


def pinnedBC(membrane):
    bc = []
    bnd = df.CompiledSubDomain("on_boundary")
    if HOOP_PRESTRESS:
        hoop = df.Expression(('(sigma_0/E)*x[0]', '(sigma_0/E)*x[1]', '0'),
                             sigma_0=sigma_0,
                             E=E,
                             degree=1)
        bc.append(df.DirichletBC(membrane.V, hoop, bnd))
    else:
        bc.append(df.DirichletBC(membrane.V, Constant((0, 0, 0)), bnd))
    return bc


class ParametricSphere:
    def __init__(self, radius, res=15):

        self.r = radius

        xi_1, xi_2, r = sp.symbols('x[0], x[1], r')
        # stereographic projection from unit disc to unit hemisphere
        h = xi_1**2 + xi_2**2
        X = 2*xi_1/(h + 1)
        Y = 2*xi_2/(h + 1)
        Z = -(h - 1)/(h + 1)

        gamma_sp = [r*X, r*Y, r*Z]
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

        from mshr import Circle, generate_mesh
        domain = Circle(df.Point(0, 0), 1)
        self.mesh = generate_mesh(domain, res)


def mosek_inflate(self, p, i=0, max_iter=100, tol=1e-4):
    def streamprinter(text):
        sys.stdout.write(text)
        sys.stdout.flush()
        
    dx = df.dx(self.mesh)
    u = self.u
    E_list = []
    E = 1
    itr = 0

    u_old = df.Function(self.V)
    u_old.assign(self.u)

    with df.Timer(f"Mosek Inflate Interval {i}, Iteration {itr}"):
        while E > tol:
            if itr > max_iter:
                break

            prob = fo.MosekProblem("No-compression membrane model")
            prob.streamprinter = streamprinter
            _ = prob.add_var(self.V, bc=self.bc)
            prob.var[0] = u

            self.energy = energy = SVKMembrane(u, self, lamb, mu, degree=2)
            prob.add_convex_term(Constant(t)*self.J_A*energy)

            U_air_list = linear_volume_potential(self, p)
            # print(type(U_air_list[0]))
            for dU in U_air_list:
                prob.add_obj_func(-Constant(p/3)*dU*dx(self.mesh))
            io = WrinklePlotter(self, energy)
            self.io.add_plotter(io.plot, 'output/xdmf_write_interval', 0)
            prob.parameters["presolve"] = True
            prob.optimize()
            l1 = np.mean(df.project(self.lambda1,
                                    self.Vs).compute_vertex_values(self.mesh))

            E = df.errornorm(u, u_old)
            E_list.append(E)
            print(f'E={E},  l1={l1}, itr={itr}')
            itr += 1
            # l1_old = l1
            u_old.assign(u)

    self.io.write_fields()
    return E_list, itr


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


PRM_COORDS = np.load('input_data/prm_coords.npy',
                     allow_pickle=True).astype('float64')

PRESSURES = np.load('input_data/pressures.npy',
                    allow_pickle=True).astype('float64')

p_prn = PressureInterpolator(PRM_COORDS, PRESSURES)

geo = ParametricSphere(R, res=24)

input_dict = {
        'mesh': geo.mesh,
        'geometry': geo,
        'thickness': t,
        'output_file_path': f'{LOAD}/wind',
        'pressure': p,
        'Boundary Conditions': pinnedBC,
        'inflation_solver': 'Custom Newton'}

mem = ParametricMembrane(input_dict)
mem.p_ext.assign(project(p_prn, df.FunctionSpace(mem.mesh, 'CG', 1)))

if LOAD == "asymmetric":
    x = df.SpatialCoordinate(mem.mesh)
    
    p1 = Constant(1121)
    p2 = Constant(-1209)
    p3 = Constant(-.4*p1)
    mem.p_ext.assign(project(df.conditional(df.gt(x[1],0),
                                            (1-x[1])*p2 + (x[1])*p3,
                                            mem.p_ext),
                             df.FunctionSpace(mem.mesh, 'CG', 1)))

# Write intial inflation stresses

mosek_inflate(mem, p, max_iter=4)
mem.gas.setup()


# %%
output_dir = f"{LOAD}/{LOAD}_mosek_output"

class LogSolver:
    def __init__(self, mem):
        self.mem = mem

    def solve(self, p, max_iter=100, tol=1e-4):
        mem = self.mem
        dx = df.dx(mem.mesh)
        R = df.FunctionSpace(mem.mesh, "Real", 0)
        R3 = df.VectorFunctionSpace(mem.mesh, "Real", 0, dim=3)
        u = mem.u

        self.E_list = E_list = []
        E = 1
        itr = 1
        self.VOL = VOL = []

        u_old = df.Function(mem.V)
        u_old.assign(mem.u)

        while E > tol:
            if itr > max_iter:
                break

            def streamprinter(text):
                sys.stdout.write(text)
                sys.stdout.flush()
                with open(f"{output_dir}/timings_{itr}.log", "a") as out:
                    out.write(text)

            prob = fo.MosekProblem("log-term")
            prob.streamprinter = streamprinter
            y, u_ = prob.add_var([R3, mem.V],
                                 cone=[fo.Exp(3), None],
                                 bc=[None, mem.bc],
                                 name=['y', '_u'])
            prob.var[1] = mem.u
            u = mem.u
            v_lin_u, v_lin_c = linear_volume_potential_split(mem)

            # y_1 = 1
            prob.add_eq_constraint(R,
                                   A=lambda mu: [mu * y[1] * dx],
                                   b=1,
                                   name='one')

            prob.add_obj_func([-(3*mem.gas.constant) * y[2] * dx])

            # y_0 = \int V_hat
            prob.add_eq_constraint(R,
                                   A = lambda mu: [mu * y[0] * dx,
                                                   -mu*(Constant(1/3)*sum(v_lin_u)*dx)],
                                   b = lambda mu: mu*(Constant(1/3)*sum(v_lin_c)*dx),
                                   name='lamb')

            g3_bar = project(mem.gsub3, mem.V)
            prob.add_obj_func([None, dot(mem.p_ext*g3_bar, u)*dx])

            mem.energy = energy = SVKMembrane(u, mem, lamb, mu, degree=2)
            prob.add_convex_term(Constant(t)*mem.J_A*energy)
            if itr == 1 :
                io = WrinklePlotter(mem, energy)
                mem.io.add_plotter(io.plot, 'output/xdmf_write_interval', 0)
            else:
                io.energy = energy
            prob.parameters["presolve"] = True

            with df.Timer(f"MOSEK iteration {itr}"):
                prob.optimize()

            with df.Timer(f"I/O iteration {itr}"):
                mem.io.write_fields()

            E = df.errornorm(u, u_old)
            E_list.append(E)
            V_hat = Constant(1/3)*(sum(v_lin_u) + sum(v_lin_c))*dx
            VOL.append(df.assemble(V_hat)/3)

            print(f'E={E},  itr={itr}')
            u_old.assign(u)
            itr += 1
            
        self.prob = prob
        self.energy = energy



if not os.path.exists(f'{output_dir}'):
    os.makedirs(f'{output_dir}')

import glob
files = glob.glob(f'{output_dir}/*', recursive=True)

for f in files:
    os.remove(f)

solver = LogSolver(mem)
with df.Timer("Simulation Total"):
    solver.solve(200, max_iter=15, tol=1e-4)


#%%

out_path = '../submission/figures/'
if not os.path.exists(out_path):
    os.makedirs(out_path)
from matplotlib.ticker import MaxNLocator

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=[6.5, 3])

# Plot volume and pressure
vols = np.array(solver.VOL)
pressures = float(mem.gas.constant)/vols

dim = np.arange(1, len(vols) + 1)

ax0.plot(dim, vols, label='Volume', lw=1, marker='.', c=colors[0])
ax0.axhline(mem.gas.V_0, c=colors[0], ls='-.', label='Initial inflated volume')
ax0.xaxis.set_major_locator(MaxNLocator(integer=True))
ax0.set_xlabel('Iterations')
ax0.set_ylabel('Enclosed volume')

ax01 = ax0.twinx() 
ax01.plot(dim, pressures, label='Pressure', lw=1, marker='x', c=colors[1])
ax01.axhline(float(mem.p_0), c=colors[1], ls='--', label='Initial pressure')
ax01.set_ylabel('Internal pressure')

handles,labels = [],[]
for ax in fig.axes:
    for h,l in zip(*ax.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)
plt.legend(handles,labels)

# Plot outer iterations
ax1.plot(dim, solver.E_list, label=r'$|u-\overline{u}|$', lw=1, marker='.')
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
ax1.set_yscale('log')

ax1.set_xlabel('Iterations')
ax1.set_ylabel('Error')
ax1.legend()

ax0.grid(True)
ax1.grid(True)
ax0.text(.5, 1.1, '(a)',
          horizontalalignment='center',
          verticalalignment='top',
          transform=ax0.transAxes)

ax1.text(.5, 1.1, '(b)',
          horizontalalignment='center',
          verticalalignment='top',
          transform=ax1.transAxes)

plt.tight_layout()
plt.savefig(out_path + f'{LOAD}wind_iterations.pdf', dpi=600)
#%%


time_table = df.timings(df.TimingClear.keep,
                        [df.TimingType.wall, df.TimingType.system])
with open(f"{LOAD}/{LOAD}_dolfin_timings.log", "w") as out:
    out.write(time_table.str(True))
    
df.list_timings(df.TimingClear.clear,
                [df.TimingType.wall, df.TimingType.system])