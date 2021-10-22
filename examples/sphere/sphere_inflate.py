#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 15:40:57 2021

@author: alexanderniewiarowski
"""
import fenics_optim as fo
import dolfin as df
from dolfin import dot, cross, sqrt, det, project, dx, Constant, FunctionSpace
import numpy as np
import sympy as sp
import json
from fenics_wrinkle.parametric_membrane import ParametricMembrane
from fenics_wrinkle.utils import expand_ufl, eigenvalue
import fenics_wrinkle.geometry.sphere as sphere
from fenics_wrinkle.materials.INH import INHMembrane
from fenics_wrinkle.io import WrinklePlotter

from matplotlib import rc_file
rc_file('../submission/journal_rc_file.rc')
import matplotlib.pyplot as plt

INTERVALS = 5
R = 1
mu = 500
T = 0.01


def inflation_test(membrane):
    '''
    plane stress inflation of incompressible neohookean sphere
    '''

    mem = membrane
    thickness = float(mem.thickness)

    # set up arrays for quantities of interest
    num_vertex = mem.mesh.num_vertices()

    # 2D arrays (element wise results)
    computed_stretches = np.zeros((INTERVALS, num_vertex))  # computed stretch
    computed = np.zeros((INTERVALS, num_vertex))  # computed rhs (1 - 1/lambda^4)

    # 1D arrays
    volumes = np.linspace(1, np.sqrt(7), INTERVALS)  # desired volume ratios, limit pt at
    pressures = (2*mu*T/R)*((1/volumes)**(1/3) - (1/volumes)**(7/3))
    pressures[0] = 0.1

    # pr2muH = pressures*R/(2*mu*thickness)  # Analytical 2pR/muT

    # For plotting
    mean_computed = np.zeros(INTERVALS)
    mean_computed_stretch = np.zeros(INTERVALS)
    mean_computed_s1 = np.zeros(INTERVALS)
    mean_computed_s2 = np.zeros(INTERVALS)
    E_list = []
    itr_list = []

    mem.u.vector()[:] = 0  # make sure initial stretch is zero

    # Compute the principal stretches lambda_1 r/R
    l1 = mem.lambda1

    # Stresses
    semi_analytical_stress = np.zeros((INTERVALS, num_vertex))
    computed_stress_s1 = np.zeros((INTERVALS, num_vertex))
    computed_stress_s2 = np.zeros((INTERVALS, num_vertex))

    for i, inc_pressure in enumerate(pressures):
        with df.Timer(f"Mosek Inflate Interval {i}"):
            E, itr = mosek_inflate(mem, inc_pressure, i=i)
            energy = mem.energy
            fcp = {"quadrature_degree": energy.degree}

        E_list.append(E)
        itr_list.append(itr)
        # Element-wise stretches for current pressure
        computed_stretches[i] = df.project(l1, mem.Vs).compute_vertex_values(mem.mesh)

        # Next compute analytical plane strain result
        computed[i] = 1/computed_stretches[i] - pow(computed_stretches[i], -7)

        # Save the average stretch for plotting purposes
        mean_computed_stretch[i] = np.mean(computed_stretches[i])
        mean_computed[i] = np.mean(computed[i, :])

        # Stresses
        J = sqrt(det(mem.C_n)/det(mem.C_0))
        t = Constant(T)/J
        stress = Constant(mu*T)*(Constant(1)-mem.lambda1**-6)/t
        Vs = FunctionSpace(mem.mesh, 'CG', 1)
        semi_analytical_stress[i] = project(stress, mem.Vs).compute_vertex_values(mem.mesh)
        s1, s2 = eigenvalue(energy.get_cauchy_stress())
        computed_stress_s1[i] = project(s1, Vs,
                                        form_compiler_parameters=fcp).compute_vertex_values(mem.mesh)
        computed_stress_s2[i] = project(s2, Vs,
                                        form_compiler_parameters=fcp).compute_vertex_values(mem.mesh)
        mean_computed_s1[i] = np.mean(computed_stress_s1[i, :])
        mean_computed_s2[i] = np.mean(computed_stress_s2[i, :])

    results = {'mean_computed_stretch': mean_computed_stretch,
               'mean_computed': mean_computed,
               'mean_computed_s1': mean_computed_s1,
               'mean_computed_s2': mean_computed_s2,
               'pressures': pressures,
               'error': np.array(E_list),
               'itrs': np.array(itr_list)}
    return results


def plot_results(mean_computed_stretch,
                 mean_computed,
                 mean_computed_s1,
                 mean_computed_s2,
                 pressures,
                 **kwargs):

    fig, axs = plt.subplots(1, 2, figsize=[6.5, 3])

    # Stretches - Analytical relationship
    ax = axs[0]
    x = np.linspace(min(mean_computed_stretch),
                    max(mean_computed_stretch), 500)
    y = 1/x - pow(x, -7)
    ax.plot(x, y,
            c='k',
            lw=0.5,
            label=r'$\lambda^{-1}-\lambda^{-7}$')

    # Stretches - Computed results
    ax.plot(mean_computed_stretch, mean_computed,
            c='r',
            ls='',
            marker='*',
            label='Computed')

    theoretical = []
    l = sp.symbols('l')
    for p in pressures[:-1]:
        theoretical.append(sp.nsolve(p*R/(2*mu*T) - 1/l + l**(-7),
                                     (1, 7**(1/6)-1e-2),
                                     solver='bisect',
                                     verify=True))
    theoretical.append(7**(1/6))
    ax.scatter(theoretical, pressures*R/(2*mu*T), label='Theoretical')

    error = kwargs['error']
    itr = kwargs['itrs']
    for i in range(len(error)):
        ax.text(mean_computed_stretch[i] + 0.01,
                pressures[i]*R/(2*mu*T) - 0.01,
                s=f'{itr[i]}')
    ax.axvline(7**(1/6),
               c='k',
               ls='dashed',
               lw=0.5)

    ax.set_title(r'a)')
    ax.set_xlabel(r'Stretch Ratio $\lambda=r/R$')
    ax.set_ylabel(r'$pR / 2\mu H$')
    ax.legend(loc='best')

    # Principal stresses - Analytical relationship
    ax = axs[1]

    y = float(mu) * (1-x**-6)*x**2
    ax.plot(x, y,
            c='k',
            lw=0.5,
            label=r'$\mu(1-\lambda^{-6})\lambda^2$')

    # Principal stresses - Computed results

    ax.plot(mean_computed_stretch, mean_computed_s1,
            'r+',
            label=r'$\sigma_1$')
    ax.plot(mean_computed_stretch, mean_computed_s2, 'gx',
            label=r'$\sigma_2$')

    l = np.array(theoretical)
    ax.plot(theoretical, mu*(1 - l**-6)*l**2,
            ls='',
            marker='1',
            ms=15,
            markeredgewidth=0.5,
            label=r'Theoretical')
    ax.axvline(7**(1/6),
               c='k',
               ls='dashed',
               lw=0.5)

    ax.set_title(r'b)')
    ax.set_xlabel(r'Stretch Ratio $\lambda=r/R$')
    ax.set_ylabel(r'$\sigma$')
    ax.legend(loc='best')

    plt.tight_layout()

    import os
    if not os.path.exists('../submission/figures/'):
        os.makedirs('../submission/figures/')

    plt.savefig('../submission/figures/sphere_inflate_stretch_stress.pdf', dpi=600)


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

            self.energy = energy = INHMembrane(u, self, mu, degree=2)
            prob.add_convex_term(self.thickness*mu/2*self.J_A*energy)

            U_air_list = linear_volume_potential(self, p)
            print(type(U_air_list[0]))
            for dU in U_air_list:
                prob.add_obj_func(-Constant(p/3)*dU*dx(self.mesh))
            io = WrinklePlotter(self, energy)
            self.io.add_plotter(io.plot, 'output/xdmf_write_interval', 0)
            prob.parameters["presolve"] = True
            prob.optimize()
            l1 = np.mean(df.project(self.lambda1, self.Vs).compute_vertex_values(self.mesh))
            E = df.errornorm(u, u_old)
            E_list.append(E)
            print(f'E={E},  l1={l1}, itr={itr}')
            itr += 1
            u_old.assign(u)

    self.io.write_fields()
    return E_list, itr


def run_test():
    mesh = df.RectangleMesh(sphere.p0, sphere.p1, 40, 10)
    geo = sphere.ParametricSphere(R)
    pbc = sphere.PeriodicBoundary()

    # limit pressure
    p_max = (2*mu*T/R)*((1/sqrt(7))**(1/3) - (1/sqrt(7))**(7/3))
    input_dict = {
            'mesh': mesh,
            'geometry': geo,
            'thickness': T,
            'pressure': p_max,
            'Boundary Conditions': sphere.pinnedBCMembrane,
            'pbc': pbc}

    input_dict['output_file_path'] = 'sphere_inflate'
    struc = ParametricMembrane(input_dict)

    with df.Timer("Inflation Test Total"):
       results = inflation_test(struc)

    df.list_timings(df.TimingClear.keep,
                    [df.TimingType.wall, df.TimingType.system])

    json_results = {}
    for key in results.keys():
        json_results[key] = results[key].tolist()
    with open('results/sphere_inflate_test_results.json', 'w') as fp:
        json.dump(json_results, fp,  indent=4)
    return results

#%%
if __name__ == "__main__":

# %% save results
    results = run_test()

# %% plot results
    with open('results/sphere_inflate_test_results.json', 'r') as fp:
        results = json.load(fp)

    for key in results.keys():
        results[key] = np.array(results[key])
    plot_results(**results)
