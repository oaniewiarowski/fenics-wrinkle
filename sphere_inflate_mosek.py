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
from fenicsmembranes.parametric_membrane import ParametricMembrane
from fenics_wrinkle.utils import expand_ufl
import fenics_wrinkle.geometry.sphere as sphere
from fenics_wrinkle.materials.INH import INHMembrane
# from test_trilinear_tools import *

INTERVALS = 2


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
    computed = np.zeros((INTERVALS, num_vertex))        # computed rhs (1 - 1/lambda^4)

    # 1D arrays
    pressures = np.linspace(0.01, mem.p_0.values()[0], INTERVALS)  # inflation pressures
    pr2muH = pressures*R/(2*mu*thickness)  # Analytical 2pR/muT

    # For plotting
    mean_computed = np.zeros(INTERVALS)
    mean_computed_stretch = np.zeros(INTERVALS)
    mean_computed_s1 = np.zeros(INTERVALS)
    mean_computed_s2 = np.zeros(INTERVALS)

    mem.u.vector()[:] = 0  # make sure initial stretch is zero

    # Compute the principal stretches lambda_1 r/R
    l1 = mem.lambda1

    # Stresses
    semi_analytical_stress = np.zeros((INTERVALS, num_vertex))
    computed_stress_s1 = np.zeros((INTERVALS, num_vertex))
    computed_stress_s2 = np.zeros((INTERVALS, num_vertex))

    for i, inc_pressure in enumerate(pressures):
        # try:
        with df.Timer(f"Mosek Inflate Interval {i}"):
            mosek_inflate(mem, inc_pressure, i)
            energy = mem.energy
            fcp = {"quadrature_degree": energy.degree}
        mem.io.write_fields()

        # Element-wise stretches for current pressure
        computed_stretches[i] = df.project(l1, mem.Vs).compute_vertex_values(mem.mesh)

        # Next compute analytical plane strain result
        computed[i] = 1/computed_stretches[i] - pow(computed_stretches[i], -7)

        # Check that all stretches at all dofs are equal to analytical value pR/muH
        try:
            np.testing.assert_array_almost_equal(computed[i, :], pr2muH[i], decimal=1)
        except:
            print('********** FAIL STRETCH ***************')
        # Save the average stretch for plotting purposes
        mean_computed_stretch[i] = np.mean(computed_stretches[i])
        mean_computed[i] = np.mean(computed[i, :])

        # Stresses
        J = sqrt(det(mem.C_n)/det(mem.C_0))
        t = Constant(T)/J
        stress = Constant(mu*T)*(Constant(1)-mem.lambda1**-6)/t
        Vs = FunctionSpace(mem.mesh, 'CG', 1)
        semi_analytical_stress[i] = project(stress, mem.Vs).compute_vertex_values(mem.mesh)
        s1, s2 = energy.get_cauchy_stress()
        computed_stress_s1[i] = project(s1, Vs,
                                        form_compiler_parameters=fcp).compute_vertex_values(mem.mesh)
        computed_stress_s2[i] = project(s2, Vs,
                                        form_compiler_parameters=fcp).compute_vertex_values(mem.mesh)
        mean_computed_s1[i] = np.mean(computed_stress_s1[i, :])
        mean_computed_s2[i] = np.mean(computed_stress_s2[i, :])
        # np.testing.assert_array_almost_equal(computed_stress_s1[i, :], semi_analytical_stress[i,:], decimal=1)
        # np.testing.assert_array_almost_equal(computed_stress_s2[i, :], semi_analytical_stress[i,:], decimal=1)

        # except:
        #     break

    if PLOTTING:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2)
        fig.suptitle('Inflation of an INH Sphere')

        # Stretches - Analytical relationship
        ax = axs[0]
        x = np.linspace(min(mean_computed_stretch),
                        max(mean_computed_stretch), 500)
        y = 1/x - pow(x, -7)
        ax.plot(x, y, 'k', label='Analytical: '+r'$pR / 2\mu T$')

        # Stretches - Computed results
        ax.plot(mean_computed_stretch, mean_computed, 'r*',
                label='Computed: '+r'$\lambda^{-1}-\lambda^{-7}$')

        ax.set_ylabel(r'$pR / 2\mu T$')
        ax.set_xlabel(r'Stretch Ratio $\lambda=r/R$')

        ax.legend(loc='upper left')
        ax.set_title("Stretches")

        # Principal stresses - Analytical relationship
        ax = axs[1]
        ax.set_title("Cauchy stresses")

        y = float(mu*T)*(1-x**-6)/(float(T)/x**2)
        ax.plot(x, y, 'k',
                label='Analytical: '+r'$\mu(1-\lambda^{-6})\lambda^2$')

        # Principal stresses - Computed results
        ax.plot(mean_computed_stretch, mean_computed_s1, 'r+',
                label=r'Computed $\sigma_1$')
        ax.plot(mean_computed_stretch, mean_computed_s2, 'gx',
                label=r'Computed $\sigma_2$')
        ax.set_xlabel(r'Stretch Ratio $\lambda=r/R$')
        ax.set_ylabel(r'$\sigma$')
        ax.legend(loc='upper left')
        plt.tight_layout()


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


def mosek_inflate(self, p, i):
    u = self.u
    for j in range(2):
        with df.Timer(f"Mosek Inflate Interval {i}, Iteration {j}"):
            prob = fo.MosekProblem("No-compression membrane model")
            _ = prob.add_var(self.V, bc=self.bc)
            prob.var[0] = u

            self.energy = energy = INHMembrane(u, self, degree=2)
            prob.add_convex_term(self.thickness*self.material.mu/2*self.J_A*energy)

            U_air_list = linear_volume_potential(self, p)
            print(type(U_air_list[0]))
            for dU in U_air_list:
                prob.add_obj_func(-Constant(p/3)*dU*dx(struc.mesh))
            prob.parameters["presolve"] = True
            prob.optimize()


mesh = df.RectangleMesh(sphere.p0, sphere.p1, 80, 20)

R = 1
mu = 500
T = 0.01
geo = sphere.ParametricSphere(R)
pbc = sphere.PeriodicBoundary()

input_dict = {
        'mesh': mesh,
        'geometry': geo,
        'thickness': T,
        'material': 'Incompressible NeoHookean',
        'mu': mu,
        'cylindrical': True,
        'pressure': 5,
        'Boundary Conditions': sphere.pinnedBCMembrane,
        'pbc': pbc,
        'inflation_solver': 'Custom Newton'}


PLOTTING = True
input_dict['output_file_path'] = 'sphere_inflate'
struc = ParametricMembrane(input_dict)

with df.Timer("Inflation Test Total"):
    inflation_test(struc)
df.list_timings(df.TimingClear.keep,
                [df.TimingType.wall, df.TimingType.system])
