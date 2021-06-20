#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 15:40:57 2021

@author: alexanderniewiarowski
"""
from fenics_optim import *
from dolfin import *
import numpy as np
from fenicsmembranes.parametric_membrane import ParametricMembrane
from fenics_wrinkle.utils import *
from fenics_wrinkle.geometry.sphere import *



# from fenics_optim.quadratic_cones import get_slice
# from fenicsmembranes.parametric_membrane import *

# import numpy as np
# from fenics_wrinkle.bm_data import *
from fenics_wrinkle.materials.INH import INHMembrane


from test_trilinear_tools import *


INTERVALS = 2

def inflation_test(membrane):
    '''
    plane stress inflation of incompressible neohookean sphere
    '''

    mem = membrane
    rad = R
    thickness = float(mem.thickness)
    # mu = mem.material.mu

    # set up arrays for quantities of interest
    num_vertex = mem.mesh.num_vertices()

    # 2D arrays (element wise results)
    computed_stretches = np.zeros((INTERVALS, num_vertex))  # store computed stretch
    computed = np.zeros((INTERVALS, num_vertex))        # store computed rhs (1 - 1/lambda^4)

    # 1D arrays        
    pressures = np.linspace(0.01, mem.p_0.values()[0], INTERVALS)  # the inflation pressures
    pr2muH = pressures*rad/(2*mu*thickness)  # Analytical 2pR/muT

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
    computed_stress_s1  = np.zeros((INTERVALS, num_vertex))
    computed_stress_s2  = np.zeros((INTERVALS, num_vertex))
    


    for i, inc_pressure in enumerate(pressures):
        # try:
        with Timer(f"Mosek Inflate Interval {i}"):
            mosek_inflate(mem, inc_pressure, i)
            energy = mem.energy
        mem.io.write_fields()

        # Element-wise stretches for current pressure 
        computed_stretches[i] = project(l1, mem.Vs).compute_vertex_values(mem.mesh)
        
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
        s1, s2 = cauchy(mem)
        computed_stress_s1[i] = project(s1, Vs,
                                        form_compiler_parameters={"quadrature_degree": energy.degree}).compute_vertex_values(mem.mesh)
        computed_stress_s2[i] = project(s2, Vs,
                                        form_compiler_parameters={"quadrature_degree": energy.degree}).compute_vertex_values(mem.mesh)
        mean_computed_s1[i] = np.mean(computed_stress_s1[i, :])
        mean_computed_s2[i] = np.mean(computed_stress_s2[i, :])
        # np.testing.assert_array_almost_equal(computed_stress_s1[i, :], semi_analytical_stress[i,:], decimal=1)
        # np.testing.assert_array_almost_equal(computed_stress_s2[i, :], semi_analytical_stress[i,:], decimal=1)
            
        # except:
        #     break

    if PLOTTING:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1,2)
        fig.suptitle('Inflation of an INH Sphere')
        # Analytical relationship
        x = np.linspace(min(mean_computed_stretch), max(mean_computed_stretch), 500)
        y = 1/x - pow(x, -7)
        ax = axs[0]
        ax.plot(x, y, 'k', label='Analytical: '+r'$pR / 2\mu T$')

        # Computed results        
        ax.plot(mean_computed_stretch, mean_computed, 'r*', label='Computed: '+r'$\lambda^{-1}-\lambda^{-7}$')
        
        ax.set_ylabel(r'$pR / 2\mu T$')
        ax.set_xlabel(r'Stretch Ratio $\lambda=r/R$')
        
        ax.legend(loc='upper left')
        
        ax.set_title("Stretches")
        # ax.set_ylim(0, 1.2)
        # ax.set_xlim(1, 2.25)
        
        ax = axs[1]
        ax.set_title("Cauchy stresses")
        
        y = float(mu*T)*(1-x**-6)/(float(T)/x**2)
        ax.plot(x, y, 'k', label='Analytical: '+r'$\mu(1-\lambda^{-6})\lambda^2$')
        ax.plot(mean_computed_stretch, mean_computed_s1, 'r+', label=r'Computed $\sigma_1$')
        ax.plot(mean_computed_stretch, mean_computed_s2, 'gx', label=r'Computed $\sigma_2$')
        ax.set_xlabel(r'Stretch Ratio $\lambda=r/R$')
        ax.set_ylabel(r'$\sigma$')
        ax.legend(loc='upper left')
        plt.tight_layout()
        
def PK2_bonet(mem):

    import ufl
    i,j = ufl.indices(2)
    energy=mem.energy
    mu = mem.material.mu
    # J = mem.j_a/mem.J_A
    J = sqrt(det(energy.C_n_el)/det(mem.C_0))  # not correct, using the total  deformation
    # J = det(energy.C_n_el)/det(mem.C_0)

    gsup = mem.get_metric(mem.gsup1, mem.gsup2)
    gsup = inv(mem.C_n)
    gsup = inv(energy.C_n_el)
    # S = as_tensor(mu*(mem.C_0_sup[i,j] - (J**-2)*gsup[i,j]), [i,j])
    S = mu*as_tensor(mem.C_0_sup[i,j] - det(mem.C_0)/det(energy.C_n_el)*gsup[i,j], [i,j])
    

    return S
# def PK2_bonet(mem):

#     import ufl
#     i,j = ufl.indices(2)

#     mu = mem.material.mu
#     # J = mem.j_a/mem.J_A
#     J = sqrt(det(mem.C_n)/det(mem.C_0))  # not correct, using the total  deformation
#     # J = det(energy.C_n_el)/det(mem.C_0)

#     gsup = mem.get_metric(mem.gsup1, mem.gsup2)
#     gsup = inv(mem.C_n)
#     # gsup = inv(energy.C_n_el)
#     # S = as_tensor(mu*(mem.C_0_sup[i,j] - (J**-2)*gsup[i,j]), [i,j])
#     S = mu*as_tensor(mem.C_0_sup[i,j] - det(mem.C_0)/det(mem.C_n)*gsup[i,j], [i,j])
#     return S
def cauchy(struc):
    S = PK2_bonet(struc)
    Fnsup = as_tensor([struc.gsup1, struc.gsup2]).T
    t1, t2 = eigenvalue(Fnsup.T*struc.F_n*S*struc.F_n.T*struc.F_n)
    return t1, t2 

def linear_volume_potential(mem, p):
    '''
    u_bar.(g1 x g2_bar) + u_bar.(g1_bar x g2) + 
    u.g3_bar + X.(g1 x g2_bar) + X.(g1_bar x g2) 
    '''
    g1_bar = project(mem.gsub1, mem.V)
    g2_bar = project(mem.gsub2, mem.V)
    g3_bar = project(mem.gsub3, mem.V)
    u_bar = Function(mem.V)
    u_bar.assign(mem.u)
    
    u = mem.u
    
    # List of linear terms returned by expand_ufl
    dV_LIN = expand_ufl(dot(u_bar, cross(mem.Gsub1 + u.dx(0), g2_bar))) +\
             expand_ufl(dot(u_bar, cross(g1_bar, mem.Gsub2 + u.dx(1)))) +\
             expand_ufl(dot(u, g3_bar)) +\
             expand_ufl(dot(mem.gamma, cross(mem.Gsub1 + u.dx(0), g2_bar))) +\
             expand_ufl(dot(mem.gamma, cross(g1_bar, mem.Gsub2 + u.dx(1))))
    
    # PI = 0
    
    # for d in dV_LIN:
    #     PI += -Constant(p/3)*d*dx(mem.mesh)
    # PI_list = [-Constant(p/3)*dV*dx(mem.mesh) for dV in dV_LIN]
    return dV_LIN
 
def mosek_inflate(self, p, i):
    u = self.u
    for j in range(3):
        with Timer(f"Mosek Inflate Interval {i}, Iteration {j}"):
            prob = MosekProblem("No-compression membrane model")
            u__ = prob.add_var(self.V, bc=self.bc)
            prob.var[0] = u
    
            self.energy = energy = INHMembrane(u, self, degree=2)
            prob.add_convex_term(self.thickness*self.material.mu/2*self.J_A*energy)
            
            U_air_list = linear_volume_potential(self, p)
            print(type(U_air_list[0]))
            for dU in U_air_list:
                prob.add_obj_func(-Constant(p/3)*dU*dx(struc.mesh))
            prob.parameters["presolve"] = True
            prob.optimize()
    
 
mesh = RectangleMesh(p0, p1, 80, 20)

R = 1
mu =500
T=0.01
geo = ParametricSphere(R)
pbc = PeriodicBoundary()

input_dict = {
        'mesh': mesh,
        'geometry': geo,
        'thickness': T,
        'material': 'Incompressible NeoHookean',
        'mu': mu,
        'cylindrical': True,
        'pressure': 5,
        'Boundary Conditions': pinnedBCMembrane,
        'pbc': pbc,
        'inflation_solver': 'Custom Newton'}


PLOTTING = True
input_dict['output_file_path'] = 'sphere_inflate'
struc = ParametricMembrane(input_dict)  
    
# mosek_inflate(struc, 1)
with Timer("Inflation Test Total"):
    inflation_test(struc)
list_timings(TimingClear.keep, [TimingType.wall, TimingType.system])