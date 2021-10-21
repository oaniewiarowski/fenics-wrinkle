#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 23:11:43 2021

@author: alexanderniewiarowski
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 14:40:10 2021

@author: alexanderniewiarowski
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 22:47:45 2021

@author: alexanderniewiarowski
"""

import fenics_optim as fo
import dolfin as df
from dolfin import (assemble, project, norm, dx, ds, dot, cross, Constant)
from fenicsmembranes.parametric_membrane import ParametricMembrane
import matplotlib.pyplot as plt
import numpy as np
from fenics_wrinkle.bm_data import KannoIsotropic
from fenics_wrinkle.materials.INH import INHMembrane

import sympy as sp
import sympy.printing.ccode as ccode

from fenics_wrinkle.utils import expand_ufl
from fenics_wrinkle.io import WrinklePlotter
bm = KannoIsotropic()

N = 30
p = 5



import fenics_wrinkle.geometry.sphere as sphere
mesh = df.RectangleMesh(sphere.p0, sphere.p1, 20, 20)
sph = False
R = 1
mu = 500
T = 0.01
geo = sphere.ParametricSphere(R)
pbc = sphere.PeriodicBoundary()
def pinnedBCMembrane(membrane):
    bc = []
    bot = df.CompiledSubDomain("(near(x[1], 0) && on_boundary)")
    top = df.CompiledSubDomain("(near(x[1], pi/2) && on_boundary)", pi=df.pi)
    bc.append(df.DirichletBC(membrane.V, Constant((0,0,0)), top))
    # bc.append(df.DirichletBC(membrane.V.sub(0), Constant((0)), bot))
    # bc.append(df.DirichletBC(membrane.V.sub(1), Constant((0)), bot))

    return bc
input_dict = {
        'mesh': mesh,
        'geometry': geo,
        'thickness': T,
        'material': 'Incompressible NeoHookean',
        'mu': mu,
        'output_file_path': 'updated_lagrangian_check',
        'cylindrical': False,
        'pressure': p,
        'Boundary Conditions': pinnedBCMembrane,
        'pbc': pbc,
        'inflation_solver': 'Custom Newton'}


mat = 'Incompressible NeoHookean'


#%% Conventional solve


membrane = ParametricMembrane((input_dict))
membrane.inflate(p)

print("Total Potential:", assemble(membrane.Pi - p/3*dot(membrane.gsub3, membrane.u+membrane.gamma)*dx(membrane.mesh)))
print("Elastic Potential:", assemble(membrane.Pi))
print("Volume:", float(membrane.calculate_volume(membrane.u)))

# Next compute analytical plane strain result
if input_dict['Boundary Conditions'] == 'Roller':
    computed_stretches = project(membrane.l1, membrane.Vs).compute_vertex_values(membrane.mesh)
    computed = 1 - (1/pow(computed_stretches, 4))
    analytical = p*1/(bm.mu*bm.t)
    np.testing.assert_array_almost_equal(computed[:], analytical, decimal=4)

#%% Check that total potential energy approach gives the same answer
dV = (1/3)*dot(membrane.gamma+membrane.u, membrane.gsub3)*dx(membrane.mesh)



PI = membrane.Pi - Constant(p)*dV 
F = df.derivative(PI, membrane.u, membrane.v)
df.solve(F==0, membrane.u, membrane.bc)
membrane.io.write_fields()
boyle_nrm = membrane.gas.constant

# add pressure constrained loading
khat = Constant((0,0,-1))
PI -= df.dot(khat,membrane.u)*dx(membrane.mesh)
F = df.derivative(PI, membrane.u, membrane.v)
df.solve(F==0, membrane.u, membrane.bc)

membrane.io.write_fields()
print("Total Potential:", assemble(PI))
print("Elastic Potential:", assemble(membrane.Pi))
print("Volume Potential:", assemble(- Constant(p)*dV))
print("Volume:", assemble(dV))

boyle_pressure_check = float(boyle_nrm/membrane.calculate_volume(membrane.u))
boyle_vol_check = float(membrane.calculate_volume(membrane.u))


#%% 1) Convex problem setup



input_dict = {
        'mesh': mesh,
        'geometry': geo,
        'thickness': T,
        'material': 'Incompressible NeoHookean',
        'mu': mu,
        'output_file_path': 'inflation_log_term_sphere',
        'cylindrical': False,
        'pressure': p,
        'Boundary Conditions': pinnedBCMembrane,
        'pbc': pbc,
        'inflation_solver': 'Custom Newton'}
khat = Constant((0,0,-1))
mem = ParametricMembrane((input_dict))
mem.inflate(p)
print("Volume:", float(mem.calculate_volume(mem.u)))
dV = dot(mem.gamma+mem.u, mem.gsub3)*dx(mem.mesh)
PI = mem.Pi - Constant(p/3)*dV  - df.dot(khat,mem.u)*dx(mem.mesh)
F = df.derivative(PI, mem.u, mem.v)
df.solve(F==0, mem.u, mem.bc)
mem.io.write_fields()
boyle_nrm = membrane.gas.constant
boyle_pressure_check = float(boyle_nrm/mem.calculate_volume(mem.u))
boyle_vol_check = float(mem.calculate_volume(mem.u))

dv = (1/3)*dot(mem.gamma+mem.u, mem.gsub3)*dx(mem.mesh)

print("Elastic Potential:", assemble(mem.Pi))
print("Volume:", float(mem.calculate_volume(mem.u)))

#%%
##%% 3) Begin trilinearization loop
PI_INT = []
VOL = []
NORM_u = [norm(mem.u)]

PI_INT.append(assemble(mem.Pi))
VOL.append(boyle_vol_check)

gsub1 = project(mem.gsub1, mem.V)
gsub2 = project(mem.gsub2, mem.V)
gsub3 = project(mem.gsub3, mem.V)
NORM_g1 = [norm(gsub1)]
NORM_g2 = [norm(gsub2)]
NORM_g3 = [norm(gsub3)]
# mem.u.vector()[:]=0
# gsub1 = project(mem.gsub1, mem.V)
# gsub2 = project(mem.gsub2, mem.V)
# gsub3 = project(mem.gsub3, mem.V)
u = mem.u
u_old = df.Function(mem.V)
u_old.assign(u)

def linear_volume_potential_split(mem):
    import ufl
    from ufl.algorithms import expand_derivatives
    """

    u_bar.(g1 x g2_bar) + u_bar.(g1_bar x g2) +\
    u.g3_bar + X.(g1 x g2_bar) + X.(g1_bar x g2)

    """
    def applyvec(F):
        # for bc in membrane.bc:
        #     bc.apply(F.vector())
        return F
    g1_bar = applyvec(project(mem.gsub1, mem.V))
    g2_bar = applyvec(project(mem.gsub2, mem.V))
    g3_bar = project(mem.gsub3, mem.V)
    u_bar = df.Function(mem.V)
    u_bar.assign(mem.u)

    u = mem.u

    # This generates a list of linear terms returned by expand_ufl
    # Currently, expand_ufl doesn't support more complex expressions and
    # it is not possible to accomplish this with one call to expand_ufl
    # Vg1u + Vg2u + Vu
    # dV_LINu = expand_ufl(dot(u.dx(0), cross(g2_bar, u_bar))) +\
    #           expand_ufl(dot(u.dx(1), cross(u_bar, g1_bar))) +\
    dV_LINu = expand_ufl(dot(u, g3_bar))
    #  Vg1u_const + Vg2u_const
    dV_LINu_const = []#expand_ufl(dot(mem.Gsub1, cross(g2_bar, u_bar))) +\
                    # expand_ufl(dot(mem.Gsub2, cross(u_bar, g1_bar)))


    # Vg1X + Vg2X
    dV_LINX = []#expand_ufl(dot(u.dx(0), cross(g2_bar,mem.gamma))) +\
                  # expand_ufl(dot(u.dx(1), cross(mem.gamma, g1_bar))) 
                   
                   
     # Vg1X + Vg2X
    # dV_LINX_const = expand_ufl(dot(mem.Gsub1, cross(g2_bar, mem.gamma))) +\
    #                 expand_ufl(dot(mem.Gsub2, cross(mem.gamma, g1_bar))) +\
    dV_LINX_const =                expand_ufl(dot(mem.gamma, g3_bar))
    return dV_LINX + dV_LINu, dV_LINX_const + dV_LINu_const


##%% 3a) linear in u

R = df.FunctionSpace(mem.mesh, "Real", 0)
R3 = df.VectorFunctionSpace(mem.mesh, "Real", 0, dim=3)

for i in range(3):

    prob = fo.MosekProblem("No-compression membrane model")
    y, u_ = prob.add_var([R3, mem.V], cone=[fo.Exp(3), None], bc=[None] + mem.bc, name=['y','_u'])

    prob.var[1] = mem.u
    prob.add_obj_func( -df.dot(khat, mem.u)*dx(mem.mesh))



    # Volume terms
    dV1 = Constant(1/3)*dot(u, gsub3)*dx(mem.mesh)
    dV2 = Constant(1/3)*dot(mem.gamma, gsub3)*dx(mem.mesh)


    prob.add_eq_constraint(R, A=lambda mu: [mu * y[1] * dx(mem.mesh)], b=1, name='one') 


    prob.add_obj_func(-Constant(boyle_nrm) * y[2] * dx(mem.mesh))

    

    # dvlin = (linear_volume_potential(mem))
    dvlin_split, dvlinconst = linear_volume_potential_split(mem)
    # dvlin2, dvlin3 = (linear_volume_potential(mem))
    prob.add_eq_constraint(R, A=lambda mu: [mu * Constant(1)*y[0] * dx(mem.mesh),
                                            -mu*Constant(1/3)*sum(dvlin_split)*dx(mem.mesh)  ], 
                           b=lambda mu: mu*Constant(1/3)*sum(dvlinconst)*dx(mem.mesh) , name='lamb') 
    energy = INHMembrane(u, mem, degree=2)
    prob.add_convex_term(bm.t*bm.mu/2*mem.J_A*energy)
    io = WrinklePlotter(mem, energy)
    mem.io.add_plotter(io.plot, 'output/xdmf_write_interval', 0)    

    prob.parameters["presolve"] = True
    prob.optimize()
    mem.io.write_fields()

    Pi_int = energy.evaluate()#assemble(mem.Pi)
    PI_INT.append(Pi_int)
    print("Elastic Potential:", Pi_int)

    vol = float(mem.calculate_volume(mem.u)) #assemble(dV1 + dV2 + dV)
    # print(assemble(Constant(1/3)*dvlin*dx(mem.mesh) + dV_roller_terms*mem.ds(2)))
    VOL.append(vol)
    print("Volume:", vol)
    print("Check Volume:", float(mem.calculate_volume(u)))
    # np.testing.assert_almost_equal(vol, float(mem.calculate_volume(mem.u)))
    # Update
    gsub1 = project(mem.gsub1, mem.V)
    gsub2 = project(mem.gsub2, mem.V)
    gsub3 = project(mem.gsub3, mem.V)
    NORM_u.append(norm(u))
    NORM_g1.append(norm(gsub1))
    NORM_g2.append(norm(gsub2))
    NORM_g3.append(norm(gsub3))

    u_old.assign(u)
    y2 =assemble(y[2]*dx(mem.mesh))
    y0 =assemble(y[0]*dx(mem.mesh))
    print(-y2)
    b=float(boyle_nrm)
    print(-y2*b)
    print( -b*df.ln(y0))



    [print(assemble(prob.get_var('y')[i]*dx(mem.mesh))) for i in range(3)]
    print('x.n', assemble(dot(mem.get_position(),mem.gsub3)*dx(mem.mesh))/3)
lamb = prob.get_lagrange_multiplier('lamb')
print(assemble(Constant(lamb)*((Constant(1/9)*sum(dvlin_split) + Constant(1/9)*sum(dvlinconst))*dx(mem.mesh) )))
# plot_result()

##%% plot
def plot_result():
    title = f"{input_dict['Boundary Conditions']}: {mat}, p:{p} \n {input_dict['cylindrical']}"
    fig, ax = plt.subplots(2,2, figsize=[8,12])
    ax = ax.flatten()
    fig.suptitle(title)
    ax[0].plot(PI_INT[1:], '-*', label = 'Pi_int')
    ax[0].text(0.25, 0.25,
            f'ref: {PI_INT[0]:e} \n final: {PI_INT[-1]:e}',
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax[0].transAxes)
    ax[0].legend()
    ax[0].axhline(y=PI_INT[0])
    ax[1].plot(VOL[1:], '-*',label = 'volume')
    ax[1].text(0.25, 0.25,
        f'ref: {VOL[0]:.5f} \n final: {VOL[-1]:.5f}',
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax[1].transAxes)
    ax[1].legend()
    ax[1].axhline(y=VOL[0])


    ax[2].plot(NORM_u[1:], '-*', label = '|u|')
    ax[2].legend()
    ax[2].axhline(y=NORM_u[0])
    ax[2].text(0.25, 0.25,
        f'ref: {NORM_u[0]:e} \n final: {NORM_u[-1]:e}',
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax[2].transAxes)
    ax[3].plot(NORM_g1[1:], '-*',label = '|g1|')
    ax[3].plot(NORM_g2[1:], '-*',label = '|g2|')
    ax[3].plot(NORM_g3[1:], '-*',label = '|g3|')
    ax[3].legend()
    ax[3].axhline(y=NORM_g1[0])
    ax[3].axhline(y=NORM_g2[0])
    ax[3].axhline(y=NORM_g3[0])
plot_result()
