#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 22:47:45 2021

@author: alexanderniewiarowski
"""

import fenics_optim as fo
from fenicsmembranes.parametric_membrane import ParametricMembrane
import matplotlib.pyplot as plt
import numpy as np
from fenics_wrinkle.bm_data import *
from fenics_wrinkle.materials.INH import INHMembrane
from fenics_wrinkle.materials.svk import *
import sympy as sp
import sympy.printing.ccode as ccode
from fenics_wrinkle.test_trilinear_tools import *

from geometry import Cylinder

bm = KannoIsotropic()

ux = 10
uy = 1
N = 40
p=5
# mesh = UnitSquareMesh(N, N)
r=1
geo = Cylinder(r)
mat = 'SVK'
mat = 'Incompressible NeoHookean'
#%% Conventional solve
input_dict = {
        # 'mesh': mesh,
        'resolution': [20,20],
        'geometry': geo,
        'thickness': bm.t,
        'material': mat,
        'mu': bm.mu,
        'lmbda': bm.lamb,
        'cylindrical': True,
        'output_file_path': 'updated_lagrangian_check',
        'pressure': p,
        'Boundary Conditions': 'Roller'}

membrane = ParametricMembrane((input_dict))
membrane.inflate(p)

print("Total Potential:", assemble(membrane.Pi - p/3*dot(membrane.gsub3, membrane.u+membrane.gamma)*dx(membrane.mesh)))
print("Elastic Potential:", assemble(membrane.Pi))
print("Volume:", float(membrane.calculate_volume(membrane.u)))
# Next compute analytical plane strain result
if True: #self.input_dict['Boundary Conditions'] == 'Roller':
    computed_stretches = project(membrane.l1, membrane.Vs).compute_vertex_values(membrane.mesh)
    computed = 1 - (1/pow(computed_stretches, 4))
    analytical = p*1/(bm.mu*bm.t)
    np.testing.assert_array_almost_equal(computed[:], analytical, decimal=4)

#%% Check that total potential energy approach gives the same answer 
dV = (1/3)*dot(membrane.gamma+membrane.u, membrane.gsub3)*dx(membrane.mesh)

if True: #self.input_dict['Boundary Conditions'] == 'Roller':
    front_bd = CompiledSubDomain("(near(x[1], 0) && on_boundary)")
    back_bd = CompiledSubDomain("(near(x[1], 1) && on_boundary)")
    mesh_func = MeshFunction("size_t", membrane.mesh, membrane.mesh.topology().dim()-1)
    front_bd.mark(mesh_func, 1)
    back_bd.mark(mesh_func, 2)
    File("mesh_func.pvd") << mesh_func
    membrane.ds = Measure('ds', domain=membrane.mesh, subdomain_data=mesh_func)
    
    pos = membrane.get_position()
    x = pos[0]
    y = pos[2]
    # area_front = -0.5*(x*membrane.gsub1[2] - y*membrane.gsub1[0])*membrane.ds(1)
    area_back = -0.5*(x*membrane.gsub1[2] - y*membrane.gsub1[0])*membrane.ds(2)
    dV += geo.l/3*(area_back)

PI = membrane.Pi - Constant(p)*dV
F = derivative(PI, membrane.u, membrane.v)
solve(F==0, membrane.u, membrane.bc )
membrane.io.write_fields()

print("Total Potential:", assemble(PI))
print("Elastic Potential:", assemble(membrane.Pi))
print("Volume:", assemble(dV))

if True: #self.input_dict['Boundary Conditions'] == 'Roller':
    computed_stretches = project(membrane.l1, membrane.Vs).compute_vertex_values(membrane.mesh)
    computed = 1 - (1/pow(computed_stretches, 4))
    analytical = p*1/(bm.mu*bm.t)
    np.testing.assert_array_almost_equal(computed[:], analytical, decimal=4)



#%% 1) Convex problem setup

input_dict = {
        'resolution': [30,30],
        'diagonal':'crossed',
        'geometry': geo,
        'thickness': bm.t,
        'material': mat,
        'mu': bm.mu,
        'lmbda': bm.lamb,
        'cylindrical': False,
        
        'output_file_path': 'updated_lagrangian_INH',
        'pressure': p,
        'Boundary Conditions': 'Capped'}

mem = ParametricMembrane((input_dict))
##%% 2) Newton solve, update u, g_3
mem.inflate(p)
mem.io.write_fields()
   
    
print("Elastic Potential:", assemble(mem.Pi))
print("Volume:", float(mem.calculate_volume(mem.u)))
##%% 3) Begin trilinearization loop
PI_INT = []
VOL = []
NORM_u = [norm(mem.u)]

PI_INT.append(assemble(mem.Pi))
VOL.append(float(mem.calculate_volume(mem.u)))



gsub1 = project(mem.gsub1, mem.V)
gsub2 = project(mem.gsub2, mem.V)
gsub3 = project(mem.gsub3, mem.V)
NORM_g1 = [norm(gsub1)]
NORM_g2 = [norm(gsub2)]
NORM_g3 = [norm(gsub3)]
mem.u.vector()[:]=0
gsub1 = project(mem.gsub1, mem.V)
gsub2 = project(mem.gsub2, mem.V)
gsub3 = project(mem.gsub3, mem.V)
u = mem.u
u_old = Function(mem.V)
u_old.assign(u)


if True: #self.input_dict['Boundary Conditions'] == 'Roller':
    front_bd = CompiledSubDomain("(near(x[1], 0) && on_boundary)")
    back_bd = CompiledSubDomain("(near(x[1], 1) && on_boundary)")
    mesh_func = MeshFunction("size_t", mem.mesh, mem.mesh.topology().dim()-1)
    front_bd.mark(mesh_func, 1)
    back_bd.mark(mesh_func, 2)
    mem.ds = Measure('ds', domain=mem.mesh, subdomain_data=mesh_func)
    
    pos = mem.get_position()
    x = mem.gamma[0] + u[0]
    y = mem.gamma[2] + u[2]
    area_back = -0.5*(x*mem.gsub1[2] - y*mem.gsub1[0])*mem.ds(2)
    # area_back = x*mem.gsub1[2] - y*mem.gsub1[0]

    dV = Constant(geo.l/3)*(area_back)
    dV_roller_terms = [-0.5*(mem.gamma[0]*gsub1[2])*mem.ds(2),
                        -0.5*(u[0]*gsub1[2])*mem.ds(2),
                        0.5*(mem.gamma[2]*gsub1[0])*mem.ds(2),
                        0.5*(u[2]*gsub1[0])*mem.ds(2)]

    # dV_roller_terms = expand_ufl(dV_roller_terms)
    dV_roller_terms = [geo.l/3*d for d in dV_roller_terms]
    np.testing.assert_almost_equal(assemble((1/3)*dot(mem.gamma+mem.u, mem.gsub3)*dx(mem.mesh))+ sum([assemble(d) for d in dV_roller_terms]), float(mem.calculate_volume(mem.u)), decimal=5)
else:
    dV=0

#%% 3a) linear in u


for i in range(5):

    prob = MosekProblem("No-compression membrane model")
    u__ = prob.add_var(mem.V, bc=mem.bc)
    prob.var[0] = u

    # energy = SVKMembraneLIN_u(u, mem, bm.lam1b, bm.mu, gsub1=gsub1, gsub2=gsub2, degree=2)
    if mat =='SVK':
        energy = SVKMembrane(u, mem, bm.lamb, bm.mu, degree=5)
        prob.add_convex_term(Constant(bm.t)*mem.J_A*energy)
    if mat =='Incompressible NeoHookean':
        energy = INHMembrane(u, mem, degree=2)
        prob.add_convex_term(bm.t*bm.mu/2*mem.J_A*energy)
        
    # Volume terms
    dV1 = Constant(1/3)*dot(u, gsub3)*dx(mem.mesh)
    dV2 = Constant(1/3)*dot(mem.gamma, gsub3)*dx(mem.mesh)
    
    u_old_x_u1 = expand_ufl(dot(u_old, cross(mem.Gsub1 + u.dx(0),gsub2)))
    u_old_x_u2 = expand_ufl(dot(u_old, cross(gsub1,mem.Gsub2 + u.dx(1))))
    
    gamma_x_u1 = expand_ufl(dot(mem.gamma, cross(mem.Gsub1+u.dx(0),gsub2)))
    gamma_x_u2 = expand_ufl(dot(mem.gamma, cross(gsub1,mem.Gsub2+u.dx(1))))
    
    gsub1_2 = (mem.Gsub1[2] + u[2].dx(0))
    gsub1_0 = (mem.Gsub1[0] + u[0].dx(0))
    ###
    X = mem.gamma[0]
    Z = mem.gamma[2]
    G1 = mem.Gsub1
    expanded = (X*G1[2] + X*u[2].dx(0) + u[0]*G1[2] + u[0]*(gsub1[2]-G1[2]) -Z*G1[0] -Z*u[0].dx(0) - u[2]*G1[0] - u[2]*(gsub1[0]-G1[0]) + u_old[0]*u[2].dx(0) -u_old[2]*u[0].dx(0))
    
    # dV_roller_terms = -0.5*(x*gsub1_2 - y*gsub1_0)
    # dV_roller_terms = [-0.5*(mem.gamma[0]*(mem.Gsub1[2] + u[2].dx(0))),
    #                     -0.5*(u[0]*(mem.Gsub1[2] + u[2].dx(0))),
    #                     0.5*(mem.gamma[2]*(mem.Gsub1[0] + u[0].dx(0))),
    #                     0.5*(u[2]*(mem.Gsub1[0] + u[0].dx(0)))]
    dV_roller_terms = expand_ufl(expanded)
    dV_roller_terms = [-.5*geo.l/3*d for d in dV_roller_terms]
    
    prob.add_obj_func(-p*dV1)
    # prob.add_obj_func(-p*dV2)  # constant
    
    for d in u_old_x_u1:
        prob.add_obj_func(-Constant(p/3)*d*dx(mem.mesh))
    for d in u_old_x_u2:
        prob.add_obj_func(-Constant(p/3)*d*dx(mem.mesh))
    for d in gamma_x_u1:
        prob.add_obj_func(-Constant(p/3)*d*dx(mem.mesh))
    for d in gamma_x_u2:
        prob.add_obj_func(-Constant(p/3)*d*dx(mem.mesh))
    if True: #self.input_dict['Boundary Conditions'] == 'Roller':
        for d in dV_roller_terms:
            # prob.add_obj_func(-Constant(p)*d)
            prob.add_obj_func(-Constant(p)*d*mem.ds(2))
            
    prob.parameters["presolve"] = True
    prob.optimize()
    mem.io.write_fields()
    
    Pi_int = energy.evaluate()#assemble(mem.Pi)
    PI_INT.append(Pi_int)
    print("Elastic Potential:", Pi_int)
    
    vol = assemble(dV1 + dV2 + dV)
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
plot_result()

#%% plot
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
#%%    
        


vol = (1/3)*dot(membrane.gamma + membrane.u, membrane.gsub3)*dx(membrane.mesh)

pos = membrane.get_position()
x = pos[0]
y = pos[2]
area = -0.5*(x*membrane.gsub1[2] - y*membrane.gsub1[0])*membrane.ds(1)

VOL = assemble(vol + membrane.kwargs['geometry'].t/3*area)
                    volume += membrane.vol_correct
PI = membrane.Pi - dot(Constant(p)*membrane.gsub3, membrane.u)*dx(membrane.mesh)
F = derivative(PI, membrane.u, membrane.v)
solve(F==0, membrane.u, membrane.bc )
membrane.io.write_fields()
print(assemble(PI))