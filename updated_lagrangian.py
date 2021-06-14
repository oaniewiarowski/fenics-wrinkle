#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 22:47:45 2021

@author: alexanderniewiarowski
"""

from fenics_optim import *
from fenics_optim.quadratic_cones import get_slice
from fenicsmembranes.parametric_membrane import *
import matplotlib.pyplot as plt
import numpy as np
from bm_data import *
from materials.INH import INHMembrane
from materials.svk import *
import sympy as sp
import sympy.printing.ccode as ccode


class Cylinder():
    def __init__(self, radius, length=1):
        self.t = length
        self.r = radius
        self.l = length
        Area = -0.5*radius**2
        self.vol_correct = (Area*self.l/3) #TODO make sure this is loaded, currently, the vol_correct is assigned based on BC and incorrect (==0)

        x1, x2, r, l = sp.symbols('x[0], x[1], r, l')

        X = r*sp.cos((1-x1)*sp.pi)
        Y = l*x2
        Z = r*sp.sin((1-x1)*sp.pi)
        
        gamma_sp = [X, Y, Z]

        gamma = Expression([ccode(val) for val in gamma_sp],
                           r=radius,
                           l=length,
                           pi=pi,
                           degree=4)
        
        # G_1 = ∂X/xi^1
        Gsub1 = Expression([ccode(val.diff(x1)) for val in gamma_sp],
                           r=radius,
                           l=length,
                           pi=pi,
                           degree=4)

        # G_2 = ∂X/xi^2
        Gsub2 = Expression([ccode(val.diff(x2)) for val in gamma_sp],
                           r=radius,
                           l=length,
                           pi=pi,
                           degree=4)

        self.gamma = gamma
        self.Gsub1 = Gsub1
        self.Gsub2 = Gsub2
        self.l = length


bm = KannoIsotropic()

ux = 10
uy = 1
N = 40
p=7
# mesh = UnitSquareMesh(N, N)
r=1
geo = Cylinder(r)
#%% Conventional solve
input_dict = {
        # 'mesh': mesh,
        'resolution': [20,20],
        'geometry': geo,
        'thickness': bm.t,
        'material': 'SVK',
        'mu': bm.mu,
        'lmbda': bm.lamb,
        'cylindrical': True,
        'output_file_path': 'updated_lagrangian_check',
        'pressure': p,
        'Boundary Conditions': 'Capped'}

membrane = ParametricMembrane((input_dict))
membrane.inflate(p)

print("Total Potential:", assemble(membrane.Pi - p/3*dot(membrane.gsub3, membrane.u+membrane.gamma)*dx(membrane.mesh)))
print("Elastic Potential:", assemble(membrane.Pi))
print("Volume:", membrane.calculate_volume(membrane.u))
# Next compute analytical plane strain result
if input_dict['Boundary Conditions'] == 'Roller':
    computed_stretches = project(membrane.l1, membrane.Vs).compute_vertex_values(membrane.mesh)
    computed = 1 - (1/pow(computed_stretches, 4))
    analytical = p*1/(bm.mu*bm.t)
    np.testing.assert_array_almost_equal(computed[:], analytical, decimal=4)

#%% Check that total potential energy approach gives the same answer 
dV = (1/3)*dot(membrane.gamma+membrane.u, membrane.gsub3)*dx(membrane.mesh)

if input_dict['Boundary Conditions'] == 'Roller':
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

if input_dict['Boundary Conditions'] == 'Roller':
    computed_stretches = project(membrane.l1, membrane.Vs).compute_vertex_values(membrane.mesh)
    computed = 1 - (1/pow(computed_stretches, 4))
    analytical = p*1/(bm.mu*bm.t)
    np.testing.assert_array_almost_equal(computed[:], analytical, decimal=4)



#%% 1) Convex problem setup
mat = 'SVK'
input_dict = {
        'resolution': [20,20],
        'geometry': geo,
        'thickness': bm.t,
        'material': mat,
        'mu': bm.mu,
        'lmbda': bm.lamb,
        'cylindrical': True,
        'output_file_path': 'updated_lagrangian',
        'pressure': p,
        'Boundary Conditions': 'Capped'}

mem = ParametricMembrane((input_dict))
#%% 2) Newton solve, update u, g_3
mem.inflate(p)
mem.io.write_fields()
print("Elastic Potential:", assemble(mem.Pi))
print("Volume:", float(mem.calculate_volume(mem.u)))
#%% 3) Begin trilinearization loop
PI_INT = []
VOL = []
PI_INT.append(assemble(mem.Pi))
VOL.append(float(mem.calculate_volume(mem.u)))
gsub1 = project(mem.gsub1, mem.V)
gsub2 = project(mem.gsub2, mem.V)
gsub3 = project(mem.gsub3, mem.V)
#%% 3a) linear in u

for i in range(2):
    
    prob = MosekProblem("No-compression membrane model")
    u__ = prob.add_var(mem.V, bc=mem.bc)
    prob.var[0] = mem.u   # replace
    u = mem.u
    
    if mat == 'SVK':
        # energy = SVKMembraneLIN_u(u, mem, bm.lamb, bm.mu, gsub1=gsub1, gsub2=gsub2, degree=2)
        energy = SVKMembrane(u, mem, bm.lamb, bm.mu, degree=2)
        prob.add_convex_term(Constant(bm.t)*mem.J_A*energy)
    else:
        energy = INHMembrane(u, mem, degree=2)
        prob.add_convex_term(mem.J_A*Constant(bm.t*bm.mu/2)*energy)
        
    # Volume terms
    dV1 = Constant(1/3)*dot(u, gsub3)*dx(mem.mesh)
    dV2 = Constant(1/3)*dot(mem.gamma, gsub3)*dx(mem.mesh)
    prob.add_obj_func(-Constant(p)*dV1)
    prob.add_obj_func(-Constant(p)*dV2)
    prob.parameters["presolve"] = True
    prob.optimize()
    mem.io.write_fields()
    
    Pi_int = assemble(mem.Pi)
    PI_INT.append(Pi_int)
    print("Elastic Potential:", Pi_int)
    
    vol = assemble(dV1+ dV2)
    VOL.append(vol)
    print("Volume:", vol)
    
    # Update
    gsub1 = project(mem.gsub1, mem.V)
    gsub2 = project(mem.gsub2, mem.V)
    gsub3 = project(mem.gsub3, mem.V)
    
##%% 3b) linear in g_1, fixed u , g_2


# for i in range(1):
    u = mem.u
    gsub2 = project(mem.gsub2, mem.V)

    prob = MosekProblem("No-compression membrane model")
    
    g1 = prob.add_var(mem.V, bc=mem.bc)
    
    if mat == 'SVK':
        energy = SVKMembraneLIN_g1(g1, mem, bm.lamb, bm.mu, gsub2=gsub2, degree=2)
        prob.add_convex_term(Constant(bm.t)*mem.J_A*energy)
    else:
        energy = INHMembrane(u, mem, degree=2)
        prob.add_convex_term(mem.J_A*Constant(bm.t*bm.mu/2)*energy)
        
    # Volume terms    
    g3_list = my_cross(g1, gsub2)

    DV1 = my_dot(u, g3_list)
    DV2 = my_dot(mem.gamma, g3_list)
    
    for d in DV1:
        prob.add_obj_func(-Constant(p/3)*d*dx(mem.mesh))
    for d in DV2:
        prob.add_obj_func(-Constant(p/3)*d*dx(mem.mesh))
    
    prob.parameters["presolve"] = True
    prob.optimize()
    mem.io.write_fields()
    
    
    u_ = TrialFunction(mem.V)
    gradu1 = g1 - mem.Gsub1
    gradu2 = u.dx(1)
    f = as_tensor([gradu1, gradu2]).T
    
    a = inner(grad(u_), grad(mem.v))*dx(mem.mesh)
    L = inner(f, grad(mem.v))*dx(mem.mesh)
    solve(a==L, mem.u, mem.bc)
    mem.io.write_fields()
    
    dV1 = Constant(1/3)*dot(u, gsub3)*dx(mem.mesh)
    dV2 = Constant(1/3)*dot(mem.gamma, gsub3)*dx(mem.mesh)
    
    Pi_int = assemble(mem.Pi)
    PI_INT.append(Pi_int)
    print("Elastic Potential:", Pi_int)
    
    vol = assemble(dV1+ dV2)
    VOL.append(vol)
    print("Volume:", vol)
    
## %% 3c) linear in g2, fix u, g1
# for i in range(1):
    u = mem.u
    gsub1 = project(mem.gsub1, mem.V)

    prob = MosekProblem("No-compression membrane model")
    
    g2 = prob.add_var(mem.V, bc=mem.bc)
    
    if mat == 'SVK':
        energy = SVKMembraneLIN_g1(g2, mem, bm.lamb, bm.mu, gsub1=gsub1, degree=2)
        prob.add_convex_term(Constant(bm.t)*mem.J_A*energy)
    else:
        energy = INHMembrane(u, mem, degree=2)
        prob.add_convex_term(mem.J_A*Constant(bm.t*bm.mu/2)*energy)
        
    # Volume terms
    g3_list = my_cross(gsub1, g2)

    DV1 = my_dot(u, g3_list)
    DV2 = my_dot(mem.gamma, g3_list)
    
    for d in DV1:
        prob.add_obj_func(-Constant(p/3)*d*dx(mem.mesh))
    for d in DV2:
        prob.add_obj_func(-Constant(p/3)*d*dx(mem.mesh))

    
    prob.parameters["presolve"] = True
    prob.optimize()
    mem.io.write_fields()
    
    u_ = TrialFunction(mem.V)
    gradu1 = u.dx(0)
    gradu2 = g2 - mem.Gsub2
    f = as_tensor([gradu1, gradu2]).T
    
    a = inner(grad(u_), grad(mem.v))*dx(mem.mesh)
    L = inner(f, grad(mem.v))*dx(mem.mesh)
    solve(a==L, mem.u, mem.bc)
    mem.io.write_fields()
    
    dV1 = Constant(1/3)*dot(u, gsub3)*dx(mem.mesh)
    dV2 = Constant(1/3)*dot(mem.gamma, gsub3)*dx(mem.mesh)
    
    Pi_int = assemble(mem.Pi)
    PI_INT.append(Pi_int)
    print("Elastic Potential:", Pi_int)
    
    vol = assemble(dV1+ dV2)
    VOL.append(vol)
    print("Volume:", vol)
    
        # Update
    gsub1 = project(mem.gsub1, mem.V)
    gsub2 = project(mem.gsub2, mem.V)
    gsub3 = project(mem.gsub3, mem.V)
#%%
fig, ax = plt.subplots(1,2)
ax[0].plot(PI_INT, '-*', label = 'Pi_int')
ax[0].legend()
ax[0].axhline(y=PI_INT[0])
ax[1].plot(VOL, '-*',label = 'volume')
ax[1].legend()
ax[1].axhline(y=VOL[0])
#%%
class Volume(ConvexFunction):

    def __init__(self, u, mem, **kwargs):
        self.mem = mem
        ConvexFunction.__init__(self, u, parameters=None, **kwargs)

    def conic_repr(self, u):
        gsub3 = project(self.mem.Gsub3, self.mem.V)
        vol = (1/3)*dot(self.mem.gamma + u, gsub3)
      
        # t_bar = self.add_var(dim=2, cone=Quad(2))
        self.add_eq_constraint([vol,-t_bar[1]])
        self.set_linear_term([ None,t_bar[0]])
        
        
        
        

for i in np.linspace(0.1,1,10):
    print('load step:', f)
    # gamma = project(mem.gamma + mem.u, mem.V)
    # geo.Gsub1 = gamma.dx(0)
    # geo.Gsub2 = gamma.dx(1)
    # mem = ParametricMembrane((input_dict))
    # mem.io.write_fields()   
    
    # mem.setup_kinematics()
    # mem.u.vector()[:] = 0
    # mem.Gsub3 = cross(mem.Gsub1, mem.Gsub2)
    # mem.J_A =  sqrt(dot(mem.Gsub3, mem.Gsub3))


    
    prob = MosekProblem("No-compression membrane model")

    u__ = prob.add_var(mem.V, bc=mem.bc)
    prob.var[0] = mem.u   # replace
    u = mem.u
    
    

    
    
    # r = prob.add_var(FunctionSpace(mem.mesh, 'R', 0), cone=None)

    
   
    
    ja = project(mem.j_a, mem.Vs)
    # v = Volume(u, mem, degree=2)
    prob.add_obj_func(Constant(i*p)*dot((u), gsub3)*dx(mem.mesh))
    prob.add_obj_func(-ja*Constant(i*p)*dot(mem.gamma, gsub3)*dx(mem.mesh))
    # vol = (1/3)*assemble(dot(mem.gamma+mem.u, mem.gsub3)*dx(mem.mesh)) 
    # prob.add_obj_func(-Constant(p)*v)
    prob.parameters["presolve"] = True
    prob.optimize()
    mem.io.write_fields()
    
    #%%
V = FunctionSpace(membrane.mesh, 'R', 0)
v = Function(V)
v_ = TestFunction(V)

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