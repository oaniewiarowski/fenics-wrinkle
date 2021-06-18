#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 14:22:42 2021

@author: alexanderniewiarowski
"""

import unittest
from dolfin import *
from fenicsmembranes.parametric_membrane import *
import matplotlib.pyplot as plt
import numpy as np
from fenics_wrinkle.bm_data import *
from test_trilinear_tools import*

from fenics_wrinkle.geometry import Cylinder

bm = KannoIsotropic()
p = 7
geo = Cylinder(1)


class TestVolumePotential:#(unittest.TestCase):
    def __init__(self):

        self.input_dict = {
                    'resolution': [20, 20],
                    'geometry': geo,
                    'thickness': bm.t,
                    'mu': bm.mu,
                    'lmbda': bm.lamb,
                    'cylindrical': True,
                    'pressure': p,
                    'output_file_path': 'test_output/vol_pot/'}


    def test_INH_roller(self):
        name = 'INH_roller'
        self.input_dict['output_file_path'] += name
        self.input_dict['material'] = 'Incompressible NeoHookean'
        self.input_dict['Boundary Conditions'] = 'Roller'
        self.runNRM()
        self.runNRM_expanded()

        for key in self.dV_dict_results.keys():
            np.testing.assert_almost_equal(self.dV_expanded_results[key],
                                           self.dV_dict_results[key])
    def test_INH_capped(self):
        name = 'INH_capped'
        self.input_dict['output_file_path'] += name
        self.input_dict['material'] = 'Incompressible NeoHookean'
        self.input_dict['Boundary Conditions'] = 'Capped'
        self.runNRM()
        self.runNRM_expanded()

        for key in self.dV_dict_results.keys():
            np.testing.assert_almost_equal(self.dV_expanded_results[key],
                                           self.dV_dict_results[key])

    def runNRM(self):
        '''
        1) Compute reference solution using validated model (virtual work, NRM)
        2) Compute solution using iterative method and linear potentialv (NRM)
        3) Check they are equal.
        '''
        mem = ParametricMembrane(self.input_dict)
        mem.inflate(p)
        mem.io.write_fields()

        print("Elastic Potential:", assemble(mem.Pi))
        print("Volume:", float(mem.calculate_volume(mem.u)))

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
            back_bd = CompiledSubDomain("(near(x[1], 1) && on_boundary)")
            mesh_func = MeshFunction("size_t", mem.mesh, mem.mesh.topology().dim()-1)
            back_bd.mark(mesh_func, 1)
            mem.ds = Measure('ds', domain=mem.mesh, subdomain_data=mesh_func)

            x = mem.gamma[0] + u[0]
            y = mem.gamma[2] + u[2]
            area_back = -0.5*(x*mem.gsub1[2] - y*mem.gsub1[0])*mem.ds(1)
            dV = Constant(geo.l/3)*(area_back)
            
            dV_roller_terms = [-0.5*(mem.gamma[0]*gsub1[2])*mem.ds(1),
                                -0.5*(u[0]*gsub1[2])*mem.ds(1),
                                0.5*(mem.gamma[2]*gsub1[0])*mem.ds(1),
                                0.5*(u[2]*gsub1[0])*mem.ds(1)]
        
            dV_roller_terms = [geo.l/3*d for d in dV_roller_terms]
            np.testing.assert_almost_equal(assemble((1/3)*dot(mem.gamma+mem.u, mem.gsub3)*dx(mem.mesh)) + sum([assemble(d) for d in dV_roller_terms]),
                                            float(mem.calculate_volume(mem.u)), decimal=5)
            np.testing.assert_almost_equal(assemble((1/3)*dot(mem.gamma+mem.u, mem.gsub3)*dx(mem.mesh)) + assemble(dV),
                                            float(mem.calculate_volume(mem.u)), decimal=5)
        else:
            dV=0
        
        for i in range(0,9):
            PI = mem.Pi
                
            # Volume terms - correct for NRM
            dV1 = Constant(1/3)*dot(u, mem.gsub3)*dx(mem.mesh)
            dV2 = Constant(1/3)*dot(mem.gamma, mem.gsub3)*dx(mem.mesh)
            
            # volume terms - expanded for NRM test            
            dV_dict = {'u_old_x_u1': Constant(1/3)*dot(u_old, cross(mem.gsub1, gsub2))*dx(mem.mesh),
                       'u_old_x_u2': Constant(1/3)*dot(u_old, cross(gsub1,mem.gsub2))*dx(mem.mesh),
                       'u_g3': Constant(1/3)*dot(u, gsub3)*dx(mem.mesh),
                       'gamma_x_u1': Constant(1/3)*dot(mem.gamma, cross(mem.gsub1,gsub2))*dx(mem.mesh),
                       'gamma_x_u2': Constant(1/3)*dot(mem.gamma, cross(gsub1,mem.gsub2))*dx(mem.mesh)
                       }

            self.dV_dict = dV_dict
            
            for key in dV_dict.keys():
                PI += -p*dV_dict[key]
        
            if True: #self.input_dict['Boundary Conditions'] == 'Roller':
                gsub1_2 = (mem.Gsub1[2] + u[2].dx(0))
                gsub1_0 = (mem.Gsub1[0] + u[0].dx(0))
                ###
                X = mem.gamma[0]
                Z = mem.gamma[2]
                G1 = mem.Gsub1
                expanded = (X*G1[2] + X*u[2].dx(0) + u[0]*G1[2] + u[0]*(gsub1[2]-G1[2]) -Z*G1[0] -Z*u[0].dx(0) - u[2]*G1[0] - u[2]*(gsub1[0]-G1[0]) + u_old[0]*u[2].dx(0) -u_old[2]*u[0].dx(0))
                
                dV_roller_terms = expand_ufl(expanded)
                dV_roller_terms = [-.5*geo.l/3*d for d in dV_roller_terms]
                for d in dV_roller_terms:
                    # prob.add_obj_func(-Constant(p)*d)
                     PI+=-Constant(p)*d*mem.ds(1)
                     
            F = derivative(PI, mem.u, mem.v)
            solve(F==0, mem.u, mem.bc )
            mem.io.write_fields()
            
            Pi_int = assemble(mem.Pi)
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
        
        self.dV_dict_results={}
        for key in dV_dict.keys():
            self.dV_dict_results[key] = assemble(dV_dict[key])
        self.plot(PI_INT, VOL, NORM_u, NORM_g1, NORM_g2, NORM_g3)
            
    def runNRM_expanded(self):
        '''
        1) Compute reference solution using validated model (virtual work, NRM)
        2) Compute solution using iterative method and linear potentialv (NRM)
        3) Check they are equal.
        '''
        mem = ParametricMembrane(self.input_dict)
        mem.inflate(p)
        mem.io.write_fields()

        print("Elastic Potential:", assemble(mem.Pi))
        print("Volume:", float(mem.calculate_volume(mem.u)))

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
            back_bd = CompiledSubDomain("(near(x[1], 1) && on_boundary)")
            mesh_func = MeshFunction("size_t", mem.mesh, mem.mesh.topology().dim()-1)
            back_bd.mark(mesh_func, 1)
            mem.ds = Measure('ds', domain=mem.mesh, subdomain_data=mesh_func)

            x = mem.gamma[0] + u[0]
            y = mem.gamma[2] + u[2]
            area_back = -0.5*(x*mem.gsub1[2] - y*mem.gsub1[0])*mem.ds(1)
            dV = Constant(geo.l/3)*(area_back)
            
            dV_roller_terms = [-0.5*(mem.gamma[0]*gsub1[2])*mem.ds(1),
                                -0.5*(u[0]*gsub1[2])*mem.ds(1),
                                0.5*(mem.gamma[2]*gsub1[0])*mem.ds(1),
                                0.5*(u[2]*gsub1[0])*mem.ds(1)]
        
            dV_roller_terms = [geo.l/3*d for d in dV_roller_terms]
            np.testing.assert_almost_equal(assemble((1/3)*dot(mem.gamma+mem.u, mem.gsub3)*dx(mem.mesh)) + sum([assemble(d) for d in dV_roller_terms]),
                                            float(mem.calculate_volume(mem.u)), decimal=5)
            np.testing.assert_almost_equal(assemble((1/3)*dot(mem.gamma+mem.u, mem.gsub3)*dx(mem.mesh)) + assemble(dV),
                                            float(mem.calculate_volume(mem.u)), decimal=5)
        else:
            dV=0
        
        
        for i in range(0,9):
            PI = mem.Pi
          
                        # Volume terms - correct for NRM
            dV1 = Constant(1/3)*dot(u, mem.gsub3)*dx(mem.mesh)
            dV2 = Constant(1/3)*dot(mem.gamma, mem.gsub3)*dx(mem.mesh)
            
            dV_expanded = {'u_old_x_u1': expand_ufl(dot(u_old, cross(mem.Gsub1 + u.dx(0),gsub2))),
                           'u_old_x_u2': expand_ufl(dot(u_old, cross(gsub1,mem.Gsub2 + u.dx(1)))),
                           'u_g3': expand_ufl(dot(u, gsub3)),
                           'gamma_x_u1': expand_ufl(dot(mem.gamma, cross(mem.Gsub1+u.dx(0),gsub2))),
                           'gamma_x_u2': expand_ufl(dot(mem.gamma, cross(gsub1,mem.Gsub2+u.dx(1))))
                           }            
            self.dV_expanded = dV_expanded
    
            for key in dV_expanded.keys():
                for d in dV_expanded[key]:
                    PI += -Constant(p/3)*d*dx(mem.mesh)

            if True: #self.input_dict['Boundary Conditions'] == 'Roller':
                gsub1_2 = (mem.Gsub1[2] + u[2].dx(0))
                gsub1_0 = (mem.Gsub1[0] + u[0].dx(0))
                X = mem.gamma[0]
                Z = mem.gamma[2]
                G1 = mem.Gsub1
                expanded = (X*G1[2] + X*u[2].dx(0) + u[0]*G1[2] + u[0]*(gsub1[2]-G1[2]) -Z*G1[0] -Z*u[0].dx(0) - u[2]*G1[0] - u[2]*(gsub1[0]-G1[0]) + u_old[0]*u[2].dx(0) -u_old[2]*u[0].dx(0))
                
                dV_roller_terms = expand_ufl(expanded)
                dV_roller_terms = [-.5*geo.l/3*d for d in dV_roller_terms]
                
                
                for d in dV_roller_terms:
                     PI += -Constant(p)*d*mem.ds(1)
                     
            F = derivative(PI, mem.u, mem.v)
            solve(F==0, mem.u, mem.bc )
            mem.io.write_fields()
            
            Pi_int = assemble(mem.Pi)
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
            
        self.dV_expanded_results={}
        for key in dV_expanded.keys():
            form = sum(dV_expanded[key])
            self.dV_expanded_results[key] = assemble(Constant(1/3)*form*dx(mem.mesh))
        self.plot(PI_INT, VOL, NORM_u, NORM_g1, NORM_g2, NORM_g3)
            
            
    def plot(self, PI_INT, VOL, NORM_u, NORM_g1, NORM_g2, NORM_g3):
        mat = self.input_dict['material']
        fig, ax = plt.subplots(1,2,)
        fig.suptitle(self.input_dict['Boundary Conditions'] +': ' + mat)
        ax[0].plot(PI_INT[1:], '-*', label = 'Pi_int')
        ax[0].legend()
        ax[0].axhline(y=PI_INT[0])
        ax[1].plot(VOL[1:], '-*',label = 'volume')
        ax[1].legend()
        ax[1].axhline(y=VOL[0])
        
        fig, ax = plt.subplots(1,2)
        fig.suptitle(self.input_dict['Boundary Conditions'] + ': ' +mat)
        ax[0].plot(NORM_u[1:], '-*', label = '|u|')
        ax[0].legend()
        ax[0].axhline(y=NORM_u[0])
        ax[1].plot(NORM_g1[1:], '-*',label = '|g1|')
        ax[1].plot(NORM_g2[1:], '-*',label = '|g2|')
        ax[1].plot(NORM_g3[1:], '-*',label = '|g3|')
        ax[1].legend()
        ax[1].axhline(y=NORM_g1[0])
        ax[1].axhline(y=NORM_g2[0])
        ax[1].axhline(y=NORM_g3[0])
            
if __name__=="__main__":
    # suite = unittest.TestSuite() # make an empty TestSuite
    # suite.addTest(TestVolumePotential('test_INH_roller')) # add the test you want from a test class
    # runner = unittest.TextTestRunner() # the runner is what orchestrates the test running
    # runner.run(suite)
    
    test = TestVolumePotential()
    test.test_INH_roller()
    test.test_INH_capped()