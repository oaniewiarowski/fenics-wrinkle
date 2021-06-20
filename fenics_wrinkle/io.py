#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 14:37:32 2021

@author: alexanderniewiarowski
"""

from dolfin import *
import ufl
from fenics_wrinkle.utils import eigenvalue, eig_vecmat


class WrinklePlotter:

    def __init__(self, mem, energy):
        self.mem = mem
        self.energy = energy
        self.fcp = {"quadrature_degree": energy.degree}
        self.Vs = FunctionSpace(mem.mesh, 'CG', 1)
        self.V = VectorFunctionSpace(mem.mesh, 'CG', 1)

        E_el = energy.E_el
        E_w = energy.E_w

        # setup scalar outputs
        E_el1, E_el2 = eigenvalue(E_el)
        E_w1, E_w2 = eigenvalue(E_w)
        # sigma_1, sigma_2 = eigenvalue(energy.cauchy)
        self.scalar_fields = {'E_el1': E_el1,
                              'E_el2': E_el2,
                              'E_w1': E_w1,
                              'E_w2': E_w2,
                              # 'sigma_1': sigma_1,
                              # 'sigma_2': sigma_2
                              }

        self.data = {}
        for name, field in self.scalar_fields.items():
            self.data[name] = project(field,
                                      self.Vs,
                                      form_compiler_parameters=self.fcp)
            self.data[name].rename(name, name)

        # setup vector outputs
        E_el_v1, E_el_v2 = eig_vecmat(E_el)
        E_w_v1, E_w_v2 = eig_vecmat(E_w)
        # sigma_v1, sigma_v2 = eigenvalue(energy.cauchy)
        self.vector_fields = {'E_el_v1': E_el_v1,
                              'E_el_v2': E_el_v2,
                              'E_w_v1': E_w_v1,
                              'E_w_v2': E_w_v2,
                              # 'sigma_v1': sigma_v1,
                              # 'sigma_v2': sigma_v2
                              }

        for name, field in self.vector_fields.items():
            self.data[name] = project(field,
                                      self.V,
                                      form_compiler_parameters=self.fcp)
            self.data[name].rename(name, name)

    def plot(self):
        mem = self.mem
        energy = self.energy
        xdmf = mem.io.xdmf

        for name, field in self.scalar_fields.items():
            self.data[name].assign(project(field,
                                           self.Vs,
                                           form_compiler_parameters=self.fcp))
            
        for name, field in self.vector_fields.items():
            self.data[name].assign(project(field,
                                           self.V,
                                           form_compiler_parameters=self.fcp))

        for func in self.data.values():
                # fix t-1 in main io (t incremented in primary plotter func)
                xdmf.xdmf_file.write(func, xdmf.t-1)  
                
                
    def write_eigval(self, A, l1_name, l2_name):
        mem = self.mem
        energy = self.energy
        l1, l2 = eigenvalue(A)
        
        l1 = project(l1, FunctionSpace(mem.mesh, 'CG',1), form_compiler_parameters={"quadrature_degree": energy.degree})
        l1.rename(l1_name, l1_name)
        
        l2 = project(l2, FunctionSpace(mem.mesh, 'CG',1), form_compiler_parameters={"quadrature_degree": energy.degree})
        l2.rename(l2_name, l2_name)
        
        mem.io.add_extra_output_function(l1)
        mem.io.add_extra_output_function(l2)
        mem.io.write_fields()
        
    
    def write_eigvec(self, A, v1_name, v2_name):
        mem = self.mem
        energy = self.energy
        v1, v2 = eig_vecmat(A)
        S = VectorFunctionSpace(mem.mesh, 'CG', 1)
        eig1 = project(v1, S, form_compiler_parameters={"quadrature_degree": energy.degree})
        eig1.rename(v1_name, v1_name)
        eig2 = project(v2, S, form_compiler_parameters={"quadrature_degree": energy.degree})
        eig2.rename(v2_name, v2_name)
        mem.io.add_extra_output_function(eig1)
        mem.io.add_extra_output_function(eig2)
        mem.io.write_fields()
        
    

