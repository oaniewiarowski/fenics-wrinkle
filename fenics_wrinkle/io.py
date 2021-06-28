#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 14:37:32 2021

@author: alexanderniewiarowski
"""
import dolfin as df
from dolfin import as_vector, project, FunctionSpace, VectorFunctionSpace
import ufl
from fenics_wrinkle.utils import eigenvalue, eig_vecmat


class WrinklePlotter:

    def __init__(self, mem, energy):
        self.mem = mem
        self.energy = energy
        self.fcp = {"quadrature_degree": energy.degree}
        self.Vs = FunctionSpace(mem.mesh, 'CG', 1)
        self.V = VectorFunctionSpace(mem.mesh, 'CG', 1, dim=2)
        self.V3 = VectorFunctionSpace(mem.mesh, 'CG', 1, dim=3)

        E_el = energy.E_el
        E_w = energy.E_w

        # setup scalar outputs
        E_el1, E_el2 = eigenvalue(E_el)
        E_w1, E_w2 = eigenvalue(E_w)
        sigma_1, sigma_2 = eigenvalue(energy.get_cauchy_stress())

        self.scalar_fields = {'E_el1': E_el1,
                              'E_el2': E_el2,
                              'E_w1': E_w1,
                              'E_w2': E_w2,
                               'sigma_1': sigma_1,
                               'sigma_2': sigma_2,
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
        sigma_v1, sigma_v2 = eig_vecmat(energy.get_cauchy_stress())

        # project 2x1 eigenvectors onto deformed surface for plotting
        # strangely this also works is sigma is 2x1... why?
        norm_vec = lambda v: v/df.sqrt(df.dot(v, v))
        P = df.Identity(3) - df.outer(mem.n, mem.n)  # projection tensor
        sigma_v1_3d = norm_vec(P*as_vector([sigma_v1[0], sigma_v1[1], 0]))
        sigma_v2_3d = norm_vec(P*as_vector([sigma_v2[0], sigma_v2[1], 0]))

        self.vector_fields = {'E_el_v1': E_el_v1,
                              'E_el_v2': E_el_v2,
                              'E_w_v1': E_w_v1,
                              'E_w_v2': E_w_v2,
                              'sigma_v1': sigma_v1,
                              'sigma_v2': sigma_v2,
                               'sigma_v1_3d': sigma_v1_3d,
                               'sigma_v2_3d': sigma_v2_3d
                              }

        for name, field in self.vector_fields.items():
            if field.ufl_shape == (2, ):
                V = self.V
            else:
                V = self.V3
            self.data[name] = project(field,
                                      V,
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
            if field.ufl_shape == (2, ):
                V = self.V
            else:
                V = self.V3

            self.data[name].assign(project(field,
                                           V,
                                           form_compiler_parameters=self.fcp))

        for func in self.data.values():
            # fix t-1 in main io (t incremented in primary plotter func)
            xdmf.xdmf_file.write(func, xdmf.t-1)

        s1, s2 = eigenvalue(energy.get_cauchy_stress())
        if not hasattr(self, 's1_max'):
            s1_vertex_vals = project(s1, self.Vs,
                                     form_compiler_parameters=self.fcp
                                     ).compute_vertex_values()
            self.s1_max = df.Constant(max(s1_vertex_vals))

        s1_max = self.s1_max
        thresh = df.Constant(1e-2)  # df.Constant(0.01)

        # s_1 > thresh*s1_max ?
        s1_gt = df.gt(s1, thresh*s1_max)
        s1_lt = df.lt(s1, thresh*s1_max)

        # s_2 /max{1, s1} > thresh?
        s2_gt = df.gt(s2/ufl.Max(df.Constant(1), s1), thresh)
        s2_lt = df.lt(s2/ufl.Max(df.Constant(1), s1), thresh)

        self.state = df.conditional(ufl.And(s1_gt, s2_gt), 2,
                                    df.conditional(ufl.And(s1_gt, s2_lt),
                                                   1, 0))

        self.state_func = project(self.state,
                                  self.Vs,
                                  form_compiler_parameters=self.fcp)
        self.state_func.rename('state', 'state')
        xdmf.xdmf_file.write(self.state_func, xdmf.t-1)

        self.state_func = project(self.state,
                                  df.FunctionSpace(mem.mesh, 'DG', 0),
                                  form_compiler_parameters=self.fcp)
        self.state_func.rename('statedg', 'statedg')
        xdmf.xdmf_file.write(self.state_func, xdmf.t-1)

    def write_eigval(self, A, l1_name, l2_name):
        mem = self.mem
        l1, l2 = eigenvalue(A)

        l1 = project(l1, self.Vs, form_compiler_parameters=self.fcp)
        l1.rename(l1_name, l1_name)

        l2 = project(l2, self.Vs, form_compiler_parameters=self.fcp)
        l2.rename(l2_name, l2_name)

        mem.io.add_extra_output_function(l1)
        mem.io.add_extra_output_function(l2)
        mem.io.write_fields()

    def write_eigvec(self, A, v1_name, v2_name):
        mem = self.mem
        v1, v2 = eig_vecmat(A)
        eig1 = project(v1, self.V3, form_compiler_parameters=self.fcp)
        eig1.rename(v1_name, v1_name)
        eig2 = project(v2, self.V3, form_compiler_parameters=self.fcp)
        eig2.rename(v2_name, v2_name)
        mem.io.add_extra_output_function(eig1)
        mem.io.add_extra_output_function(eig2)
        mem.io.write_fields()
