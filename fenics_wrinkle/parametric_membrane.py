#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 16:38:13 2019

@author: alexanderniewiarowski
"""

import dolfin as df
from dolfin import FunctionSpace, Function, Constant, Identity, assemble, dx
from dolfin import dot, inner, cross, outer, sqrt, det, inv, as_tensor
from fenics_wrinkle.calculus_utils import contravariant_base_vector
from fenics_wrinkle.gas import get_gas_law
from .base_io import InputOutputHandling


df.parameters["form_compiler"]["optimize"] = True
df.parameters["form_compiler"]["cpp_optimize"] = True
df.parameters["form_compiler"]["cpp_optimize_flags"] = '-Ofast'
df.parameters["form_compiler"]["representation"] = "uflacs"

# Optimization options for the form compiler
df.ffc_options = {"optimize": True, "quadrature_degree": 5}
df.parameters["form_compiler"]["quadrature_degree"] = 5
df.ffc_options = {"optimize": True,
                  "eliminate_zeros": True,
                  "precompute_basis_const": True,
                  "precompute_ip_const": True}
    
class ParametricMembrane(object):
    """
    Parametric membrane class
    """

    def __init__(self, kwargs):

        self.thickness = None
        self.data = {}
        self.kwargs = kwargs

        geo = kwargs.get("geometry")
        self.gamma = geo.gamma
        self.Gsub1 = geo.Gsub1
        self.Gsub2 = geo.Gsub2

        self.nsd = self.gamma.ufl_function_space().ufl_element().value_size()
        self._get_mesh()

        pbc = self.kwargs.get('pbc', None)
        self.Ve = df.VectorElement("CG",
                                   self.mesh.ufl_cell(),
                                   degree=2,
                                   dim=self.nsd)
        self.V = FunctionSpace(self.mesh, self.Ve, constrained_domain=pbc)
        self.Vs = FunctionSpace(self.mesh, 'CG', 2)  # should this be 2?

        # Construct spaces for plotting discontinuous fields
        self.W = df.VectorFunctionSpace(self.mesh, 'DG', 0, dim=self.nsd)
        self.Z = FunctionSpace(self.mesh, 'DG', 1)

        # Define trial and test function
        self.du = df.TrialFunction(self.V)
        self.v = df.TestFunction(self.V)
        self.u = Function(self.V, name="u")

        self.initial_position = df.interpolate(self.gamma, self.V)
        self.p_ext = Function(self.Vs, name="External pressure")
        self.l1 = Function(self.Vs, name="lambda 1")
        self.l2 = Function(self.Vs, name="lambda 2")
        self.l3 = Function(self.Vs, name="lambda 3")
        self.s11 = Function(self.Vs, name="stress1")
        self.s22 = Function(self.Vs, name="stress2")
        self.normals = Function(self.W, name="Surface unit normals")

        self.data = {'u': self.u,
                     'gamma': self.gamma,
                     'p_ext': self.p_ext,
                     'n': self.normals,
                     'l1': self.l1,
                     'l2': self.l2,
                     'l3': self.l3,
                     's11': self.s11,
                     's22': self.s22}

        self.thickness = Constant(kwargs['thickness'], name='thickness')

        self.setup_kinematics()

        self.bc = kwargs['Boundary Conditions'](self)

        # Create input/ouput instance and save initial states
        self.io = InputOutputHandling(self)
        self.output_file_path = kwargs["output_file_path"]
        self.io.setup()

        # volume correction for open surfaces
        self.vol_correct = 0

        # Initialize internal gas (if any)
        self.p_0 = Constant(kwargs['pressure'], name="initial pressure")
        gas_name = kwargs.get("gas", "Isentropic Gas")
        if gas_name is not None:
            gas_class = get_gas_law(gas_name)
            self.gas = gas_class(self)

        # Write initial state
        self.io.write_fields()
        self.vol_0 = assemble((1/self.nsd)*dot(self.gamma, self.Gsub3)*dx(self.mesh))
        self.area_0 = assemble(sqrt(dot(self.Gsub3, self.Gsub3))*dx(self.mesh))
        print(f"Initial volume: {float(self.vol_0)}")
        print(f"Initial area: {self.area_0}")

    def _get_mesh(self):
        if self.kwargs.get('mesh', None) is not None:
            self.mesh = self.kwargs['mesh']
            return

        res = self.kwargs.get("resolution")
        assert len(res) == self.nsd - 1, "Mesh resolution does not match dimension"

        diag = self.kwargs.get("mesh diagonal", 'crossed')
        self.mesh = df.UnitSquareMesh(res[0], res[1], diag)

    def setup_kinematics(self):

        # Get the contravariant tangent basis
        self.Gsup1 = contravariant_base_vector(self.Gsub1, self.Gsub2)
        self.Gsup2 = contravariant_base_vector(self.Gsub2, self.Gsub1)

        # Reference normal
        self.Gsub3 = cross(self.Gsub1, self.Gsub2)
        self.Gsup3 = self.Gsub3/dot(self.Gsub3, self.Gsub3)

        # Construct the covariant convective basis
        self.gsub1 = self.Gsub1 + self.u.dx(0)
        self.gsub2 = self.Gsub2 + self.u.dx(1)

        # Construct the contravariant convective basis
        self.gsup1 = contravariant_base_vector(self.gsub1, self.gsub2)
        self.gsup2 = contravariant_base_vector(self.gsub2, self.gsub1)

        # Deformed normal
        self.gsub3 = cross(self.gsub1, self.gsub2)
        self.gsup3 = self.gsub3/dot(self.gsub3, self.gsub3)

        # Deformation gradient
        gradu = outer(self.u.dx(0), self.Gsup1) + outer(self.u.dx(1), self.Gsup2)
        I = Identity(self.nsd)

        self.F = I + gradu
        self.C = self.F.T*self.F  # from initial to current

        # 3x2 deformation tensors
        self.F_0 = as_tensor([self.Gsub1, self.Gsub2]).T
        self.F_n = as_tensor([self.gsub1, self.gsub2]).T

        # 2x2 surface metrics
        self.C_0 = self.get_metric(self.Gsub1, self.Gsub2)
        self.C_0_sup = self.get_metric(self.Gsup1, self.Gsup2)
        self.C_n = self.get_metric(self.gsub1, self.gsub2)

        self.lambda1, self.lambda2, self.lambda3 = self.get_lambdas()

        self.I1 = inner(inv(self.C_0), self.C_n)
        self.I2 = det(self.C_n)/det(self.C_0)

        self.E = 0.5*(self.C_n - self.C_0)

        # Unit normals
        self.J_A = sqrt(dot(self.Gsub3, self.Gsub3))
        self.N = self.Gsub3/self.J_A
        self.j_a = sqrt(dot(self.gsub3, self.gsub3))
        self.n = self.gsub3/self.j_a

    def get_metric(self, i, j):
        return as_tensor([[dot(i, i), dot(i, j)],
                          [dot(j, i), dot(j, j)]])

    def get_lambdas(self):
        C_n = self.F_n.T*self.F_n
        C_0 = self.F_0.T*self.F_0
        I1 = inner(inv(C_0), C_n)
        I2 = det(C_n)/det(C_0)
        Q = I1**2 - 4*I2
        delta = df.conditional(df.gt(abs(Q), df.DOLFIN_EPS_LARGE*10),
                               sqrt(Q), 0)

        lambda1 = sqrt(0.5*(I1 + delta))
        lambda2 = sqrt(0.5*(I1 - delta))
        lambda3 = sqrt(det(self.C_0)/det(self.C_n))

        return lambda1, lambda2, lambda3

    def get_position(self):
        return self.gamma + self.u

    def calculate_volume(self, u):
        volume = assemble((1/self.nsd)*dot(self.gamma + u, self.gsub3)*dx(self.mesh))
        return volume