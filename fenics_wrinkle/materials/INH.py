#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 14:03:49 2021

@author: alexanderniewiarowski
"""

from dolfin import (as_tensor, Identity, grad, assemble, dx, Constant,
                    tr, det, inv)
from fenics_optim import ConvexFunction
from fenics_optim.quadratic_cones import Pow, RQuad, SDP
from fenics_wrinkle.utils import block_to_vect
import ufl
from ufl import zero as O


class INHMembrane(ConvexFunction):
    """
    Incompressible neohookean material.

    psi = tr(C) + 1/det(C)
    """

    def __init__(self, u, mem, **kwargs):
        self.mem = mem
        ConvexFunction.__init__(self, u, parameters=None, **kwargs)

    def conic_repr(self, u):

        F0 = self.mem.F_0

        # d>= 1/s^2
        s = self.add_var(dim=3, cone=Pow(3, 1/3))
        d_ = s[0]

        # C11*C22 >= C12^2 + s^2
        r = self.add_var(dim=4, cone=RQuad(4))

        C11 = 2*r[0]
        C22 = r[1]
        C12 = r[2]
        self.C_bar_el = C_bar_el = as_tensor([[C11, C12],
                                              [C12, C22]])

        self.C_n_el = C_n_el = self.mem.C_0*C_bar_el

        # [X, s, r]
        self.add_eq_constraint([None, s[2], None], b=1)
        self.add_eq_constraint([None, -s[1], r[3]], b=0)
        # self.set_linear_term([None, d_, C11 + C22])

        d = self.mem.nsd
        # dim = 10 or 15 = top dim + geo dim + 1
        vect_Z = self.add_var(dim=sum(range(2+d+1)), cone=SDP(2+d))

        vect_I = block_to_vect([[O(2, 2), -F0.T],
                                [-F0, Identity(d)]])

        vect_grad_u_T = block_to_vect([[O((2, 2)), grad(u).T],
                                       [grad(u),  O(d, d)]])

        vect_Cnel = block_to_vect([[C_n_el,    O(2, d)],
                                   [O(d, 2), O(d, d)]])

        self.add_eq_constraint([vect_grad_u_T, None, -vect_Cnel, vect_Z],
                               b=vect_I)
        c = self.add_var(dim=1, cone=None, lx=3, ux=3)

        self.set_linear_term([None, d_, C11 + C22, None, -c[0]])

        self.E_el = 0.5*(self.C_n_el - self.mem.C_0)
        self.E_w = -(self.mem.E - self.E_el)

    def _my_evaluate(self):
        mem = self.mem
        I_C = tr(self.C_bar_el) + 1/det(self.C_bar_el)
        energy = 0.5*mem.material.mu*(I_C - Constant(mem.nsd))
        return assemble(mem.thickness*energy*mem.J_A*dx(mem.mesh),
                        form_compiler_parameters={"quadrature_degree": self.degree})

    def get_2PK_stress(self):
        """Stresses based on wrinkled metric."""
        mem = self.mem
        i, j = ufl.indices(2)
        gsup = inv(self.C_n_el)
        S = as_tensor(mem.C_0_sup[i, j] - det(mem.C_0)/det(self.C_n_el)*gsup[i, j],
                      [i, j])
        return mem.material.mu*S

    def get_cauchy_stress(self):
        mem = self.mem
        F_n = mem.F_n
        S = self.get_2PK_stress()
        F_nsup = as_tensor([mem.gsup1, mem.gsup2]).T
        sigma = F_nsup.T*F_n*S*F_n.T*F_n
        return sigma
