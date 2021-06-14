#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 14:03:49 2021

@author: alexanderniewiarowski
"""

from dolfin import as_tensor, Identity, grad
from fenics_optim import ConvexFunction
from fenics_optim.quadratic_cones import Pow, RQuad, SDP
from utils import block_to_vect
from ufl import zero as O


class INHMembrane(ConvexFunction):

    """
    Incompressible neohookean material.
    psi = tr(C) + 1/det(C)
    """

    def conic_repr(self, u, mem):
        F0 = mem.F_0

        # d>= 1/s^2
        s = self.add_var(dim=3, cone=Pow(3, 1/3))
        d_ = s[0]

        # C11*C22 >= C12^2 + s^2
        r = self.add_var(dim=4, cone=RQuad(4))

        C11 = 2*r[0]
        C22 = r[1]
        C12 = r[2]
        self.C_elastic = C = as_tensor([[C11, C12],
                                        [C12, C22]])

        # [X, s, r]
        self.add_eq_constraint([None, s[2], None], b=1)
        self.add_eq_constraint([None, -s[1], r[3]], b=0)
        self.set_linear_term([None, d_, C11 + C22])

        d = mem.nsd
        # dim = 10 or 15 = top dim + geo dim + 1
        vect_Z = self.add_var(dim=sum(range(2+d+1)), cone=SDP(2+d))

        vect_I = block_to_vect([[O(2, 2), -F0.T],
                                [-F0, Identity(mem.nsd)]])

        vect_grad_u_T = block_to_vect([[O((2, 2)), grad(u).T],
                                       [grad(u),  O(d, d)]])

        vect_Cnel = block_to_vect([[C,      O(2, d)],
                                   [O(d, 2), O(d, d)]])

        self.add_eq_constraint([vect_grad_u_T, None, -vect_Cnel, vect_Z],
                               b=vect_I)
