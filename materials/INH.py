#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 14:03:49 2021

@author: alexanderniewiarowski
"""

from dolfin import as_tensor, as_vector
# from fenicsmembranes.parametric_membrane import *
from fenics_optim import ConvexFunction
from fenics_optim.quadratic_cones import Pow, RQuad, SDP


class INHMembrane(ConvexFunction):
    '''
    Incompressible neohookean material
    psi = (mu/2)(tr(C) + 1/det(C) - 3)
    '''

    def conic_repr(self, X, mem):
        Gsub1 = mem.Gsub1
        Gsub2 = mem.Gsub2
        t = mem.thickness
        mu = mem.material.mu

        # d>= 1/s^2
        s = self.add_var(dim=3, cone=Pow(3, 1/3))
        d = s[0]

        # C11*C22 >= C12^2 + s^2
        r = self.add_var(dim=4, cone=RQuad(4))

        C11 = 2*r[0]
        C22 = r[1]
        C12 = r[2]
        self.C_elastic = as_tensor([[C11, C12],
                                    [C12, C22]])

        # [X, s, r]
        self.add_eq_constraint([None, s[2], None], b=1)
        self.add_eq_constraint([None, -s[1], r[3]], b=0)
        self.set_linear_term([None, (t*mu/2)*d, (t*mu/2)*(C11 + C22)])

        if mem.nsd == 2:
            vect_Z = self.add_var(dim=10, cone=SDP(4))
            vect_grad_u_transp = as_vector([0, 0, 0, 0, 0, X[2], 0, X[0], X[1], X[3]])
            vect_Cnel = as_vector([-C11, -C22, 0, 0, -C12, 0, 0, 0, 0, 0])
            vect_I = as_vector([0, 0, 1, 1, 0, 0, 0, -1, -1, 0])

        if mem.nsd == 3:
            vect_Z = self.add_var(dim=15, cone=SDP(5))

            vect_grad_u_transp = as_vector([0, 0, 0, 0, 0,
                                            0, X[3], 0, 0,
                                            X[0], X[4], 0,
                                            X[1], X[5],
                                            X[2]])
            vect_Cnel = as_vector([-C11, -C22, 0, 0, 0,
                                   -C12, 0, 0, 0,
                                   0, 0, 0,
                                   0, 0,
                                   0])
            vect_I = as_vector([0, 0, 1, 1, 1,
                                0, -Gsub2[0], 0, 0,
                                -Gsub1[0], -Gsub2[1], 0,
                                -Gsub1[1], -Gsub2[2],
                                -Gsub1[2]])

        self.add_eq_constraint([vect_grad_u_transp, None, vect_Cnel, vect_Z],
                               b=vect_I)
