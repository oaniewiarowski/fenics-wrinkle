#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 21:58:34 2021

@author: alexanderniewiarowski
"""

from dolfin import *
import sympy as sp

R = r = 1
p0 = Point(0, 0)
p1 = Point(2*pi, pi/2)


class ParametricSphere():
    def __init__(self, radius):
        self.t = Constant(0.01)
        self.r = radius
        self.vol_correct = 0

        x1, x2, r = sp.symbols('x[0], x[1], r')

        X = r*sp.cos(x1)*sp.sin(x2)
        Y = -r*sp.sin(x1)*sp.sin(x2)
        Z = r*sp.cos(x2)
        
        gamma_sp = [X, Y, Z]

        ccode = lambda z: sp.printing.ccode(z)

        gamma = Expression([ccode(val) for val in gamma_sp],
                           r=radius,
                           degree=4)
        
        # G_1 = ∂X/xi^1
        Gsub1 = Expression([ccode(val.diff(x1)) for val in gamma_sp],
                           r=radius,
                           degree=4)

        # G_2 = ∂X/xi^2
        Gsub2 = Expression([ccode(val.diff(x2)) for val in gamma_sp],
                           r=radius,
                           degree=4)

        self.gamma = gamma
        self.Gsub1 = Gsub1
        self.Gsub2 = Gsub2


class PeriodicBoundary(SubDomain):

    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        return bool(x[0] < DOLFIN_EPS_LARGE and x[0] > - DOLFIN_EPS_LARGE and on_boundary)

    # Map right boundary (H) to left boundary (G)
    def map(self, x, y):
        y[0] = x[0] - 2*pi
        y[1] = x[1] 
        


def pinnedBCMembrane(membrane):
    bc =[]
    bot = CompiledSubDomain("(near(x[1], 0) && on_boundary)")
    top = CompiledSubDomain("(near(x[1], pi/2) && on_boundary)", pi=pi)
    bc.append(DirichletBC(membrane.V.sub(2), Constant((0)), top))
    bc.append(DirichletBC(membrane.V.sub(0), Constant((0)), bot))
    bc.append(DirichletBC(membrane.V.sub(1), Constant((0)), bot))
    return bc
