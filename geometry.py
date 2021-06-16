#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 00:02:07 2021

@author: alexanderniewiarowski
"""

from dolfin import Expression, pi
import sympy as sp
import sympy.printing.ccode as ccode

class Cylinder():
    def __init__(self, radius, length=1):
        self.t = length
        self.r = radius
        self.l = length
        Area = pi*0.5*radius**2
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