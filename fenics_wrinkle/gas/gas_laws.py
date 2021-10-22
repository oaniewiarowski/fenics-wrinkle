#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 13:23:17 2020

@author: alexanderniewiarowski
"""

import dolfin as df
from dolfin import Constant, assemble, dx
from . import register_gas_law

@register_gas_law('Isentropic Gas')
class IsentropicGas():
    """
        p = p_0(V_0/V)^k

        Calculates and saves the gas constant by inflating membrane to p_0.

        Used for initialization or when internal air mass changes.

    """
    def __init__(self, membrane):
        self.ready = False
        self.mem = mem = membrane
        self.kappa = mem.kwargs.get("kappa", 1)
        self.p_0 = mem.p_0
        
    def setup(self):
        '''
        Called after inflation solve!
        '''
        mem = self.mem
        self.V_0 = mem.calculate_volume(mem.u)
        self.constant = Constant(self.p_0*self.V_0**self.kappa, name='gas_constant')

        self.ready = True

    def update_pressure(self):
        mem = self.mem
        self.V = mem.calculate_volume(mem.u)
        
        self.p = self.constant/(self.V**self.kappa)

        # update dpDV 
        self.dpdV = -self.kappa*self.p/self.V

        return Constant(self.p)


@register_gas_law('Boyle')
class Boyle():
    """
    Calculates and saves the Boyle constant by inflating membrane to p_0.

    Used for initialization or when internal air mass changes.
    """
    def __init__(self, membrane, **kwargs):
        self.mem = mem = membrane
        self.p_0 = mem.p_0

    def setup(self):
        mem = self.mem
        self.V_0 = mem.calculate_volume(mem.u)
        self.boyle = Constant(self.V_0*self.p_0, name='Boyle_constant')

    def update_pressure(self):
        mem = self.mem
        self.V = mem.calculate_volume(mem.u)
        self.p = self.boyle/self.V
        self.dpdV = -self.boyle/(self.V**2)
