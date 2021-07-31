#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 15:04:52 2021

@author: alexanderniewiarowski
"""

from dolfin import project, dot, cross, Function
from fenics_wrinkle.utils import expand_ufl


def linear_volume_potential_split(mem):
    """

    u_bar.(g1 x g2_bar) + u_bar.(g1_bar x g2) +\
    u.g3_bar + X.(g1 x g2_bar) + X.(g1_bar x g2)

    """

    g1_bar = project(mem.gsub1, mem.V)
    g2_bar = project(mem.gsub2, mem.V)
    g3_bar = project(mem.gsub3, mem.V)
    u_bar = Function(mem.V)
    u_bar.assign(mem.u)

    u = mem.u

    # This generates a list of linear terms returned by expand_ufl
    # Currently, expand_ufl doesn't support more complex expressions and
    # it is not possible to accomplish this with one call to expand_ufl
    # Vg1u + Vg2u + Vu
    dV_LINu = expand_ufl(dot(u.dx(0), cross(g2_bar, u_bar))) +\
              expand_ufl(dot(u.dx(1), cross(u_bar, g1_bar))) +\
              expand_ufl(dot(u, g3_bar))

    #  Vg1u_const + Vg2u_const
    dV_LINu_const = expand_ufl(dot(mem.Gsub1, cross(g2_bar, u_bar))) +\
                    expand_ufl(dot(mem.Gsub2, cross(u_bar, g1_bar)))

    # Vg1X + Vg2X
    dV_LINX = expand_ufl(dot(u.dx(0), cross(g2_bar, mem.gamma))) +\
              expand_ufl(dot(u.dx(1), cross(mem.gamma, g1_bar)))

    # Vg1X + Vg2X
    dV_LINX_const = expand_ufl(dot(mem.Gsub1, cross(g2_bar, mem.gamma))) +\
                    expand_ufl(dot(mem.Gsub2, cross(mem.gamma, g1_bar))) +\
                    expand_ufl(dot(mem.gamma, g3_bar))

    return dV_LINX + dV_LINu, dV_LINX_const + dV_LINu_const



def linear_volume_potential(mem, p):
    """

    u_bar.(g1 x g2_bar) + u_bar.(g1_bar x g2) +\
    u.g3_bar + X.(g1 x g2_bar) + X.(g1_bar x g2)

    """
    g1_bar = project(mem.gsub1, mem.V)
    g2_bar = project(mem.gsub2, mem.V)
    g3_bar = project(mem.gsub3, mem.V)
    u_bar = Function(mem.V)
    u_bar.assign(mem.u)

    u = mem.u

    # This generates a list of linear terms returned by expand_ufl
    # Currently, expand_ufl doesn't support more complex expressions and
    # it is not possible to accomplish this with one call to expand_ufl
    dV_LIN = expand_ufl(dot(u_bar, cross(mem.Gsub1 + u.dx(0), g2_bar))) +\
             expand_ufl(dot(u_bar, cross(g1_bar, mem.Gsub2 + u.dx(1)))) +\
             expand_ufl(dot(u, g3_bar)) +\
             expand_ufl(dot(mem.gamma, cross(mem.Gsub1 + u.dx(0), g2_bar))) +\
             expand_ufl(dot(mem.gamma, cross(g1_bar, mem.Gsub2 + u.dx(1))))

    return dV_LIN
