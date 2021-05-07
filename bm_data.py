#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 18:36:16 2021

@author: alexanderniewiarowski

Wrinkling benchmark data
"""
import numpy as np

class Mosler:
    width = 200
    height = 100
    t = 0.2 
    lamb = 1852.07
    mu = 207.90

class KannoIsotropic:
    width = 200 # mm
    height = 100
    t = 0.025 # 25 um
    E = E = 3500
    nu = nu = 0.31
    mu = mu = E/2/(1+nu)
    lamb = lamb = E*nu/(1+nu)/(1-2*nu)
    lamb_bar = 2*lamb*mu/(lamb+2*mu)


class KannoOrthotropic:
    E1 = 1E5
    E2 = 10E5
    v12 = 0.3
    G = G = 0.4E5
    C = np.array([[1/E1, -v12/E2, 0],
                  [-v12/E2, 1/E2, 0],
                  [0, 0, 1/(2*G)]])

    C = np.linalg.inv(C)


def C(lamb, mu):
    return np.array([[lamb+2*mu, lamb, 0],
                     [lamb, lamb+2*mu, 0],
                     [0, 0, mu]])


def C_plane_stress(E, nu):
    return (E/(1-nu**2))*np.array([[1, nu, 0],
                                   [nu, 1, 0],
                                   [0, 0, (1-nu)/2]])
