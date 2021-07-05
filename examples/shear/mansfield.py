#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 14:23:14 2021

@author: alexanderniewiarowski
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import interpolate
# 0.820305 a/L = 1/2
RES = 2000

a = sp.sympify(100)  # height
u = sp.sympify(5)  # ux mm
v = sp.sympify(1)  # uy mm
E = 3500  # youngs modulus
t = 0.025

u_hat = sp.sqrt(u**2 + v**2)


alpha_star = -sp.atan(u/v)/2 + sp.pi/2
alpha_0 = sp.pi/3.495  # works with  a0_tol = float(alpha_0) #+ 1.9e-3
alpha_0 = 0.90280065  # sp.pi/3.485 # works with  a0_tol = float(alpha_0) #+ 1.9e-3
a0_tol = float(alpha_0) + 1e-3
alpha_L = np.pi/2
print(float(alpha_star))
print(float(alpha_0))
print('*******')

assert float(alpha_star) < float(alpha_0)
# %% Build relationship between alpha and mu
den = sp.sin(2*alpha_star - alpha_0)**2 * sp.sin(alpha_0)**2
mu = sp.symbols('mu')


def get_mu(alpha):
    """EQ C-17a."""
    num = sp.sin(2*alpha_star - alpha)**2 * sp.sin(alpha)**2
    f = mu*(sp.ln(mu)/(1 - mu))**2 - num/den
    bounds = (1e-12, 1-1e-6)
    return sp.nsolve(f, bounds, solver='bisect', verify=False)


a_list = np.linspace(a0_tol, alpha_L, RES)
mu_list = [get_mu(alpha) for alpha in a_list]
alpha_to_mu = interpolate.interp1d(a_list, mu_list, fill_value="extrapolate")

fig, ax = plt.subplots()
ax.plot(a_list, mu_list)
ax.set_xlabel('alpha')
ax.set_ylabel('mu')
# %%
def calc_x0():
    """Create lookup table for alpha <--> x0."""

    def integrand(alpha):
        # Eq C-11
        mu = alpha_to_mu(alpha)
        return sp.csc(alpha)**2 * (sp.sympify(1) + mu)/(sp.sympify(1) - mu)

    alpha_list = np.linspace(float(alpha_0), alpha_L, RES)
    x0_prime_list = [integrand(alpha) for alpha in alpha_list]

    return (a/2) * integrate.cumtrapz(x0_prime_list, alpha_list, initial=0)


x0_list = calc_x0()
alpha_to_x0 = interpolate.interp1d(np.linspace(float(alpha_0), alpha_L, RES),
                                   x0_list,
                                   fill_value="extrapolate")
x0_to_alpha = interpolate.interp1d(x0_list,
                                   np.linspace(float(alpha_0), alpha_L, RES),
                                   fill_value="extrapolate")
# %%
def calc_eta(x0, y):
    alpha = float(x0_to_alpha(x0))
    mu = alpha_to_mu(alpha)
    return sp.csc(alpha) * (-y + (a/2) * ((1 + mu)/(1 - mu)))


@np.vectorize
def x0_to_x(x0, y, alpha=None):
    alpha = float(x0_to_alpha(x0)) if alpha is None else alpha
    x = x0 + np.cos(alpha)*(calc_eta(x0, 0) - calc_eta(x0, y))
    return x, y


x0 = np.linspace(0, 105, 50)
y = np.linspace(-50, 50, 50)
X0, Y = np.meshgrid(x0, y)
XS, YS = x0_to_x(X0, Y)
fig, ax = plt.subplots()

ax.plot(XS, YS)
ax.axvline(0)
# %%
@np.vectorize
def strain_x0(x0, y):
    alpha = float(x0_to_alpha(x0))
    mu = float(alpha_to_mu(alpha))
    num = u_hat * sp.sin(2*alpha_star - alpha) * sp.sin(alpha)
    den = (y - (a/2) * ((1 + mu)/(1 - mu))) * sp.ln(mu)
    x, y = x0_to_x(x0, y, alpha=alpha)
    return x, y, float(num/den)

x0 = np.linspace(0, 105, 50)
y = np.linspace(-50, 50, 50)
X0, Y = np.meshgrid(x0, y)
XS, YS, ES = strain_x0(X0, Y)

fig, ax = plt.subplots()
n = 20
e_max = 0.06
levels = np.linspace(0, e_max, n+1)
cs = ax.contourf(XS,YS, ES, levels=levels, extend='max')
cs.cmap.set_over('pink')
cs.set_clim(0, e_max)
plt.colorbar(cs)

ax.axvline(0)
ax.axhline(0)
ax.plot(XS,YS, 'k',lw=.2)
#%%
def calculate_phi():
    '''total strain energy'''
    def integrand_(alpha):
        # mu = get_mu(alpha)
        mu = float(alpha_to_mu(alpha))
        return sp.sin(2*alpha_star - alpha)/(-sp.log(mu))
    
    return E * t * u_hat**2  * integrate.romberg(integrand_, a0_tol, alpha_L)
phi = calculate_phi()

#%%

'''
# this stuff  doesn't really work 
from scipy import optimize
@np.vectorize
def x_to_x0(x,y):
    def x0_to_x_(x0):
        return  x0 + np.cos(float(x0_to_alpha(x0)))*(float(calc_eta(x0, 0)) - float(calc_eta(x0,y))) - x
    return optimize.newton(x0_to_x_, 50), y
X0_, Y_ = x_to_x0(X0, Y)
plt.plot(X0_,Y_)
    # sp.nsolve(x0 + np.cos(float(x0_to_alpha(x0)))*(calc_eta(x0, 0) - calc_eta(x0,y)) - x)
#%%


@np.vectorize
def alpha_eta_to_xy(alpha, eta):
    x0  = (alpha_to_x0(alpha))
    mu = alpha_to_mu(alpha)
    y = -(eta - (sp.csc(alpha) * (a/2) * ((1 + mu)/(1 - mu)))/sp.csc(alpha))
    return x0, y


xs, ys = alpha_eta_to_xy(np.ones(150)*(1), np.linspace(110, 229, 150))
fig, ax = plt.subplots()
ax.plot(xs, ys)
'''


