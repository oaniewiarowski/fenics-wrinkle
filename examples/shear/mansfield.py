#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 14:23:14 2021

@author: alexanderniewiarowski

Original solution:
https://www.jstor.org/stable/77716

Implementation follows Appendix C of De Rooij thesis:
https://repository.tudelft.nl/islandora/object/uuid:407b1564-04d7-4f24-badf-61bb960ec9d2
"""
import os
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc_file
from scipy import integrate
from scipy import interpolate

rc_file('../submission/journal_rc_file.rc')

RES = 2000
tol = 1e-3
alpha_L = np.pi/2

a = sp.sympify(100)  # height
u = sp.sympify(5)  # ux mm
v = sp.sympify(1)  # uy mm
E = 3500  # youngs modulus
t = 0.025

u_hat = sp.sqrt(u**2 + v**2)
alpha_star = -sp.atan(u/v)/2 + sp.pi/2

def f(alpha_0):
    """search for alpha_0"""
    a0_tol = alpha_0 + tol
    a_list = np.linspace(alpha_0, alpha_L, RES) # FIXME
    mu_list = [get_mu(alpha, alpha_0) for alpha in a_list]
    alpha_to_mu = interpolate.interp1d(a_list, mu_list,
                                       fill_value="extrapolate")
    x0_list = calc_x0(alpha_0, alpha_to_mu=alpha_to_mu)
    alpha_to_x0 = interpolate.interp1d(a_list, x0_list,
                                       fill_value="extrapolate")
    x_guessed.append(alpha_to_x0(alpha_L))
    return alpha_to_x0(alpha_L) - 100


def bisection(f, a, b, N=10):
    a_n = a
    b_n = b
    for n in range(1, N+1):
        print(n)
        m_n = (a_n + b_n)/2
        f_m_n = f(m_n)
        print("guess:", m_n)
        print(f_m_n)
        if f(a_n)*f_m_n < 0:
            a_n = a_n
            b_n = m_n
        elif f(b_n)*f_m_n < 0:
            print("*")
            a_n = m_n
            b_n = b_n
        elif f_m_n == 0:
            print("Found exact solution.")
            return m_n
        else:
            print("Bisection method fails.")
            return None
    return (a_n + b_n)/2


# Build relationship between alpha and mu
def get_mu(alpha, alpha_0):
    """EQ C-17a."""
    den = sp.sin(2*alpha_star - alpha_0)**2 * sp.sin(alpha_0)**2
    mu = sp.symbols('mu')
    num = sp.sin(2*alpha_star - alpha)**2 * sp.sin(alpha)**2
    f = mu*(sp.ln(mu)/(1 - mu))**2 - num/den
    bounds = (1e-12, 1-1e-6)
    return sp.nsolve(f, bounds, solver='bisect', verify=False)


def calc_x0(alpha_0, alpha_to_mu=None):
    """Create lookup table for alpha <--> x0."""

    def integrand(alpha):
        # Eq C-11
        mu = alpha_to_mu(alpha)
        return sp.csc(alpha)**2 * (sp.sympify(1) + mu)/(sp.sympify(1) - mu)

    alpha_list = np.linspace(float(alpha_0 + tol), alpha_L, RES)
    x0_prime_list = [integrand(alpha) for alpha in alpha_list]

    return (a/2) * integrate.cumtrapz(x0_prime_list, alpha_list, initial=0)


def calc_eta(x0, y):
    alpha = float(x0_to_alpha(x0))
    mu = alpha_to_mu(alpha)
    return sp.csc(alpha) * (-y + (a/2) * ((1 + mu)/(1 - mu)))


@np.vectorize
def x0_to_x(x0, y, alpha=None):
    alpha = float(x0_to_alpha(x0)) if alpha is None else alpha
    x = x0 + np.cos(alpha) * (calc_eta(x0, 0) - calc_eta(x0, y))
    return x, y


@np.vectorize
def strain_x0(x0, y):
    alpha = float(x0_to_alpha(x0))
    mu = float(alpha_to_mu(alpha))
    num = u_hat * sp.sin(2*alpha_star - alpha) * sp.sin(alpha)
    den = (y - (a/2) * ((1 + mu)/(1 - mu))) * sp.ln(mu)
    x, y = x0_to_x(x0, y, alpha=alpha)
    return x, y, float(num/den)



def calculate_phi():
    '''total strain energy'''

    def integrand_(alpha):
        # mu = get_mu(alpha)
        mu = float(alpha_to_mu(alpha))
        return sp.sin(2*alpha_star - alpha)/(-sp.log(mu))
    
    return E * t * u_hat**2  * integrate.romberg(integrand_, a0_tol, alpha_L)


#%%

if __name__=="__main__":
    


    # alpha_0 = sp.pi/3.495  # works with  a0_tol = float(alpha_0) #+ 1.9e-3
    # alpha_0 = 0.90280065  # sp.pi/3.485 # works with  a0_tol = float(alpha_0) #+ 1.9e-3
    # a0_tol = float(alpha_0) + 1e-3
    # alpha_0 = sp.pi/3.485 # works with  a0_tol = float(alpha_0) #+ 1.9e-3
    # alpha_0 = .885

    print(float(alpha_star))
    print('*******')
    
    # assert float(alpha_star) < float(alpha_0)
    # alpha_0 guesses for u = 5
    guess_low = 0.886
    guess_high  = 0.925
    x_guessed = []
    N = 15
    # alpha_0 = bisection(f, guess_low, guess_high, N)
    '''
    1
    guess: 0.9055
    -4.989948839675293
    2
    guess: 0.89575
    8.194347824161042
    *
    3
    guess: 0.900625
    0.7156305493209345
    *
    4
    guess: 0.9030625
    -2.303824969585449
    5
    guess: 0.90184375
    -0.8409497149731493
    6
    guess: 0.901234375
    -0.07513465274858788
    7
    guess: 0.9009296875
    0.31702559460740076
    *
    8
    guess: 0.9010820312500001
    0.12015323564114055
    *
    9
    guess: 0.9011582031250001
    0.022312865275736726
    *
    10
    guess: 0.9011962890625
    -0.026459798423871916
    11
    guess: 0.90117724609375
    -0.0020857178823519007
    12
    guess: 0.901167724609375
    0.010110507721179829
    *
    13
    guess: 0.9011724853515626
    0.004011628822979674
    *
    14
    guess: 0.9011748657226564
    0.0009627639914384645
    *
    15
    guess: 0.9011760559082032
    -0.0005615248033734588
    '''
    alpha_0 = 0.9011754608154298
    # 0.9011754608154298 for u =5, N=15 itr
    a0_tol = alpha_0 + tol
    #%%
    a_list = np.linspace(a0_tol, alpha_L, RES)
    mu_list = [get_mu(alpha, alpha_0) for alpha in a_list]
    alpha_to_mu = interpolate.interp1d(a_list, mu_list, fill_value="extrapolate")
    
    fig, ax = plt.subplots()
    ax.plot(a_list, mu_list)
    ax.set_xlabel('alpha')
    ax.set_ylabel('mu')
    ax.axvline(alpha_L)
    ax.axhline(1)
    
    # %%
    x0_list = calc_x0(alpha_0, alpha_to_mu)
    alpha_to_x0 = interpolate.interp1d(np.linspace(float(a0_tol), alpha_L, RES),
                                       x0_list,
                                       fill_value="extrapolate")
    x0_to_alpha = interpolate.interp1d(x0_list,
                                       np.linspace(float(a0_tol), alpha_L, RES),
                                       fill_value="extrapolate")
    
    
    # %%
    # this takes a while
    x0 = np.linspace(0, 100, 100)
    y = np.linspace(-50, 50, 50)
    X0, Y = np.meshgrid(x0, y)
    XS, YS, ES = strain_x0(X0, Y)
    
    E_X = np.vstack((-XS, XS))
    E_Y = np.vstack((-YS, YS))
    ES = np.vstack((ES, ES))
    #%%
    x0 = np.linspace(0, 100, 25)
    y = np.linspace(-50, 50, 50)
    X0_course, Y_course = np.meshgrid(x0, y)
    XS_course, YS_course = x0_to_x(X0_course, Y_course)
    RAYS_X = np.vstack((-XS_course, XS_course))
    RAYS_Y = np.vstack((-YS_course, YS_course))
    

    #%%
    out_path = '../submission/figures/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for i in range(2):
        size = [6.5, 3.25] if i == 0 else [3.25, 1.75]
        fig, ax = plt.subplots(figsize=size)
        n = 256  # number of colors to use
        e_max = 0.06  # cutoff for strains in colormap
        levels = np.linspace(0, e_max, n+1)

        cs = ax.contourf(E_X, E_Y, ES,
                         cmap='coolwarm', levels=levels, extend='max')
        cs.cmap.set_over('pink')
        cs.set_clim(0, e_max)
        

        cb = plt.colorbar(cs, format='%.3f', label='Strain', shrink=0.8, pad=0.01)

        from matplotlib import ticker
        tick_locator = ticker.MaxNLocator(nbins=5)
        cb.locator = tick_locator
        cb.update_ticks()

        ax.vlines(0, -60, 60, 'k', ls='--', lw=0.25)
        ax.hlines(0, -110, 110, 'k', ls='--', lw=0.25)
        ax.plot(np.linspace(-45, 45, 100),
                np.tan(alpha_0)*np.linspace(-45, 45, 100),
                'k', '--', lw=0.25)
        ax.text(5, 1, r'$\alpha_0 = $' + f'{np.rad2deg(alpha_0):.3f}',
                fontsize=8)
        ax.plot(RAYS_X, RAYS_Y, 'k', lw=0.2)

        ax.hlines(-50, -100, 100, 'k', lw=0.5)
        ax.hlines(50, -100, 100, 'k', lw=0.5)
        ax.vlines(-100, -50, 50, 'k', lw=0.5)
        ax.vlines(100, -50, 50, 'k', lw=0.5)
        plt.axis('equal')
        plt.axis('off')
        plt.tight_layout()

        name = 'large' if i == 0 else 'small'
        plt.savefig(out_path+f'mansfield_solution_{name}.pdf', dpi=600)
    #%%
    data = {'rays_x': RAYS_X,
            'rays_y': RAYS_Y,
            'E_x': E_X,
            'E_y': E_Y,
            'E_vals': ES}
    if not os.path.exists('mansfield/'):
            os.makedirs('mansfield/')
    for key in data.keys():
        with open(f'mansfield/{key}.npy', 'wb') as f:
            np.save(f, data[key])

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


