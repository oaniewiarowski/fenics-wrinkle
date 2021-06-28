#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 23:08:34 2021

@author: alexanderniewiarowski

Currently, the Paraview data is saved manually to csv files
(This can be automated, but installing paraview via conda would have been risky)
** Thickness function is saved as the last timestep so need to fstforward before saving the data!!!
"""
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib import rc_file
# import argparse

rc_file('../submission/journal_rc_file.rc')
#plt.style.use('seaborn-whitegrid')

out_path = '../submission/figures/'
if not os.path.exists(out_path):
    os.makedirs(out_path)


# For convenience 
labels = {
        'KS': r'$J = KS\left( (\lambda_1 - \bar{\lambda})^2\right)$',
        'abs': r'$J = \int ( \lambda_1 - \bar{\lambda} )^2 d\xi$',
        'lsq': r'$\hat{J} = \int ( \lambda_1 - \bar{\lambda} )^2 d\xi$'
        }
#%%
if __name__ == "__main__":
        
    # csv file saved in Paraview. File > Save Data
    fname = 'results/hypar'
    df = pd.read_csv(f'{fname}.csv')

    
    fig, (ax0, ax1) = plt.subplots(1,2, figsize=[6.5, 3])
    
    # ax0.plot(df['Points:0'], df.sigma_1, label=r'$\sigma_1$')
    # ax0.plot(df['Points:0'], df.sigma_2, label=r'$\sigma_2$')
    x = df['arc_length']
    x = x-(x.max()-x.min())/2
    ax0.plot(x, df.sigma_1, label=r'$\sigma_1$')
    ax0.plot(x, df.sigma_2, label=r'$\sigma_2$')


    ax0.set_xlabel(r'$S (mm)$')
    ax0.set_ylabel(r'$\sigma$')
    ax0.legend()
    
    ax1.plot(x, df.E_el1, label=r'$E_el1$')
    ax1.plot(x, df.E_el2, label=r'$E_el2$')
    
    ax1.set_xlabel(r'$S (mm)$')
    ax1.set_ylabel(r'$\sigma$')
    ax1.legend()
    
    ax0.grid(True)
    ax1.grid(True)
    ax0.text(.5, 1.1, '(a)', horizontalalignment='center', verticalalignment='top', transform=ax0.transAxes)
    ax1.text(.5, 1.1, '(b)', horizontalalignment='center', verticalalignment='top', transform=ax1.transAxes)

    plt.tight_layout()
    plt.savefig(out_path+f'hypar_stresses.pdf', dpi=600)

