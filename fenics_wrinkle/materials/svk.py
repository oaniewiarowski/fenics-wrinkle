#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 10:19:49 2021

@author: alexanderniewiarowski
"""

from dolfin import (as_vector, as_matrix, as_tensor, inv,
                    TensorFunctionSpace, project, UnitSquareMesh,
                    dot, Identity, grad, Constant)
import numpy as np

from fenics_optim import ConvexFunction
from fenics_optim.quadratic_cones import RQuad, SDP, get_slice
from fenics_wrinkle.utils import block_to_vect, as_block_matrix
from ufl import zero as O
from ufl import Index



# class _NonLinearMembrane(ConvexFunction):
#     def conic_repr(self, u, mem):
#         Gsub1 = mem.Gsub1
#         Gsub2 = mem.Gsub2
#         C0 = mem.C_0
#         t = float(mem.thickness)


#         E_ = 3500
#         nu = 0.31
#         mu = E_/2/(1+nu)
#         lamb = E_*nu/(1+nu)/(1-2*nu)
#         C = t*np.array([[lamb+2*mu, lamb,      0    ],
#                       [lamb,      lamb+2*mu, 0    ],
#                       [0,         0,         2*mu]])
#         # C = Q^T*Q
#         Q = np.linalg.cholesky(C)
#         Qinv = as_matrix(np.linalg.inv(Q))



#         # y := [z0, z1, e_11, e_22, e_12] = [ ]
#         y = self.add_var(dim=5, cone=RQuad(5), name="y")

#         # elastic strain
#         # Q*e_e - y = 0
#         Eel = dot(Qinv.T, get_slice(y, 2))  # => [y2, y3, y4]
#         '''
#         # shouldn't Eel be?:
#         y_bar = as_vector([y[2], y[3], 2*y[4]])
#         Eel = dot(Q.T, y_bar)
#         '''
#         self.add_eq_constraint([None, y[1]], b=1)  # dummy variable for Mosek

#         vect_Z = self.add_var(dim=15, cone=SDP(5))

#         vect_grad_u_transp = as_vector([0, 0, 0, 0, 0,
#                                         0, X[3], 0, 0,
#                                         X[0], X[4], 0,
#                                         X[1], X[5],
#                                         X[2]])
#         vect_Eel = as_vector([-2*Eel[0], -2*Eel[1], 0, 0, 0,
#                               -2*Eel[2], 0, 0, 0,
#                               0, 0, 0,
#                               0, 0,
#                               0])
#         vect_I = as_vector([C0[0,0], C0[1,1], 1, 1, 1,
#                             C0[0,1], -Gsub2[0], 0, 0,
#                             -Gsub1[0], -Gsub2[1], 0,
#                             -Gsub1[1], -Gsub2[2],
#                             -Gsub1[2]])

#         # e - e_e + e_w = 0 ??? (confused about the signs here)
#         self.add_eq_constraint([vect_grad_u_transp, vect_Eel, vect_Z], b=vect_I)  # b=0
#         self.set_linear_term([None, y[0]])  #  y_0 = w_ref
# class NonLinearMembrane(ConvexFunction):
#     def conic_repr(self, u):

#         # X = as_vector([u[0].dx(0), u[1].dx(0), u[2].dx(0), \
#         #            u[0].dx(1), u[1].dx(1), u[2].dx(1)])
#         y = self.add_var(dim=5, cone=RQuad(5), name="y")
       
#         Eel = dot((Qinv.T), get_slice(y, 2))  # => [y2, y3, y4] 
#         self.add_eq_constraint([None, y[1]], b=1)
#         self.set_linear_term([None, y[0]])      

#         vect_Z = self.add_var(dim=15, cone=SDP(5))

#         # vect_grad_u_transp = as_vector([0, 0, 0, 0, 0,
#         #                                 0, X[3], 0, 0,
#         #                                 X[0], X[4], 0,
#         #                                 X[1], X[5],
#         #                                 X[2]])
#         vect_grad_u_transp = as_vector([0, 0, 0, 0, 0,
#                                         0, u[0].dx(1), 0, 0,
#                                         u[0].dx(0), u[1].dx(1), 0,
#                                         u[1].dx(0), u[2].dx(1),
#                                         u[2].dx(0)])
#         vect_Eel = as_vector([-2*Eel[0], -2*Eel[1], 0, 0, 0,
#                               -2*Eel[2], 0, 0, 0,
#                               0, 0, 0,
#                               0, 0,
#                               0])
#         vect_I = as_vector([C0[0,0], C0[1,1], 1, 1, 1,
#                             C0[0,1], -Gsub2[0], 0, 0,
#                             -Gsub1[0], -Gsub2[1], 0,
#                             -Gsub1[1], -Gsub2[2],
#                             -Gsub1[2]])
        
#         # vect_I = as_vector([200*200, 100*100, 1, 1, 1,
#         #                     0, 0, 0, 0,
#         #                     -200, -100, 0,
#         #                     0, 0,
#         #                     0])

#         # e - e_e + e_w = 0 ??? (confused about the signs here)
#         self.add_eq_constraint([vect_grad_u_transp, vect_Eel, vect_Z], b=vect_I)  # b=0



# class NonLinearMembrane2D(ConvexFunction):
#     def conic_repr(self, u):
#         X = as_vector([u[0].dx(0), u[1].dx(1), u[0].dx(1), u[1].dx(0)])
#         y = self.add_var(dim=5, cone=RQuad(5), name="y")
       
#         Eel = dot((Qinv.T), get_slice(y, 2))  # => [y2, y3, y4] 
#         self.add_eq_constraint([None, y[1]], b=1)
#         self.set_linear_term([None, y[0]])      
        
#         vect_Z = self.add_var(dim=10, cone=SDP(4))
#         vect_grad_u_transp = as_vector([0, 0, 0, 0, 0, X[2], 0, X[0], X[1], X[3]])
#         vect_Eel = as_vector([-2*Eel[0], -2*Eel[1], 0, 0, -2*Eel[2], 0, 0, 0, 0, 0])
#         vect_I = as_vector([C0[0,0], C0[1,1], 1, 1, 
#                             0,  -Gsub2[0], 0,  -Gsub1[0], -Gsub2[1], 0])
    
#         self.add_eq_constraint([vect_grad_u_transp, vect_Eel, vect_Z], b=vect_I)
def cmat_contra(A):
    
    rows = []
    for row in [A[0], A[3], A[1]]:
        comp = row.reshape(2,2)
        rows.append([comp[0,0], comp[1,1], comp[0,1] ])
    return np.array(rows)


class SVKMembrane(ConvexFunction):
    def __init__(self, u, mem, lmbda, mu, **kwargs):

        a0_contra = mem.C_0_sup
        i, j, l, m = Index(), Index(), Index(), Index()

        C_contra = as_tensor((((2.0*lmbda*mu)/(lmbda + 2.0*mu))*a0_contra[i,j]*a0_contra[l,m]
                        + 1.0*mu*(a0_contra[i,l]*a0_contra[j,m] + a0_contra[i,m]*a0_contra[j,l]))
                        ,[i,j,l,m])
        A = project(C_contra,
                    TensorFunctionSpace(mem.mesh,
                                        'CG',
                                        1,
                                        (2, 2, 2, 2)))(.5, .5).reshape(4, 4)
        C_contra = cmat_contra(A)

        self.C = C_contra
        self.mem = mem
        ConvexFunction.__init__(self, u, parameters=None, **kwargs)

    def conic_repr(self, u):
        mem = self.mem
        Q = np.linalg.cholesky(self.C)  # lower triangular, np.testing.assert_array_almost_equal(C, np.matmul(Q,Q.T))
        Qinv = Constant((np.linalg.inv(Q)), name='Q')
        y = self.add_var(dim=5, cone=RQuad(5), name="y")
        Eel = dot((Qinv.T), get_slice(y, 2))  # => [y2, y3, y4]

        self.E_el = E_el = as_matrix([[Eel[0], Eel[2]],
                                      [Eel[2], Eel[1]]])
        self.add_eq_constraint([None, y[1]], b=1)
        self.set_linear_term([None, y[0]])

        d = mem.nsd
        C0 = mem.C_0
        F0 = mem.F_0
        # dim = 10 or 15 = top dim + geo dim + 1
        vect_Z = self.add_var(dim=sum(range(2+d+1)), cone=SDP(2+d))

        vect_grad_u_T = block_to_vect([[O((2, 2)), grad(u).T],
                                       [grad(u),  O(d, d)]])

        vect_Eel = block_to_vect([[-2*E_el, O(2, d)],
                                  [O(d, 2), O(d, d)]])

        vect_I = block_to_vect([[C0,  -F0.T],
                                [-F0, Identity(d)]])

        self.add_eq_constraint([vect_grad_u_T, vect_Eel, vect_Z], b=vect_I)
        
        
        self.E_w = -(self.mem.E - self.E_el)


