#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 14:25:40 2021

@author: alexanderniewiarowski
"""

from dolfin import tr, det, DOLFIN_EPS_LARGE, conditional, sqrt, gt, as_vector, dot


def eigenvalue(A):
    ''' ufl eigenvalues of 2x2 tensor '''
    assert A.ufl_shape == (2, 2)
    I1 = tr(A)
    I2 = det(A)
    # delta = sqrt(I1**2 - 4*I2)
    Q = 0.25*I1**2 - I2
    delta = conditional(gt(abs(Q), DOLFIN_EPS_LARGE),
                        sqrt(Q), 0)
    l1 = I1/2 + delta
    l2 = I1/2 - delta
    return l1, l2


def eig_vecmat(A):
    tol = DOLFIN_EPS_LARGE
    ''' eigenvectors of 2x2 tensor
    if c != 0, [l1-d,c], [l2-d,c]
    elif b != 0, [b, l1-a], [b, l2-a]
    else [1,0], [0,1]
    '''
    assert A.ufl_shape == (2, 2)
    l1, l2 = eigenvalue(A)
    a = A[0, 0]
    b = A[0, 1]
    c = A[1, 0]
    d = A[1, 1]

    v1 = conditional(gt(abs(c), tol),
                     as_vector([l1-d, c]),
                     conditional(gt(abs(b), tol),
                                 as_vector([b, l1 - a]),
                                 as_vector([1, 0])))
    v2 = conditional(gt(abs(c), tol),
                     as_vector([l2-d, c]),
                     conditional(gt(abs(b), tol),
                                 as_vector([b, l2 - a]),
                                 as_vector([0, 1])))
    v1 = v1/sqrt(dot(v1, v1))
    v2 = v2/sqrt(dot(v2, v2))

    return v1, v2
