#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 14:25:40 2021

@author: alexanderniewiarowski
"""

from dolfin import tr, det, DOLFIN_EPS_LARGE, conditional, sqrt, gt, as_vector, dot, as_matrix
from ufl import shape


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


def to_vect(X):
    """ Transforms symmetric tensor into vector by spanning upper diagonals """
    s = shape(X)
    if len(s)==2 and s[0] == s[1]:
        d = s[0]
        return as_vector([X[i, i+k]
                          for k in range(d) for i in range(d-k)])
    else:
        raise(ValueError, "Variable must be a square tensor")


def as_block_matrix(mlist):
    rows = []
    # Block rows:
    for i in range(0, len(mlist)):
        block_row_height = shape(mlist[i][0])[0]
        # Matrix rows in block row i:
        for j in range(0, block_row_height):
            row = []
            # Block columns:
            for k in range(0, len(mlist[i])):
                block_column_width = shape(mlist[i][k])[1]
                # Matrix columns in block column k:
                for l in range(0, block_column_width):
                    row += [mlist[i][k][j, l], ]
            rows += [row, ]
    return as_matrix(rows)


# def diagonalize(A):
#     '''fenics_optim.to_vect spans lower diagonal'''
#     ROW, COL = np.shape(A)
#     vec = []
#     for j in range(COL):
#         for i in range(ROW - j):
#             vec.append(A[i, i+j])
#     return vec

# import numpy as np
# test = [[0, 1, 2, 3],
#         [4, 5, 6, 7],
#         [7, 8, 9, 10],
#         [11, 12, 13, 14]]
# print(diagonalize(np.array(test)))
# print(to_vect(as_matrix(test)))