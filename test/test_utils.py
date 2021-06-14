#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 15:03:21 2021

@author: alexanderniewiarowski
"""

import unittest
from dolfin import UnitSquareMesh, as_matrix, sqrt, assemble, inner, dx
from wrinkle.utils import as_block_matrix
import numpy as np


class TestBlockMatrix(unittest.TestCase):

    def setUp(self):
        self.mesh = UnitSquareMesh(2, 2)
        self.M = as_matrix([[1, 2],
                            [3, 4]])
        self.F = as_matrix([[11, 12],
                            [13, 14],
                            [20, 30]])

    def test_non_symmetric_matrix(self):
        # Test with non-symmetric matrix:
        M = self.M
        test = as_block_matrix([[M, -M],
                                [-M, M.T]])

        desired = as_matrix([[M[0, 0], M[0, 1],   -M[0, 0], -M[0, 1]],
                             [M[1, 0], M[1, 1],   -M[1, 0], -M[1, 1]],

                             [-M[0, 0], -M[0, 1],  M[0, 0], M[1, 0]],
                             [-M[1, 0], -M[1, 1],  M[0, 1], M[1, 1]]])
        err = test - desired
        self.assertEqual(sqrt(assemble(inner(err, err)*dx(domain=self.mesh))),
                         0, 'The arrays are not equal!')

    def test_non_square_matrix(self):
        # Test with non-square matrix:
        from dolfin import VectorFunctionSpace, Function, grad
        V = VectorFunctionSpace(self.mesh, 'CG', 1, dim=3)
        u = Function(V)
        u.vector()[:] = np.random.normal(size=V.dim())
        F = grad(u)
        test = as_block_matrix([[F, -F],
                                [-F, F]])
        desired = as_matrix([[F[0, 0], F[0, 1],   -F[0, 0], -F[0, 1]],
                             [F[1, 0], F[1, 1],   -F[1, 0], -F[1, 1]],
                             [F[2, 0], F[2, 1],   -F[2, 0], -F[2, 1]],

                             [-F[0, 0], -F[0, 1],  F[0, 0], F[0, 1]],
                             [-F[1, 0], -F[1, 1],  F[1, 0], F[1, 1]],
                             [-F[2, 0], -F[2, 1],  F[2, 0], F[2, 1]]])
        err = test - desired
        self.assertEqual(sqrt(assemble(inner(err, err)*dx(domain=self.mesh))),
                         0, 'The arrays are not equal!')

    def test_variable_size_matrix(self):
        I3 = -as_matrix([[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]])
        M = self.M
        F = self.F
        test = as_block_matrix([[M, -F.T],
                                [-F, I3]])
        desired = as_matrix([
            [M[0, 0], M[0, 1],    -F[0, 0], -F[1, 0], -F[2, 0]],
            [M[1, 0], M[1, 1],    -F[0, 1], -F[1, 1], -F[2, 1]],

            [-F[0, 0], -F[0, 1],  I3[0, 0], I3[0, 1], I3[0, 2]],
            [-F[1, 0], -F[1, 1],   I3[1, 0], I3[1, 1], I3[1, 2]],
            [-F[2, 0], -F[2, 1],   I3[2, 0], I3[2, 1], I3[2, 2]]])
        err = test - desired
        self.assertEqual(sqrt(assemble(inner(err, err)*dx(domain=self.mesh))),
                         0, 'The arrays are not equal!')

    def test_block_to_vec(self):
        from wrinkle.utils import to_vect
        I3 = -as_matrix([[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]])
        M = self.M
        F = self.F
        test = to_vect(as_block_matrix([[M, -F.T],
                                        [-F, I3]]))
        from dolfin import as_vector
        desired = as_vector([M[0, 0], M[1, 1], I3[0, 0], I3[1, 1], I3[2, 2],
                             M[0, 1],  -F[0, 1], I3[0, 1], I3[1, 2],
                             -F[0, 0],  -F[1, 1], I3[0, 2],
                             -F[1, 0], -F[2, 1],
                             -F[2, 0]])

        err = test - desired
        from dolfin import VectorFunctionSpace, project
        V = VectorFunctionSpace(self.mesh, 'CG', 1, dim=15)
        des = project(desired, V)(.5, .5)
        act = project(test, V)(.5, .5)
        errmsg = f'The arrays are not equal! Desired={print(des)}, \n \
                    actual={print(act)}'
        self.assertEqual(sqrt(assemble(inner(err, err)*dx(domain=self.mesh))),
                         0, errmsg)


if __name__ == "__main__":
    unittest.main()
