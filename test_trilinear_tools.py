#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 19:47:47 2021

@author: alexanderniewiarowski
"""
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt


def my_cross(x,c):
    '''x times c'''
    # return c[2]*x[1] - c[1]*x[2], -c[2]*x[0] + c[0]*x[2], c[1]*x[0] - c[0]*x[1]
    a0 = c[2]*x[1]
    a1 = -c[1]*x[2]
    
    b0 = -c[2]*x[0]
    b1 = c[0]*x[2]
    
    c0 = c[1]*x[0]
    c1 = -c[0]*x[1]
    return [[a0, a1], [b0,b1], [c0, c1]]

def my_cross_2(x,u,c):
    '''x +u times c'''
    # return (c2 u1 - c1 u2 + c2 x1 - c1 x2, -c2 u0 + c0 u2 - c2 x0 + c0 x2, c1 u0 - c0 u1 + c1 x0 - c0 x1)
    a0 = c[2]*x[1]
    a1 = -c[1]*x[2]
    a2 = c[2]*u[1]
    a3 = -c[1]*u[2]
    
    b0 = -c[2]*x[0]
    b1 = c[0]*x[2]
    b2 = -c[2]*u[0]
    b3 = c[0]*u[2]
    
    c0 = c[1]*x[0]
    c1 = -c[0]*x[1]
    c2 = c[1]*u[0]
    c3 = -c[0]*u[1]
    return [[a0, a1, a2, a3], [b0,b1,b2,b3], [c0, c1,c2,c3]]


def test_my_cross():
    v1 = as_vector([1,2,3])
    v2 = as_vector([4,5,6])
    
    test = my_cross(v1,v2)
    check = cross(v1,v2)
    res = []
    for i in test:
        res.append(i[0] + i[1])
    res = as_vector(res)
    mesh = UnitSquareMesh(30,30)
    V = VectorFunctionSpace(mesh, 'CG', 2, dim=3)
    np.testing.assert_equal(project(res, V)(0.5,0.5), project(check, V)(0.5,0.5))


def test_my_cross_2():
    p=1.1
    x = as_vector([1,2,3])
    u = as_vector([Constant(p),Constant(p),Constant(p)])
    v2 = as_vector([4,5,6])
    
    test = my_cross_2(x,u,v2)
    v1 = x+u
    check = cross(v1,v2)

    res = []
    for i in test:
        res.append(sum(i))
    res = as_vector(res)
    mesh = UnitSquareMesh(30,30)
    V = VectorFunctionSpace(mesh, 'CG', 2, dim=3)
    np.testing.assert_almost_equal(project(res, V)(0.5,0.5), project(check, V)(0.5,0.5))

def test_my_dot():
    v1 = as_vector([1,2,3])
    v2 = as_vector([4,5,6])
    v3 = my_cross(v1,v2)
    # a = as_vector([11, 12, 13])
    a = as_vector([20, -10, -57])
    test = my_dot(a, v3)
    check = dot(a, cross(v1,v2))
    mesh = UnitSquareMesh(30,30)
    Vs = FunctionSpace(mesh, 'CG', 2)
    np.testing.assert_almost_equal(float(sum(test)), project(check, Vs)(0.5,0.5))

def my_dot(a,b):
    """

    Parameters
    ----------
    a : ufl Function
        DESCRIPTION.
    b : my_cross list
        DESCRIPTION.

    Returns
    -------
    ret_val : TYPE
        DESCRIPTION.

    """    
    ret_val = []
    for i in range(len(a)):
        for j in range(2):
            ret_val.append(a[i]*b[i][j])
    return ret_val



def inverse_grad(f, u, bc):
    V = u.function_space()
    u_ = TrialFunction(V)
    v = TestFunction(V)


    a = inner(grad(u_), grad(v))*dx(V.mesh())
    L = inner(f, grad(v))*dx(V.mesh())
    solve(a==L, u, bc)
    return u


def test_inverse_grad():
    mesh = UnitSquareMesh(30,30)
    V = VectorFunctionSpace(mesh, 'CG', 2, dim=3)
    
    u = interpolate(Expression(('sin(x[0]*pi)*sin(x[1]*pi)', '-sin(x[0]*2*pi)*sin(x[1]*2*pi)', 'sin(x[0]*pi)*sin(x[1]*pi)'), degree=2), V)
    gradu1 = project(u.dx(0), V)
    gradu2 = project(u.dx(1), V)
    f = as_tensor([gradu1, gradu2]).T
    uu = Function(V)
    bnd = CompiledSubDomain("on_boundary")
    bc = DirichletBC(V, Constant((0, 0, 0)), bnd)
    inverse_grad(f, uu, bc)
    mag_u = sqrt(dot(u,u))
    mag_uu = sqrt(dot(uu,uu))
    np.testing.assert_almost_equal(u.compute_vertex_values(), uu.compute_vertex_values(), decimal=4)
    np.testing.assert_almost_equal(norm(u), norm(uu), decimal=5)
    # fig, ax = plt.subplots()
    # plot(mag_u, title='mag_u')
    # fig, ax = plt.subplots()
    # plot(mag_uu, title='mag_uu')        




def test_expanded_ufl():
    p=1.1
    v1 = as_vector([1+Constant(p**2),2+Constant(p**3),3+Constant(p**4)])
    v2 = as_vector([4,5,6])
    check = cross(v1,v2)
    
    # # Expand compound operation cross into basic arithmetic operations.
    # check_expanded = expand_compounds(check)
    
    # # Get monomials for each expanded component of check, and put them in a list.
    # result = []
    # for i in range(0,len(check_expanded)):
    #     result_component = get_monomials(check_expanded[i])
    #     result += [result_component,]
    result = expand_ufl(check)
    
    # Test (modified to use an epsilon):
    for i in range(len(result)):
        np.testing.assert_equal(len(result[i]), 4)  # per above example
    test = as_vector([sum(result[i]) for i in range(0,len(result))])
    mesh = UnitSquareMesh(10,10)
    V = VectorFunctionSpace(mesh, 'CG', 1, dim=3)
    
    epsilon = 1e-10
    np.testing.assert_allclose(project(test, V)(0.5,0.5),
                               project(check, V)(0.5,0.5), epsilon)
    
    
    
def test_expanded_ufl_2():
    p=1.1
    v1 = as_vector([1+Constant(p**2),2+Constant(p**3),3+Constant(p**4)])
    v2 = as_vector([4,5,6])
    c = Constant((57, 89, -pi))
    check = dot(c, cross(v1,v2))
    result = expand_ufl(check)
    
    np.testing.assert_equal(len(result), 3*4)  # per new example
    test = sum(result)
    mesh = UnitSquareMesh(10,10)
    Vs = FunctionSpace(mesh, 'CG', 1)
    np.testing.assert_allclose(project(test, Vs)(0.5,0.5),
                               project(check, Vs)(0.5,0.5), 1e-10)
    

if __name__=="__main__":
    
    test_inverse_grad()
    test_my_cross()
    test_my_cross_2()
    test_my_dot()
    test_expanded_ufl()
    test_expanded_ufl_2()
    # dV1 = dot(u, gsub3)
    # dV1_00 = u[0]*g3_list[0][0]
    # dV1_01 = u[0]*g3_list[0][1]
    
    # dV1_10 = u[1]*g3_list[1][0]
    # dV1_11 = u[1]*g3_list[1][1]
    
    # dV1_20 = u[2]*g3_list[2][0]
    # dV1_21 = u[2]*g3_list[2][1]
    # dV1 = [dV1_00, dV1_01, dV1_10, dV1_11, dV1_20, dV1_21]
    
    # dV2 = dot(mem.gamma, gsub3)
    # dV2_00 = mem.gamma[0]*g3_list[0][0]
    # dV2_01 = mem.gamma[0]*g3_list[0][1]
    
    # dV2_10 = mem.gamma[1]*g3_list[1][0]
    # dV2_11 = mem.gamma[1]*g3_list[1][1]
    
    # dV2_20 = mem.gamma[2]*g3_list[2][0]
    # dV2_21 = mem.gamma[2]*g3_list[2][1]
    # dV2 = [dV2_00, dV2_01, dV2_10, dV2_11, dV2_20, dV2_21]
    
    
    
    # for d in dV1:
    #     prob.add_obj_func(-Constant(p/3)*d*dx(mem.mesh))
    # for d in dV2:
    #     prob.add_obj_func(-Constant(p/3)*d*dx(mem.mesh))