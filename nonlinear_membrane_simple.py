#!/usr/bin/env python
# coding: utf-8

# In[1]:


from dolfin import *
import ufl
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from fenicsmembranes.parametric_membrane import *
from fenics_optim import *
from fenics_optim.quadratic_cones import get_slice



width = 200
height = 100
t = 0.025
nsd = 3  # spatial dimensions

E_ = 3500
nu = 0.31

mu = E_/2/(1+nu)
lamb = E_*nu/(1+nu)/(1-2*nu)


class Geometry:
    def __init__(self):
        xi_1, xi_2, w, h = sp.symbols('x[0], x[1], w, h')
        
        self.mesh = mesh = RectangleMesh(Point(0,0), Point(width, height), 80, 40)
        if nsd==3:
            gamma_sp = [xi_1, xi_2, 0*xi_1]
        if nsd==2:
            gamma_sp = [xi_1, xi_2]

        ccode = lambda z: sp.printing.ccode(z)
        self.gamma = Expression([ccode(val) for val in gamma_sp],
                            w=width,
                            h=height,
                            degree=4)
        
        # Get the covariant tangent basis
        # G_1 = ∂X/xi^1 = [1, 0, 0]
        self.Gsub1 = Expression([ccode(val.diff(xi_1)) for val in gamma_sp],
                            w=width,
                            h=height,
                            degree=4)
        
        # G_2 = ∂X/xi^2 = [0, 1, 0]
        self.Gsub2 = Expression([ccode(val.diff(xi_2)) for val in gamma_sp],
                            w=width,
                            h=height,
                            degree=4)
geo = Geometry()

def bottom(x, on_boundary):
    return near(x[1], 0) and on_boundary
def top(x, on_boundary):
    return near(x[1], height) and on_boundary


def bcs(self):
    if nsd==3:
        bc = [DirichletBC(self.V, Constant((0., 0,0)), bottom),
              DirichletBC(self.V, Constant((5, 1,0)), top)]
    
    if nsd==2:
        bc = [DirichletBC(self.V, Constant((0., 0)), bottom),
              DirichletBC(self.V, Constant((5, 1)), top)]
    return bc

input_dict = {
        'mesh': geo.mesh,
        'geometry': geo,
        'thickness': t,
        'material': 'Incompressible NeoHookean',
        'mu': mu,
        'cylindrical': True,
        'output_file_path': 'conic_INH',
        'pressure': 0,
        'Boundary Conditions': bcs}

mem = membrane = ParametricMembrane((input_dict))


# class INHMembrane(ConvexFunction):
#     '''
#     Incompressible neohookean material
#     psi = (mu/2)(tr(C) + 1/det(C) - 3)
#     '''
#     def conic_repr(self, X):
#         # d>= 1/s^2
#         s = self.add_var(dim=3, cone=Pow(3,1/3))
#         d = s[0]

#         # C11*C22 >= C12^2 + s^2
#         r = self.add_var(dim=4, cone=RQuad(4))
        
#         C11 = 2*r[0]
#         C22 = r[1]
#         C12 = r[2]
        
#         # [X, s, r]
#         self.add_eq_constraint([None, s[2], None], b=1)
#         self.add_eq_constraint([None, -s[1], r[3]], b=0)
#         self.set_linear_term([None, (t*mu/2)*s[0], (t*mu/2)*(C11 + C22)])

#         vect_Z = self.add_var(dim=10, cone=SDP(4))
        
#         vect_grad_u_transp = as_vector([0, 0, 0, 0, 0, X[2], 0, X[0], X[1], X[3]])
#         vect_Cnel = as_vector([-C11, -C22, 0, 0, -C12, 0, 0, 0, 0, 0])
#         vect_I = as_vector([0, 0, 1, 1, 0, 0, 0, -1, -1, 0])
    
#         self.add_eq_constraint([vect_grad_u_transp, None, vect_Cnel, vect_Z], b=vect_I)

class INHMembrane3D(ConvexFunction):
    '''
    Incompressible neohookean material
    psi = (mu/2)(tr(C) + 1/det(C) - 3)
    '''
    def conic_repr(self, X):
        Gsub1=geo.Gsub1
        Gsub2 =geo.Gsub2
        
        # d>= 1/s^2
        s = self.add_var(dim=3, cone=Pow(3,1/3))
        d = s[0]

        # C11*C22 >= C12^2 + s^2
        r = self.add_var(dim=4, cone=RQuad(4))
        
        C11 = 2*r[0]
        C22 = r[1]
        C12 = r[2]
        self.C_elastic = as_tensor([[C11,C12], [C12,C22]])
        
        # [X, s, r]
        self.add_eq_constraint([None, s[2], None], b=1)
        self.add_eq_constraint([None, -s[1], r[3]], b=0)
        self.set_linear_term([None, (t*mu/2)*s[0], (t*mu/2)*(C11 + C22)])

        if nsd==2:
            vect_Z = self.add_var(dim=10, cone=SDP(4))
        
            vect_grad_u_transp = as_vector([0, 0, 0, 0, 0, X[2], 0, X[0], X[1], X[3]])
            vect_Cnel = as_vector([-C11, -C22, 0, 0, -C12, 0, 0, 0, 0, 0])
            vect_I = as_vector([0, 0, 1, 1, 0, 0, 0, -1, -1, 0])
            
        if nsd==3:
            vect_Z = self.add_var(dim=15, cone=SDP(5))
            
            vect_grad_u_transp = as_vector([0, 0, 0, 0, 0,
                                            0, X[3], 0, 0, 
                                            X[0], X[4], 0, 
                                            X[1], X[5],
                                            X[2]])
            vect_Cnel = as_vector([-C11, -C22, 0, 0, 0, 
                                   -C12, 0, 0, 0, 
                                   0, 0, 0, 
                                   0, 0, 
                                   0])
            vect_I = as_vector([0, 0, 1, 1, 1,
                                0, -Gsub2[0], 0, 0,
                                -Gsub1[0], -Gsub2[1], 0, 
                                -Gsub1[1], -Gsub2[2],
                                -Gsub1[2]])
        
        self.add_eq_constraint([vect_grad_u_transp, None, vect_Cnel, vect_Z], b=vect_I)
# In[ ]:


# # Solving the Incompressible Neohookean
# prob = MosekProblem("No-compression membrane model")
# membrane.u = prob.add_var(V, bc=bc)


# G = as_vector([u[0].dx(0), u[1].dx(1), u[0].dx(1), u[1].dx(0)])

# energy = INHMembrane(G, degree=2)
# prob.add_convex_term(energy)
# prob.parameters["presolve"] = True
# prob.optimize()


# In[42]:


# Solving the Incompressible Neohookean
prob = MosekProblem("No-compression membrane model")

u__ = prob.add_var(membrane.V, bc=membrane.bc)
prob.var[0] = membrane.u   # replace
u = membrane.u
G = as_vector([u[0].dx(0), u[1].dx(0), u[2].dx(0), \
               u[0].dx(1), u[1].dx(1), u[2].dx(1)])

energy = INHMembrane3D(G, degree=2)
prob.add_convex_term(energy)
prob.parameters["presolve"] = True
prob.optimize()
membrane.io.write_fields()



# In[46]:


def eigenvalue(A):
    ''' ufl eigenvalues of 2x2 tensor '''
    assert A.ufl_shape == (2,2)
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
    tol = DOLFIN_EPS_LARGE#*1000
    print(tol)
    ''' eigenvectors of 2x2 tensor 
    if c != 0, [l1-d,c], [l2-d,c]
    elif b != 0, [b, l1-a], [b, l2-a]
    else [1,0], [0,1]
    '''
    assert A.ufl_shape == (2,2)
    l1, l2 = eigenvalue(A)    
    # print(project(l1, mem.Vs)(pt))
    # print(project(l2, mem.Vs)(pt))
    a = A[0,0] 
    b = A[0,1] 
    c = A[1,0] 
    d = A[1,1]
    
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
    v1 = v1/sqrt(dot(v1,v1))
    v2 = v2/sqrt(dot(v2,v2))

    return v1, v2


# In[49]:

E = membrane.E
E_el = 0.5*(energy.C_elastic - membrane.C_0)

E_w = -(E - E_el)



def write_eigval(A, l1_name, l2_name):
    l1, l2 = eigenvalue(A)
    
    l1 = project(l1, FunctionSpace(mem.mesh, 'CG',1), form_compiler_parameters={"quadrature_degree": energy.degree})
    l1.rename(l1_name, l1_name)
    
    l2 = project(l2, FunctionSpace(mem.mesh, 'CG',1), form_compiler_parameters={"quadrature_degree": energy.degree})
    l2.rename(l2_name, l2_name)
    
    mem.io.add_extra_output_function(l1)
    mem.io.add_extra_output_function(l2)
    mem.io.write_fields()
    


def write_eigvec(A, v1_name, v2_name):
    v1, v2 = eig_vecmat(A)
    S = VectorFunctionSpace(mem.mesh, 'CG', 1)
    eig1 = project(v1, S, form_compiler_parameters={"quadrature_degree": energy.degree})
    eig1.rename(v1_name, v1_name)
    eig2 = project(v2, S, form_compiler_parameters={"quadrature_degree": energy.degree})
    eig2.rename(v2_name, v2_name)
    mem.io.add_extra_output_function(eig1)
    mem.io.add_extra_output_function(eig2)
    mem.io.write_fields()



write_eigval(E_w, 'Ew1', 'Ew2')
write_eigvec(E_w, 'Ew_v1', 'Ew_v2')

write_eigval(E_el, 'E_el1', 'E_el2')
write_eigvec(E_el, 'E_el_eig1', 'E_el_eig2')

# ew1, ew2 = eigenvalue(E_w)
# ew1 = project(ew1, FunctionSpace(mem.mesh, 'CG',1), form_compiler_parameters={"quadrature_degree": energy.degree})
# ew1.rename('ew1', 'ew1')
# ew2 = project(ew2, FunctionSpace(mem.mesh, 'CG',1), form_compiler_parameters={"quadrature_degree": energy.degree})
# ew2.rename('ew2', 'ew2')
# mem.io.add_extra_output_function(ew1)
# mem.io.add_extra_output_function(ew2)
# mem.io.write_fields()

#%%


def test_eig(A):
    l1, l2 = eigenvalue(A)
    v1, v2 = eig_vecmat(A)
    V = FunctionSpace(mem.mesh, 'CG',1)
    S = VectorFunctionSpace(mem.mesh, 'CG', 1)
    print("The eigenvalues are:", project(l1, V)(10,10), project(l2,V)(10,10))
    print("The eigenvectors are:", project(v1, S)(10,10), project(v2, S)(10,10))
    
test_eig(2*Identity(2))
theta=pi/4
R = as_tensor([[cos(theta), -sin(theta)], 
               [sin(theta), cos(theta)]])
I = as_tensor([[2,0], [0,1]])
test = R*I*R.T
test_eig(test)

#%%

# J = det(mem.C_n)/det(mem.C_0)  # not correct, using the total  deformation 
J = det(energy.C_elastic)/det(mem.C_0)

import ufl
i,j = ufl.indices(2)
gsup = mem.get_metric(mem.gsup1, mem.gsup2)
gsup = inv(energy.C_elastic)
sigma = as_tensor(mu/J*(mem.C_0_sup[i,j] - 1/J**2*gsup[i,j]), [i,j] )

sigma = S = as_tensor(mu*(mem.C_0_sup[i,j] - 1/J*gsup[i,j]), [i,j] )
# sigma = mem.F_n*S*mem.F_n.T
write_eigval(sigma, 'sigma1', 'sigma2')
write_eigvec(sigma, 'sigma1v', 'sigma2v')
# In[50]:


# ew1,ew2 = eigenvalue(Ew)

# ew1=sqrt(2*ew1+1)
# ew2=sqrt(2*ew2+1)
# lambdaew1 = project(ew1,FunctionSpace(mesh, 'CG',1), form_compiler_parameters={"quadrature_degree": energy.degree})
# lambdaew2=project(ew2,FunctionSpace(mesh, 'CG',1), form_compiler_parameters={"quadrature_degree": energy.degree})
# pl = plot(lambdaew1)
# plt.colorbar(pl)
# plt.show()



# In[51]:

# F = mem.F
# B0 = mem.F_0*mem.F_0.T
# Bn = mem.F_n*mem.F_n.T
# B = inv(B0)*Bn
# Jinv = det(mem.C_0)/det(mem.C_n)
# sigma = mu*Jinv*(B-mem.C_0*(Jinv**2))

# In[52]:


# # Solving the Incompressible Neohookean
# prob = MosekProblem("No-compression membrane model")
# u = prob.add_var(V, bc=bc)

# G = as_vector([u[0].dx(0), u[1].dx(1), u[0].dx(1), u[1].dx(0)])

# energy = INHMembrane(G, degree=2)
# prob.add_convex_term(energy)
# prob.parameters["presolve"] = True
# prob.optimize()


# In[53]:

# 2PK S
gsup = inv(energy.C_elastic)
S = as_tensor(mu*(mem.C_0_sup[i,j] - 1/J*gsup[i,j]), [i,j] )
s1, s2 = eigenvalue(S)
sv1, sv2 = eig_vecmat(S)

# drop z comp of Fn
Fn = as_tensor([[mem.gsub1[0], mem.gsub1[1]],[mem.gsub2[0], mem.gsub2[1]]]).T

sigma_el = (s1/J)*outer(dot(Fn,sv1), dot(Fn,sv1))

write_eigval(sigma_el, 'sigma1_el', 'sigma2_el')
write_eigvec(sigma_el, 'sigma1v_el', 'sigma2v_el')
# In[ ]:

K = derivative(mem.DPi_int(), mem.u, mem.du)
residual = action(K, mem.u) 
V = mem.V
v_reac = Function(V)
bcRx = DirichletBC(V.sub(0), Constant(1.), bottom)

bcRx.apply(v_reac.vector())
print("Horizontal reaction Rx = {}".format(assemble(action(residual, v_reac))))



