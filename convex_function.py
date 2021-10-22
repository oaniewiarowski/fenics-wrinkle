#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convex_function provides classes for the definition of convex function
and their conic representation

@author: Jeremy Bleyer, Ecole des Ponts ParisTech,
Laboratoire Navier (ENPC,IFSTTAR,CNRS UMR 8205)
@email: jeremy.bleyer@enpc.f
"""
from dolfin import *
from fenics_optim import Quad, RQuad
from fenics_optim.utils import to_list, to_vect
import ufl
import numpy as np

class BaseConvexFunction(object):
    """
    Base class for generic convex functions
    """
    def __init__(self, expr, parameters=None, interp_type="quadrature", quadrature_scheme="default", degree=1, measure=Measure("dx")):
        """
        Parameters
        ----------

        x
            the block variable on which acts the convex function
        quadrature_scheme : {"default", "vertex", "lobatto"}
            quadrature scheme used for the integral computation and additional variables
            discretization. "vertex" scheme is used only with degree=1
        degree : int
            quadrature degree
        measure : `Measure`
            measure used for the integral
        """
        self.expr = expr
        self.mesh = self.get_mesh_from_expr(self.expr)
        self.n = FacetNormal(self.mesh)
        self.parameters = parameters
        self.degree = degree
        self.quadrature_scheme = quadrature_scheme
        self.interp_type = interp_type
        metadata = {"quadrature_degree": self.degree, "quadrature_scheme": self.quadrature_scheme, "representation":"quadrature"}
        if isinstance(measure, list):
            self.dx = [m(domain=self.mesh, scheme=self.quadrature_scheme, degree=degree) for m in measure]
        else:
            self.dx = measure(domain=self.mesh, scheme=self.quadrature_scheme, metadata=metadata)
        self.constraints = []
        self.cones = []
        self.additional_variables = []
        self.ux = []
        self.lx = []
        self.c = [None]
        self.scale_factor = 1

    def get_mesh_from_expr(self, expr):
        expr = to_list(expr)
        for e in expr:
            coeffs = ufl.algorithms.analysis.extract_coefficients(e)
            for c in coeffs:
                if hasattr(c, "function_space"):
                    return c.function_space().mesh()
        raise(ValueError, "Unable to extract mesh from UFL expression")

    def declare_var(self, x):
        if isinstance(x, list):
            assert len({ufl.shape(xi) for xi in x}) == 1, \
                    "Non matching shapes for variables list"
            self.dim_x = ufl.shape(x[0])
        else:
            self.dim_x = ufl.shape(x)
        if len(self.dim_x) == 0:
            # X is a scalar variable
            self.dim_x = 0
            tensor = False
        else:
            # X is a vectorial or tensorial variable
            tensor = len(self.dim_x)==2

            self.dim_x = self.dim_x[0]
            if tensor:
                # we take its symmetric part and convert it to a vector
                x = to_vect(ufl.sym(x))
                self.dim_x = ufl.shape(x)[0]

        self.X = Function(self.generate_function_space(self.dim_x))
        self.X_expr = x
        return self.X

    def generate_function_space(self, d, tensor=False):
        pass

    def set_term(self, expr, *args):
        """
        Defines on which expression of :math:`x` operates the convex function.
        A fictitious variable :math:`X` is then substituted to the expression
        when defining the function conic representation.

        Parameters
        ----------

        expr
            expression of the optimization variable :math:`x` on which acts the convex function.
            `expr` must be a linear operation on :math:`x`

        Optional parameters can be defined.
        """
        X = self.declare_var(expr)
        self.conic_repr(X, *args)

    def conic_repr(self, X):
        pass

    def add_var(self, dim=0, cone=None, ux=None, lx=None, name=None):
        """
        Adds a (list of) auxiliary optimization variable. These variables are local and
        their interpolation is defined through the chosen quadrature scheme.
        They are added to the block structure of the optimization problem by
        following their order of declaration. Inside a ConvexFunction, the block
        structure is :math:`z=[X, Y_0, Y_1, \ldots, Y_n]` where :math:`X` is the
        global declared variable and each :math:`Y_i` are the additional variables.

        Parameters
        ----------

        dim : int, list of int
            dimension of each variable (0 for a scalar)
        cone : `Cone`, list of `Cone`
            cone in which each variable belongs (None if no constraint)
        ux : float, Function
            upper bound on variable :math:`x\leq u_x`
        lx : float, Function
            lower bound on variable :math:`x\leq l_x`
        name : str
            variable name
        """
        dim_list = to_list(dim)
        nlist = len(dim_list)
        new_V_add_var = [self.generate_function_space(d) for d in dim_list]
        self.cones += to_list(cone, nlist)
        self.ux += to_list(ux, nlist)
        self.lx += to_list(lx, nlist)
        if nlist == 1:
            new_Y = Function(new_V_add_var[0], name=name)
            self.additional_variables.append(new_Y)
            return new_Y
        else:
            new_Y = [Function(v, name=n) for (v, n) in zip(new_V_add_var, to_list(name, nlist))]
            self.additional_variables += to_list(new_Y, nlist)
            return tuple(new_Y)

    def add_global_var(self, V, cone=None, ux=None, lx=None):
        """
        Adds a (list of) global optimization variable. These variables are added to the
        block structure of the optimization problem by following their order of declaration.

        Parameters
        ----------

        V : (list of) `FunctionSpace`
            variable FunctionSpace
        cone : `Cone`, list of `Cone`
            cone in which each variable belongs (None if no constraint)
        ux : float, Function
            upper bound on variable :math:`x\leq u_x`
        lx : float, Function
            lower bound on variable :math:`l_x\leq x`
        """
        if isinstance(V, list):
            nlist = len(V)
        else:
            nlist = 1
            V = [V]
        self.cones += to_list(cone, nlist)
        self.ux += to_list(ux, nlist)
        self.lx += to_list(lx, nlist)
        if nlist == 1:
            new_Y = Function(V[0])
        else:
            new_Y = [Function(v) for v in V]
        self.additional_variables += to_list(new_Y, nlist)
        return new_Y

    def add_eq_constraint(self, Az, b=0):
        """
        Adds an equality constraint :math:`Az=b` where
        :math:`z=[X, Y_0, \ldots, Y_n]`

        Parameters
        ----------

        Az : list
            list of linear expressions of :math:`z`-blocks. Use 0 or None for an empty block.
        b : float, expression
            corresponding right-hand side
        """
        self.add_ineq_constraint(Az, b, b)

    def add_ineq_constraint(self, Az, bu=None, bl=None):
        """
        Adds an inequality constraint :math:`b_l \leq Az \leq b_u` where
        :math:`z=[X, Y_0, \ldots, Y_n]`

        Parameters
        ----------

        Az : list
            list of linear expressions of :math:`z`-blocks. Use 0 or None for an empty block.
        b_l : float, expression
            corresponding lower bound. Ignored if None.
        b_u : float, expression
            corresponding upper bound. Ignored if None
        """
        shapes = list(filter(lambda x: x is not None,
                             [ufl.shape(a) if a is not None else None for a in Az]))
        if len(set(shapes)) != 1:
            raise ValueError("Shapes of constraint are not equal: "+str(shapes))
        if len(shapes[0]) == 0: # scalar constraint
            dim = 0
        else:
            dim = shapes[0][0]
        Vcons = self.generate_function_space(dim)
        self.constraints.append({"A": Az, "bu": bu, "bl":bl, "dim": dim, "V": Vcons})

    def set_linear_term(self, cz):
        """
        Adds a linear term :math:`c^Tz` where
        :math:`z=[X, Y_0, \ldots, Y_n]`

        Parameters
        ----------

        cz : list
            list of linear expressions of :math:`z`-blocks. Use 0 or None for an empty block.
        """
        self.c = cz

    def __rmul__(self, alpha):
        # if type(alpha) in [float, int, Constant] or isinstance(alpha, ufl.core.expr.Expr) or isinstance(alpha, ufl.algebra.Product):
        self.scale_factor = alpha
        return self

    def apply_on_problem(self, prob):
        """
        Appends function variables, constraints and objective to the global problem
        """
        self.declare_var(self.expr)
        if self.parameters is None:
            self.conic_repr(self.X)
        elif isinstance(self.parameters, tuple):
            self.conic_repr(self.X, *self.parameters)
        else:
            self.conic_repr(self.X, self.parameters)


class MeshConvexFunction(BaseConvexFunction):
    def generate_function_space(self, d, tensor=False):
        if self.interp_type == "nodal":
            if tensor:
                return TensorFunctionSpace(self.mesh, self.interpolation, self.degree, shape=(d, d), symmetry=True)
            else:
                if d > 0:
                    return VectorFunctionSpace(self.mesh, self.interpolation, self.degree, dim=d)
                else:
                    return FunctionSpace(self.mesh, self.interpolation, self.degree)
        elif self.interp_type == "quadrature":
            if tensor:
                element = TensorElement("Quadrature", self.mesh.ufl_cell(), degree=self.degree,
                                        shape=d, quad_scheme=self.dx.metadata()["quadrature_scheme"], symmetry=True)
            else:
                if d > 0:
                    element = VectorElement("Quadrature", self.mesh.ufl_cell(), degree=self.degree,
                                        dim=d, quad_scheme=self.dx.metadata()["quadrature_scheme"])
                else:
                    element = FiniteElement("Quadrature", self.mesh.ufl_cell(), degree=self.degree,
                                        quad_scheme=self.dx.metadata()["quadrature_scheme"])
            return FunctionSpace(self.mesh, element)
        elif self.interp_type == "vertex":
            if d > 0:
                element = VectorElement("DG", self.mesh.ufl_cell(), degree=self.degree, dim=d)
            else:
                element = FiniteElement("DG", self.mesh.ufl_cell(), degree=self.degree)
            return FunctionSpace(self.mesh, element)

    def apply_on_problem(self, prob):
        BaseConvexFunction.apply_on_problem(self, prob)
        nvar = len(prob.var)

        prob.var += self.additional_variables
        prob.Vx += [v.function_space() for v in self.additional_variables]
        prob.cones += self.cones
        prob.ux += self.ux
        prob.lx += self.lx
        for constraint in self.constraints:
            c_var = [None]*nvar
            def constraint_func(Z):
                if constraint["A"][0] not in [None, 0]:
                    for p in range(nvar):
                        c_var[p] = dot(Z, ufl.replace(constraint["A"][0], {self.X: self.X_expr}))*self.dx
                return c_var + [dot(Z, Ai)*self.dx if Ai is not None else None for Ai in constraint["A"][1:]]
            def constraint_rhs(b):
                if b is None:
                    return None
                elif b == 0:
                    return 0
                elif type(b) == list:
                    return lambda Z: dot(Z, as_vector(b))*self.dx
                else:
                    return lambda Z: dot(Z, b)*self.dx
            bu, bl = constraint["bu"], constraint["bl"]
            prob.add_ineq_constraint(constraint["V"], A=constraint_func,
                                     bu=constraint_rhs(bu), bl=constraint_rhs(bl))

        self.c_list = [None]*len(prob.var)
        if self.c[0] is not None:
            for p in range(nvar):
                c0 = ufl.replace(self.c[0], {self.X: self.X_expr})
                self.c_list[p] = c0
        for (i, ci) in enumerate(self.c[1:]):
            if ci is not None:
                self.c_list[nvar+i] = ci
        prob.add_obj_func([self.scale_factor*c*self.dx if c is not None else None for c in self.c_list])

    def compute_cellwise(self):
        """ Computes the value of int f*dx per cell """
        V0 = FunctionSpace(self.mesh, "DG", 0)
        y_ = TestFunction(V0)
        y0 = Function(V0)
        if self.c[0] is None:
            f0 = []
        else:
            f0 = [y_*ufl.replace(self.c[0], {self.X: self.X_expr})*self.dx]
        f = f0 + [y_*c*self.dx for c in self.c[1:] if c is not None]
        y0.vector().set_local(assemble(sum(f)).get_local())
        return y0

def my_restrict(X, measure):
    if measure.integral_type()=="interior_facet":
        return avg(X)
    else:
        return X

class FacetConvexFunction(BaseConvexFunction):
    """
    Parameters
    ----------

    x
        the block variable on which acts the convex function
    quadrature_scheme : {"default", "vertex", "lobatto"}
        quadrature scheme used for the integral computation and additional variables
        discretization. "vertex" scheme is used only with degree=1
    degree : int
        quadrature degree
    measure : list of `Measure`
        list of measures used for the integral. The first measure corresponds
        to **internal edges** (`dS`). The second measure correspond to **external edges** (`ds`).
    """
#    def __init__(self, x, parameters=None, quadrature_scheme="lobatto", degree=2, measure=[Measure("dS"), Measure("ds")]):
#        BaseConvexFunction.__init__(self, x, parameters=parameters, quadrature_scheme=quadrature_scheme,
#                                degree=degree, measure=measure)
    def generate_function_space(self, d, tensor=False):
        if d > 0:
            return VectorFunctionSpace(self.mesh, "Discontinuous Lagrange Trace",
                                       self.degree, dim=d)
        else:
            return FunctionSpace(self.mesh, "Discontinuous Lagrange Trace", self.degree)

    def apply_on_problem(self, prob):
        BaseConvexFunction.apply_on_problem(self, prob)
        nvar = len(prob.var)
        prob.var += self.additional_variables
        prob.Vx += [v.function_space() for v in self.additional_variables]
        prob.cones += self.cones
        prob.ux += self.ux
        prob.lx += self.lx
        for constraint in self.constraints:
            c_var = [None]*nvar
            def constraint_func(Z):
                if constraint["A"][0] not in [None, 0]:
                    for p in range(nvar):
                        c_var[p] = sum([dot(my_restrict(Z, self.dx[i]), ufl.replace(constraint["A"][0],
                                       {self.X: Xi}))*self.dx[i]
                                        for (i, Xi) in enumerate(self.X_expr)])
                return c_var + [sum([dot(my_restrict(Z, dx),
                                    ufl.replace(Ai, {Yi: my_restrict(Yi, dx)}))*dx for dx in self.dx])
                                if Ai is not None else None
                                for (Yi, Ai) in zip(self.additional_variables, constraint["A"][1:])]
            def constraint_rhs(b):
                if b is None:
                    return None
                elif b == 0:
                    return 0
                elif type(b) == list:
                    b = as_vector(b)
                else:
                    b
                    return lambda Z: sum([dot(my_restrict(Z, dx),
                                              my_restrict(as_vector(b), dx))*dx
                                          for dx in self.dx])
            bu, bl = constraint["bu"], constraint["bl"]
            prob.add_ineq_constraint(constraint["V"], A=constraint_func,
                                     bu=constraint_rhs(bu), bl=constraint_rhs(bl))

        c_list = [None]*len(prob.var)
        if self.c[0] is not None:
            for p in range(nvar):
                c_list[p] = sum([ufl.replace(self.c[0]*self.dx[i], {self.X: Xi})
                             for (i, Xi) in enumerate(self.X_expr)])
        for (i, ci) in enumerate(self.c[1:]):
            if ci is not None:
                c_list[nvar+i] = sum([ufl.replace(ci*dx, {prob.var[nvar+i]:
                                 my_restrict(prob.var[nvar+i], dx)}) for dx in self.dx])
        prob.add_obj_func([self.scale_factor*c if c is not None else None for c in c_list])

    def compute_cellwise(self):
        """ Computes the value of int f*dx per cell """
        V0 = FunctionSpace(self.mesh, "DG", 0)
        y_ = TestFunction(V0)
        y0 = Function(V0)
        if self.c[0] is None:
            f0 = []
        else:
            f0 = [ufl.replace(avg(y_)*self.c[0]*self.dx[i], {self.X: Xi})
                             for (i, Xi) in enumerate(self.X_expr)]
        for (i, ci) in enumerate(self.c[1:]):
            yi = self.additional_variables[i]
            if ci is not None:
                f0 += [ufl.replace(avg(y_)*ci*self.dx[0], {yi: avg(yi)}),
                                   y_*ci*self.dx[1]]
#                f0 += [y_*ci*self.dx[1]]
#        print(assemble(f0[0]))
        y0.vector().set_local(assemble(sum(f0)).get_local())
        return y0

class ConvexFunction(BaseConvexFunction):
    """
    A composite convex function which acts by default as a `MeshConvexFunction`.
    """
    def __init__(self, x, parameters=None, interp_type="quadrature", quadrature_scheme=None, degree=None, measure=None,
                 on_facet=False):
        """
        Use `ConvexFunction.on_facet` method to act as a `FacetConvexFunction`.

        Parameters
        ----------

        x
            the block variable on which acts the convex function
        quadrature_scheme : {"default", "vertex", "lobatto"}
            quadrature scheme used for the integral computation and additional variables
            discretization. "vertex" scheme is used only with degree=1
        degree : int
            quadrature degree
        measure : `Measure`
            measure used for the integral
        """
        self.on_facet = on_facet
        if self.on_facet:
            if measure is None:
                    measure = [Measure("dS"), Measure("ds")]
            if degree is None:
                degree = 2
            if quadrature_scheme is None:
                quadrature_scheme = "lobatto"
            FacetConvexFunction.__init__(self, x, parameters=parameters,
                                         interp_type=interp_type,
                                    quadrature_scheme=quadrature_scheme,
                                    degree=degree, measure=measure)
        else:
            if measure is None:
                measure = Measure("dx")
            if degree is None:
                degree = 1
            if quadrature_scheme is None:
                quadrature_scheme = "default"
            MeshConvexFunction.__init__(self, x, parameters=parameters,
                                         interp_type=interp_type,
                                    quadrature_scheme=quadrature_scheme,
                                    degree=degree, measure=measure)

#    @classmethod
#    def on_facet(cls, x, quadrature_scheme="lobatto",
#                 degree=2, measure=[Measure("dS"), Measure("ds")], **kwargs):
#        """
#        Transforms a `ConvexFunction` into a `FacetConvexFunction`
#
#        Parameters
#        ----------
#
#        x
#            the block variable on which acts the convex function
#        quadrature_scheme : {"default", "vertex", "lobatto"}
#            quadrature scheme used for the integral computation and additional variables
#            discretization. "vertex" scheme is used only with degree=1
#        degree : int
#            quadrature degree
#        measure : list of `Measure`
#            list of measures used for the integral. The first measure corresponds
#            to **internal edges** (`dS`). The second measure correspond to **external edges** (`ds`).
#        """
#        print(kwargs)
#        instance = FacetConvexFunction.__new__(cls)
#        FacetConvexFunction.__init__(instance, x, parameters=tuple(kwargs.values()),
#                                     quadrature_scheme=quadrature_scheme,
#                                     degree=degree, measure=measure)
#        instance.on_facet = True
#        return instance

    def generate_function_space(self, d, tensor=False):
        if self.on_facet:
            return FacetConvexFunction.generate_function_space(self, d, tensor)
        else:
            return MeshConvexFunction.generate_function_space(self, d, tensor)

    def apply_on_problem(self, prob):
        if self.on_facet:
            return FacetConvexFunction.apply_on_problem(self, prob)
        else:
            return MeshConvexFunction.apply_on_problem(self, prob)

    def compute_cellwise(self):
        if self.on_facet:
            return FacetConvexFunction.compute_cellwise(self)
        else:
            return MeshConvexFunction.compute_cellwise(self)


class EqualityConstraint(ConvexFunction):
    """ Imposes a linear equality constraint :math:`X = b` """
    def __init__(self, x, b=0, **kwargs):
        ConvexFunction.__init__(self, x, parameters=b, **kwargs)
    def conic_repr(self, X, b):
        self.add_eq_constraint([X], b=b)

class InequalityConstraint(ConvexFunction):
    """ Imposes a linear inequality constraint :math:`bl <= X <= bu` """
    def __init__(self, x, bl=None, bu=None, **kwargs):
        ConvexFunction.__init__(self, x, parameters=(bl, bu), **kwargs)
    def conic_repr(self, X, bl, bu):
        self.add_ineq_constraint([X], bl=bl, bu=bu)

class LinearTerm(ConvexFunction):
    """ Defines the linear function :math:`c^TX` """
    def __init__(self, x, c, **kwargs):
        ConvexFunction.__init__(self, x, parameters=c, **kwargs)
    def conic_repr(self, X, c):
        self.set_linear_term([dot(c, X)])

class QuadraticTerm(ConvexFunction):
    """ Defines the quadratic function :math:`\\frac{1}{2}(X-x_0)^TQ^TQ(X-x_0)` """
    def __init__(self, x, Q=1, x0=0, **kwargs):
        ConvexFunction.__init__(self, x, parameters=(Q, x0), **kwargs)
    def conic_repr(self, X, Q, x0):
        d = self.dim_x
        if d==0:
            Y = self.add_var(3, cone=RQuad(3))
            self.add_eq_constraint([Q*X, -Y[2]], b=Q*x0)
        else:
            Y = self.add_var(d+2, cone=RQuad(d+2))
            if type(Q) in [float, int] or ufl.shape(Q)==():
                Q = Q*Identity(d)
            if x0==0:
                b = 0
            else:
                b = dot(Q, x0)
            self.add_eq_constraint([dot(Q, X), -as_vector([Y[i] for i in range(2, d+2)])], b=b)
        self.add_eq_constraint([None, Y[1]], b=1)
        self.set_linear_term([None, Y[0]])


class L2Norm(ConvexFunction):
    """ Defines the scaled L2-norm function :math:`k||X||_2` """
    def __init__(self, x, k=1, **kwargs):
        ConvexFunction.__init__(self, x, parameters=k, **kwargs)
    def conic_repr(self, X, k):
        d = self.dim_x
        if d == 0:
            Y = self.add_var(2, cone=Quad(2))
            self.add_eq_constraint([X, -Y[1]])
        else:
            Y = self.add_var(d+1, cone=Quad(d+1))
            self.add_eq_constraint([X, -as_vector([Y[i] for i in range(1, d+1)])])
        self.set_linear_term([None, k*Y[0]])

class L1Norm(ConvexFunction):
    """ Defines the scaled L1-norm function :math:`k||X||_1` """
    def __init__(self, x, k=1, **kwargs):
        ConvexFunction.__init__(self, x, parameters=k, **kwargs)
    def conic_repr(self, X, k):
        d = self.dim_x
        Y = self.add_var(d)
        self.add_ineq_constraint([X, -Y], bu=0)
        self.add_ineq_constraint([-X, -Y], bu=0)
        if d==0:
            self.set_linear_term([None, k*Y])
        else:
            self.set_linear_term([None, k*sum(Y)])

class AbsValue(L1Norm):
    """ Defines the scaled absolute value function :math:`k|X|`  """
    pass
#    def conic_repr(self, X, k):
#        assert self.dim_x == 0, "Variable is not scalar"
#        super().conic_repr(X, k)

class LinfNorm(ConvexFunction):
    """ Defines the scaled Linf-norm function :math:`k||X||_\\infty`  """
    def __init__(self, x, k=1, **kwargs):
        ConvexFunction.__init__(self, x, parameters=k, **kwargs)
    def conic_repr(self, X, k):
        d = self.dim_x
        Y = self.add_var()
        if d == 0:
            e = 1
        else:
            e = as_vector([1,]*d)
        self.add_ineq_constraint([X, -Y*e], bu=0)
        self.add_ineq_constraint([-X, -Y*e], bu=0)
        self.set_linear_term([None, k*Y])

class L2Ball(ConvexFunction):
    """ Defines the scaled L2-ball constraint :math:`||X||_2 \leq k`  """
    def __init__(self, x, k=1, **kwargs):
        ConvexFunction.__init__(self, x, parameters=k, **kwargs)
    def conic_repr(self, X, k=1):
        d = self.dim_x
        Y = self.add_var(d+1, cone=Quad(d+1))
        self.add_eq_constraint([X, -as_vector([Y[i] for i in range(1, d+1)])])
        self.add_eq_constraint([None, Y[0]], b=k)

class L1Ball(ConvexFunction):
    """ Defines the scaled L1-ball constraint :math:`||X||_1 \leq k` """
    def __init__(self, x, k=1, **kwargs):
        ConvexFunction.__init__(self, x, parameters=k, **kwargs)
    def conic_repr(self, X, k=1):
        d = self.dim_x
        Y = self.add_var(d)
        self.add_ineq_constraint([X, -Y], bu=0)
        self.add_ineq_constraint([-X, -Y], bu=0)
        if d == 0:
            self.add_eq_constraint([None, Y], b=k)
        else:
            self.add_eq_constraint([None, sum(Y)], b=k)

class LinfBall(ConvexFunction):
    """ Defines the scaled Linf-ball constraint :math:`||X||_\\infty \leq k`"""
    def __init__(self, x, k=1, **kwargs):
        ConvexFunction.__init__(self, x, parameters=k, **kwargs)
    def conic_repr(self, X, k=1):
        d = self.dim_x
        Y = self.add_var()
        if d == 0:
            e = 1
        else:
            e = as_vector([1,]*d)
        self.add_ineq_constraint([X, -Y*e], bu=0)
        self.add_ineq_constraint([-X, -Y*e], bu=0)
        self.add_eq_constraint([None, Y], b=k)
