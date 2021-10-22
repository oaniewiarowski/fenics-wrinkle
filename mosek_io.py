#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mosek_io provides interface to the Mosek optimization solver
only Linear Programming and Second-Order Cone Programming
problems are currently supported

@author: Jeremy Bleyer, Ecole des Ponts ParisTech,
Laboratoire Navier (ENPC,IFSTTAR,CNRS UMR 8205)
@email: jeremy.bleyer@enpc.f
"""
from dolfin import Function, TestFunction, TrialFunction, assemble, as_backend_type, cpp
import scipy.sparse as sp
import numpy as np
import sys
import mosek
import ufl
from fenics_optim.utils import to_list, subk_list, subl_list, \
                                half_vect2subk, half_vect2subl
from fenics_optim import Product, SDP
from itertools import compress
# unimportant value to denote infinite bounds
inf = 1e30

MOSEK_CONE_TYPES = {"quad": mosek.conetype.quad,
                    "rquad": mosek.conetype.rquad}
version = mosek.Env().getversion()
if version >= (9, 0, 0):
     MOSEK_CONE_TYPES.update({"ppow": mosek.conetype.ppow,
                              "dpow": mosek.conetype.dpow,
                              "pexp": mosek.conetype.pexp,
                              "dexp": mosek.conetype.dexp})

class MosekProblem:
    """
    This class allows to define a generic optimization problem using the
    Mosek optimization solver.
    """
    def __init__(self, name=""):
        self.name = name
        self.Vx = []
        self.Vy = []
        self.lagrange_multiplier_names = []
        self.cones = []
        self.ux = []
        self.lx = []
        self.var = []
        self.int_var = []
        self.A = []
        self.bu = []
        self.bl = []
        self.bc_dual = []
        self.bc_prim = []
        self.c = []
        self.parameters = self._default_parameters()
        self.streamprinter = None

    def _default_parameters(self):
        return {"presolve": True, "presolve_lindep": False, "log_level": 10, "tol_rel_gap": 1e-7,
            "solve_form": "free", "num_threads": 0, "dump_file": None}

    def add_var(self, V, cone=None, lx=None, ux=None, bc=None, name=None, int_var=False):
        """
        Adds a (list of) optimization variable belonging to the corresponding
        FunctionSpace V.


        Parameters
        ----------

        V : (list of) `FunctionSpace`
            variable FunctionSpace
        cone : (list of) `Cone`
            cone in which each variable belongs (None if no constraint)
        ux : (list of) float, Function
            upper bound on variable :math:`x\leq u_x`
        lx : (list of) float, Function
            lower bound on variable :math:`l_x \leq x`
        bc : (list of) `DirichletBC`
            boundary conditions applied to the variables (None if no bcs)
        name : (list of) str
            name of the associated functions
        int_var : (list of) bool
            True if variable is an integer, False if it is continuous (default)

        Returns
        -------

        x : Function tuple
            optimization variables
        """
        if not isinstance(V, list):
            V_list = [V]
            bc_list = [bc]
        else:
            V_list = V
            bc_list = bc
        nlist = len(V_list)

        self.lx += to_list(lx, nlist)
        self.ux += to_list(ux, nlist)
        self.cones += to_list(cone, nlist)
        self.bc_prim += to_list(bc_list, nlist)
        name_list = to_list(name, nlist)
        self.int_var += to_list(int_var, nlist)

        new_var = [Function(v, name=n) for (v, n) in zip(V_list, name_list)]
        self.var += new_var
        self.Vx += V_list

        if nlist == 1:
            return new_var[0]

        return tuple(new_var)

    def _update_names(self):
        """ Checks for similar variable names, in that case, appends indices at the end """
        names = [v.name() for v in self.var]
        for (i, name) in enumerate(names):
            j = 0
            while name in names[i+1:]:
                k = names.index(name)
                names[k] = name+"-"+str(j)
                j += 1
        for (v, name) in zip(self.var, names):
            v.rename(name, v.name())

    def _get_field(self, function_list, name):
        name_list = [v.name() for v in function_list]
        try:
            i = name_list.index(name)
            return function_list[i]
        except ValueError:
            raise ValueError("'{}' not in list. Available names are:\n{}".format(name, name_list))

    def get_var(self, name):
        return self._get_field(self.var, name)

    def get_lagrange_multiplier(self, name):
        return self._get_field(self.y, name)

    def add_eq_constraint(self, Vy, A=None, b=0, bc=None, name=None):
        """
        Adds a linear equality constraint :math:`Ax = b`. Impementation will
        rely on a block form :math:`[A_0,\\ldots, A_n]\\begin{Bmatrix}x_0 \\\\
        \\vdots \\\\ x_n\\end{Bmatrix}` associated with the block-wise optimization
        variables :math:`x_i`.

        Parameters
        -----------

        Vy : `FunctionSpace`
            FunctionSpace of the corresponding Lagrange multiplier
        A : function
            A function returning a list of forms corresponding to the
            constraint lhs on each block optimization variable. Parameter of the function
            is the constraint Lagrange multiplier.
        b : float, function
            A float or a function of the Lagrange multiplier corresponding to the
            constraint right-hand side (default is 0)
        bc : DirichletBC
            boundary conditions to apply on the Lagrange multiplier (will be
            applied to all columns of the constraint when possible)
        name : str
            Lagrange multiplier name
        """
        self.add_ineq_constraint(Vy, A, b, b, bc, name)

    def add_ineq_constraint(self, Vy, A=None, bu=None, bl=None, bc=None, name=None):
        """
        Adds a linear inequality constraint :math:`b_l \leq Ax \leq b_u`. Impementation will
        rely on a block form :math:`[A_0,\\ldots, A_n]\\begin{Bmatrix}x_0 \\\\
        \\vdots \\\\ x_n\\end{Bmatrix}` associated with the block-wise optimization
        variables :math:`x_i`.

        Parameters
        -----------

        Vy : `FunctionSpace`
            FunctionSpace of the corresponding Lagrange multiplier
        A : function
            A function returning a list of forms corresponding to the
            constraint lhs on each block optimization variable. Parameter of the function
            is the constraint Lagrange multiplier.
        bl : float, function
            A float or a function of the Lagrange multiplier corresponding to the
            constraint lower bound (default is None)
        bu : float, function
            A float or a function of the Lagrange multiplier corresponding to the
            constraint upper bound (default is None)
        bc : DirichletBC
            boundary conditions to apply on the Lagrange multiplier (will be
            applied to all columns of the constraint when possible)
        name : str
            Lagrange multiplier name
        """
        self.Vy.append(Vy)
        self.lagrange_multiplier_names.append(name)
        v = TestFunction(Vy)
        if callable(bu):
            self.bu.append(bu(v))
        else:
            self.bu.append(bu)
#        else:
#            self.bu.append(bu(v))
#        if bl is None or type(bl) in [int, float]:
#            self.bl.append(bl)
#        else:
#            self.bl.append(bl(v))
        if callable(bl):
            self.bl.append(bl(v))
        else:
            self.bl.append(bl)
        self.A.append(A(v))
        if isinstance(bc, list):
            self.bc_dual.append(bc)
        else:
            self.bc_dual.append([bc])

    def add_obj_func(self, obj):
        """
        Adds an objective function

        Parameters
        ----------

        obj : list of float, function
            objective function described either as a list of floats
            or as a linear function of the optimization variables with signature
            `obj(x0, x1, ..., xn)`. The function must return a list of linear functions of the :math:`x_i`.
        """
        obj = to_list(obj)
        v_test = [TestFunction(V) for V in self.Vx]
        self.c.append([ufl.replace(c, {v:v_}) if isinstance(c, ufl.form.Form) else
                       c for (c, v, v_) in zip(obj, self.var, v_test)])

    def add_convex_term(self, conv_fun):
        """ Adds the convex term `conv_fun` to the problem """
        conv_fun.apply_on_problem(self)

    def write_problem(self):
        self._update_names()

        sdp_cones = [isinstance(c, SDP) for c in self.cones]
        not_sdp_cones = [not s for s in sdp_cones]
        sdp_filter = lambda x: (list(compress(x, not_sdp_cones)), list(compress(x, sdp_cones)))

        self.nvar, self.nvar_sdp = sdp_filter([v.dim() for v in self.Vx])
        self.ncon = [v.dim() for v in self.Vy]
        lvar = len(self.nvar)+len(self.nvar_sdp)
        lcon = len(self.ncon)
        Nvar = sum(self.nvar)
        Ncon = sum(self.ncon)

        self.A_array, self.sdp_A_list = block_mat_to_sparse(self.A, self.var,
                                                             self.ncon, self.bc_dual,
                                                             sdp_cones)
        assert len(self.A) == lcon
        assert max([len(a) for a in self.A]) <= lvar
        assert len(self.bu) == lcon or self.bu is None
        assert len(self.bl) == lcon or self.bl is None
        assert self.ux is None or len(self.ux) == lvar
        assert self.lx is None or len(self.lx) == lvar

        sum_c = [None]*lvar
        for c_row in self.c:
            for (i, ci) in enumerate(c_row):
                if ci is not None and ci != 0:
                    if not isinstance(ci, ufl.form.Form) or len(ufl.algorithms.extract_arguments(ci))==1:
                        if sum_c[i] is None or sum_c[i] == 0:
                            sum_c[i] = ci
                        else:
                            sum_c[i] += ci
        cx, cx_sdp = sdp_filter(sum_c)
        self.carray = block_vect_to_array(cx, self.nvar)
        self.carray_sdp = block_vect_to_array(cx_sdp, self.nvar_sdp)

        self.bound_arrays = [None,]*4
        # no bounds on SDP variables
        inputs = [self.bu, self.bl, sdp_filter(self.ux)[0], sdp_filter(self.lx)[0]]
        default_values = [inf, -inf, inf, -inf]
        sizes = [(self.ncon, Ncon)]*2+[(self.nvar, Nvar)]*2

        for (i, (inp, deflt, size)) in enumerate(zip(inputs, default_values, sizes)):
            if inp is not None:
                self.bound_arrays[i] = block_vect_to_array(inp, size[0], default_value=deflt)
            else:
                self.bound_arrays[i] = [deflt]*size[1]

        A_bc, b_bc = bcs_to_block_mat(self.bc_prim, self.nvar)
        if A_bc != []:
            self.bound_arrays[0] += b_bc.tolist()
            self.bound_arrays[1] += b_bc.tolist()
            self.A_array = sp.vstack((self.A_array, A_bc))
            Ncon += len(b_bc)

        # SDP cones are specified separately from the other
        cone_list, cone_type, cone_alp = block_cones_to_list(sdp_filter(self.cones)[0], self.nvar)
        sdp_cone_list, _, _ = block_cones_to_list(sdp_filter(self.cones)[1], self.nvar_sdp)
        self.sdp_cones = sdp_cone_list

        assert self.A_array.shape == (Ncon, Nvar)

        self.cone_dict = {"list": cone_list, "type": cone_type,
                          "alpha": cone_alp}
        self.num_cones = len(cone_list)
        if self.parameters["log_level"] > 0:
            print("Matrix shape:", self.A_array.shape)
            print("Number of cones:", self.num_cones)

    def call_mosek(self, readfile=False):
        env = mosek.Env()
        self.task = env.Task()
        # if self.streamprinter is not None:
        #     streamprinter = self.streamprinter
        self.task.set_Stream(mosek.streamtype.log, streamprinter)
        if readfile:
            try:
                print("Reading problem file...")
                self.task.readdata("mosekio.jtask")
            except mosek.Exception:
                print("Problem reading the file")
        else:

            Ncon, Nvar = self.A_array.shape
            self.task.appendcons(Ncon)
            self.task.appendvars(Nvar)

            self.task.putaijlist(self.A_array.row,
                                 self.A_array.col,
                                 self.A_array.data)
            self.task.putclist(range(Nvar), self.carray)

            bu, bl, ux, lx = self.bound_arrays
            bk = map(get_boundkey, zip(bl, bu))
            xk = map(get_boundkey, zip(lx, ux))
            self.task.putconboundlist(range(Ncon), list(bk), bl, bu)
            self.task.putvarboundlist(range(Nvar), list(xk), lx, ux)

            # SDP variables and constraints
            if len(self.sdp_cones) > 0:
                self.barvardim = [int(-1+(1+8*len(c))**0.5)//2 for c in self.sdp_cones]
                self.task.appendbarvars(self.barvardim)
                nvar_sdp = len(self.sdp_cones)
                clist = [[self.carray_sdp[s] for s in self.sdp_cones[i]] for i in range(nvar_sdp)]
                subk = [s for d in self.barvardim for s in subk_list(d)]
                subl = [s for d in self.barvardim for s in subl_list(d)]
                subj = [i for i in range(nvar_sdp) for j in self.sdp_cones[i]]
                vals = []
                for i in range(nvar_sdp):
                    vals += clist[i]
                num = len(self.carray_sdp)
                vals = [v if k==l else v/2 for (k,l,v) in zip(subk, subl, vals)]
                self.task.putbarcblocktriplet(num, subj, subk, subl, vals)

                subi = []
                subj = []
                subk = []
                subl = []
                vals = []
                buffi = 0
                nrow = 0
                for (i, Arow) in enumerate(self.sdp_A_list):
                    buffj = 0
                    for (j, A) in enumerate(Arow):
                        d = self.barvardim[j]
                        d2 = d*(d+1)//2
                        if A is not None:
                            nrow, ncol = A.shape
                            assert nrow == self.ncon[i]
                            row, col, data = A.row, A.col, A.data
                            subi += (buffi+row).tolist()
                            subj += (buffj+np.floor_divide(col, d2)).tolist()
                            lmod = np.remainder(col, d2).tolist()
                            subk += half_vect2subk(lmod, d)
                            subl += half_vect2subl(lmod, d)
                            vals += data.tolist()
                        buffj += self.nvar_sdp[j]//d2
                    buffi += self.ncon[i]
                vals = [v if k==l else v/2 for (k,l,v) in zip(subk, subl, vals)]
                self.task.putbarablocktriplet(len(subi), subi, subj, subk, subl, vals)

            # Integer variables
            var_cumdim = np.cumsum(np.array(self.nvar))
            for (i, int_var) in enumerate(self.int_var):
                if int_var:
                    self.task.putvartypelist(range(var_cumdim[i]-self.nvar[i],var_cumdim[i]),
                                             [mosek.variabletype.type_int]*self.nvar[i])

            if self.sense == "max":
                self.task.putobjsense(mosek.objsense.maximize)

            cone_types = [mosek.conetype.quad if ct == "quad" else
                          mosek.conetype.rquad for ct in self.cone_dict["type"]]
            for k in range(self.num_cones):
                self.task.appendcone(MOSEK_CONE_TYPES[self.cone_dict["type"][k]],
                                     self.cone_dict["alpha"][k], self.cone_dict["list"][k])

        self.set_task_parameters()
        if self.parameters["dump_file"] is not None:
            self.task.writedata(self.parameters["dump_file"])
        self.task.optimize()
        self.task.solutionsummary(mosek.streamtype.msg)

    def set_task_parameters(self):
        assert all([p in self._default_parameters().keys() for p in self.parameters.keys()]), \
               "Available parameters are:\n{}".format(self._default_parameters())
        self.task.putintparam(mosek.iparam.log, self.parameters["log_level"])
        assert type(self.parameters["presolve"]) in [bool, mosek.presolvemode], \
        "Presolve parameter must be a bool or of `mosek.presolvemode` type"
        self.task.putintparam(mosek.iparam.presolve_use, self.parameters["presolve"])
        self.task.putintparam(mosek.iparam.presolve_lindep_use, self.parameters["presolve_lindep"])
        #     ... without basis identification (integer parameter)
        self.task.putintparam(mosek.iparam.intpnt_basis, mosek.basindtype.never)
        #     Set relative gap tolerance (double parameter)
        self.task.putdouparam(mosek.dparam.intpnt_co_tol_rel_gap, self.parameters["tol_rel_gap"])
        # Controls whether primal or dual form is solved
        mapping = {"free": mosek.solveform.free,
                   "primal": mosek.solveform.primal,
                   "dual": mosek.solveform.dual}
        self.task.putintparam(mosek.iparam.intpnt_solve_form,
                              mapping[self.parameters["solve_form"]])
        self.task.putintparam(mosek.iparam.sim_solve_form,
                              mapping[self.parameters["solve_form"]])
        self.task.putintparam(mosek.iparam.num_threads, self.parameters["num_threads"])
        self.task.putintparam(mosek.iparam.auto_update_sol_info, mosek.onoffkey.on)
        # self.task.putintparam(mosek.iparam.infeas_report_auto, mosek.onoffkey.on)

    def get_solution_info(self, output=True):
        int_info = ["intpnt_iter", "opt_numcon", "opt_numvar", "ana_pro_num_var"]
        double_info = ["optimizer_time", "presolve_eli_time", "presolve_lindep_time", "presolve_time",
                       "intpnt_time", "intpnt_order_time", "sol_itr_primal_obj", "sol_itr_dual_obj"]
        int_value = [self.task.getintinf(getattr(mosek.iinfitem,k)) for k in int_info]
        double_value = [self.task.getdouinf(getattr(mosek.dinfitem, k)) for k in double_info]
        info = dict(zip(int_info+double_info, int_value+double_value))
        info.update({"solution_status": str(self.task.getsolsta(mosek.soltype.itr)).split('.')[1]})
        if output:
            print("Solver information:\n{}".format(info))
        return info

    def read_solution(self, xsol, ysol, slxsol, suxsol):
        solsta = self.task.getsolsta(mosek.soltype.itr)
        version = mosek.Env.getversion()
        if version <= (9, 0, 0):
            status = [mosek.solsta.optimal,
                      mosek.solsta.near_optimal,
                      mosek.solsta.unknown]
        else: # removed near_optimal status in version 9
            status = [mosek.solsta.optimal,
                      mosek.solsta.unknown]
        if solsta in status:
            numvar = self.task.getnumvar()
            numcon = self.task.getnumcon()
            self.xx = np.zeros((numvar, ))
            self.yy = np.zeros((numcon, ))
            self.task.getxx(mosek.soltype.itr, self.xx)
            self.task.gety(mosek.soltype.itr, self.yy)
            populate_sol(xsol, self.xx)
            populate_sol(ysol, self.yy)

            if len(self.sdp_cones) > 0:
                self.barx = [[0.] * ((d*(d+1))//2) for d in self.barvardim]
                for (j, b) in enumerate(self.barx):
                    self.task.getbarxj(mosek.soltype.itr, j, b)
                    # TODO: populate sol with barx

            if slxsol is not None:
                slx = np.zeros((numvar, ))
                self.task.getslx(mosek.soltype.itr, slx)
                populate_sol(slxsol, slx)
            if suxsol is not None:
                sux = np.zeros((numvar, ))
                self.task.getsux(mosek.soltype.itr, sux)
                populate_sol(suxsol, sux)

            if solsta == mosek.solsta.unknown:
                print("Warning: Solver finished with UNKNOWN type. Solution might be inaccurate...")
            return self.task.getprimalobj(mosek.soltype.itr)
        else:
            return np.nan

    def optimize(self, sense="min", get_bound_dual=False):
        """
        Launches the optimizer with sense of optimization specified by min/max

        Returns
        -------
        pobj : float
            the computed optimal value
        """
        if sense=="minimize":
            sense="min"
        elif sense=="maximize":
            sense="max"
        self.sense = sense
        self.write_problem()
        self.call_mosek()

        self.y = [Function(v, name=name) for (v, name) in zip(self.Vy, self.lagrange_multiplier_names)]
        if get_bound_dual:
            self.slx = [Function(v) for v in self.Vx]
            self.sux = [Function(v) for v in self.Vx]
        else:
            self.slx = None
            self.sux = None
        self.pobj = self.read_solution(self.var, self.y, self.slx, self.sux)

        return self.pobj


# # A log message
def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()
    
    # with open("mosek_timings.log", "a") as out:
    #     out.write(text)

def keep_rows_csr(mat, indices):
    """
    Keep the rows denoted by ``indices`` form the CSR sparse matrix ``mat``.
    """
    if not isinstance(mat, sp.csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    indices = list(indices)
    mask = np.zeros(mat.shape[0], dtype=bool)
    mask[indices] = True
    return mat[mask]

def block_mat_to_sparse(A, var, ncon, bc_dual, sdp_cones):
    rows = []
    sdp_rows = []
    for (i, nc) in enumerate(ncon):
        cols = []
        sdp_cols = []
        for (j, v) in enumerate(var):
            nv = v.function_space().dim()
            if j >= len(A[i]):
                Ab = None
            else:
                Ab = A[i][j]
            if Ab == 0. or Ab is None:
                As = sp.coo_matrix((nc, nv))
            else:
                v_ = TrialFunction(v.function_space())
                f = ufl.replace(Ab, {v: v_})
                if len(ufl.algorithms.extract_arguments(f)) == 2:
                    AAb = assemble(ufl.lhs(f), keep_diagonal=True)
                    # try:
                    #     # print("UFLACS\n\n")
                    #     # raise
                    #     AAb = assemble(ufl.lhs(f), keep_diagonal=True,
                    #                     form_compiler_parameters={"representation": "uflacs"})
                    # except:
                    #     # print("QUAD\n\n")
                    #     AAb = assemble(ufl.lhs(f), keep_diagonal=True,
                    #                     form_compiler_parameters={"representation": "quadrature"})
                    if bc_dual[i] is not None:
                        for bc in bc_dual[i]:
                            if bc is not None:
                                try:
                                    bc.zero(AAb)
                                except:
                                    pass
                    Aa = as_backend_type(AAb).mat()
                    row, col, data = Aa.getValuesCSR()
                    As = sp.csr_matrix((data, col, row), shape=(nc, nv))
                    As.eliminate_zeros()
                else:
                    As = sp.coo_matrix((nc, nv))

            if sdp_cones[j]:
                if Ab == 0. or Ab is None:
                    sdp_cols.append(None)
                else:
                    sdp_cols.append(As.tocoo())
            else:
                cols.append(As)

        rows.append(cols)
        sdp_rows.append(sdp_cols)

    AA = sp.bmat(rows, format="coo")
    return AA, sdp_rows

def block_vect_to_array(r, ndim, bcs=None, default_value=0):
    rarray = np.zeros((0, ))
    for (i, ri) in enumerate(r):
        if ri is not None:
            if isinstance(ri, ufl.form.Form):
                if len(ufl.algorithms.extract_arguments(ri)) == 1:
                    rarray = np.concatenate((rarray, assemble(ri).get_local()))
                else:
                    rarray = np.concatenate((rarray, default_value*np.ones((ndim[i], ))))
            elif isinstance(ri, Function):
                rarray = np.concatenate((rarray, ri.vector().get_local()))
            elif type(ri) in [int, float]:
                rarray = np.concatenate((rarray, ri*np.ones((ndim[i], ))))
            else:
                rarray = np.concatenate((rarray, ri.get_local()))
        else:
            rarray = np.concatenate((rarray, default_value*np.ones((ndim[i], ))))
    return rarray.tolist()

def bcs_to_block_mat(bcs, nvar):
    cols = []
    b_bc = []
    for (i, bc) in enumerate(bcs):
        bc_list = to_list(bc)
        for bci in bc_list:
            if bc is not None:
                nv = nvar[i]
                Al = []
                for (j, nvj) in enumerate(nvar):
                    if i == j:
                        Al.append(sp.eye(nvj))
                    else:
                        Al.append(sp.csr_matrix((nv, nvj)))
                As = sp.hstack(tuple(Al)).tocsr()
                bc_lines = list(bci.get_boundary_values().keys())
                bc_values = np.array(list(bci.get_boundary_values().values()))

                idx_bc = np.argsort(bc_lines)
                As = keep_rows_csr(As, bc_lines)

                cols.append(As)
                b_bc.append(bc_values[idx_bc])

    if cols:
        A_bc = sp.vstack(tuple(cols))
        b_bc = np.concatenate(tuple(b_bc))
        return A_bc, b_bc

    return [], []

def block_cones_to_list(cones, ndim):
    buff = 0
    cone_list = []
    cone_type = []
    cone_alp = []
    for (i, c) in enumerate(cones):
        if c is None:
            buff += ndim[i]
        elif isinstance(c, Product):
            for cc in c.cones:
                d = cc.dim
                ncones = ndim[i]//d//len(c.cones)
                for j in range(ncones):
                    cone_list.append(range(buff, buff+d))
                    cone_type.append(cc.type)
                    cone_alp.append(getattr(cc, 'alp', 0))
                    buff += d
        elif isinstance(c, SDP):
            d = c.dim
            d2 = d*(d+1)//2
            ncones = ndim[i]//d2
            for j in range(ncones):
                cone_list.append(range(buff, buff+d2))
                buff += d2
        else:
            d = c.dim
            ncones = ndim[i]//d
            for j in range(ncones):
                cone_list.append(range(buff, buff+d))
                cone_type.append(c.type)
                cone_alp.append(getattr(c, 'alp', 0))
                buff += d
    return cone_list, cone_type, cone_alp

def get_boundkey(x):
    l, u = x
    if u == inf and l == -inf:
        return mosek.boundkey.fr
    elif u == inf:
        return mosek.boundkey.lo
    elif l == inf:
        return mosek.boundkey.up
    elif l == u:
        return mosek.boundkey.fx

    return mosek.boundkey.ra

def populate_sol(xsol, array):
    buff = 0
    for x in xsol:
        x.vector().set_local(array[buff:x.vector().size()+buff])
        buff += x.vector().size()
