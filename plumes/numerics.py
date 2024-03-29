r"""Classes for integrating models forward in time

Every timestepping scheme in this module takes in an argument `equation` to
its initializer. An `equation` is a Python function that takes in the state
variables of the system and returns a Firedrake `Form` object describing
the spatially-discretized, weak form of the system. See the `models` sub-
module for routines that will build these `equation` functions for you from
the requisite input data.
"""

from abc import ABC, abstractmethod
import firedrake
from firedrake import (
    inner,
    dx,
    NonlinearVariationalProblem as Problem,
    NonlinearVariationalSolver as Solver
)

__all__ = [
    "ExplicitEuler",
    "SSPRK33",
    "SSPRK34",
    "IMEX",
    "ThetaMethod",
    "ImplicitEuler",
    "ImplicitMidpoint",
]


class Integrator(ABC):
    @abstractmethod
    def step(self, timestep):
        r"""Propagate the model forward by a single timestep"""
        pass


def _solver_params(Q):
    block_parameters = {
        "ksp_type": "preonly",
        "pc_type": "ilu",
        "sub_pc_type": "bjacobi"
    }

    if not hasattr(Q, "num_sub_spaces"):
        return block_parameters

    fieldsplits = {
        f"fieldsplit_{index}": block_parameters
        for index in range(Q.num_sub_spaces())
    }

    return dict(ksp_type="preonly", pc_type="fieldsplit", **fieldsplits)


class ExplicitEuler(Integrator):
    def __init__(
        self,
        equation,
        state,
        solver_parameters=None,
        form_compiler_parameters=None
    ):
        r"""A first-order explicit timestepping scheme

        Parameters
        ----------
        equation
            A Python function that takes in the state variable and returns a
            firedrake.Form object describing the weak form of the PDE
        state : Function
            The initial state of the system
        timestep : float
            The initial timestep to use for the method
        """
        z = state.copy(deepcopy=True)
        dt = firedrake.Constant(1.0)

        F = equation(z)
        z_n = z.copy(deepcopy=True)

        Z = z.function_space()
        w = firedrake.TestFunction(Z)
        form = inner(z_n - z, w) * dx - dt * F

        params = {"form_compiler_parameters": form_compiler_parameters}
        problem = Problem(form, z_n, **params)

        params = {"solver_parameters": solver_parameters or _solver_params(Z)}
        solver = Solver(problem, **params)

        self.state = z
        self.next_state = z_n
        self.timestep = dt
        self.solver = solver

    def step(self, timestep):
        self.timestep.assign(timestep)
        self.solver.solve()
        self.state.assign(self.next_state)


class IMEX(Integrator):
    def __init__(
        self,
        equation1,
        equation2,
        state,
        solver_parameters=None,
        form_compiler_parameters=None
    ):
        r"""A first-order implicit/explicit timestepping scheme for equations
        with both stiff and non-stiff dynamics

        Parameters
        ----------
        equation1
            A Python function that takes in the state variable and returns a
            firedrake.Form object describing the weak form of the non-stiff
            part of the PDE, e.g. advection
        equation2
            Same as equation1 but for the stiff part of the PDE, e.g. friction
        state : firedrake.Function
            Initial state of the system
        timestep : float
            The initial timestep to use for the method
        """
        z = state.copy(deepcopy=True)
        dt = firedrake.Constant(1.0)

        F = equation1(z)
        z_n = z.copy(deepcopy=True)
        G = equation2(z_n)

        Z = z.function_space()
        w = firedrake.TestFunction(Z)
        form = inner(z_n - z, w) * dx - dt * (F + G)

        params = {"form_compiler_parameters": form_compiler_parameters}
        problem = Problem(form, z_n, **params)

        params = {"solver_parameters": solver_parameters or _solver_params(Z)}
        solver = Solver(problem, **params)

        self.state = z
        self.next_state = z_n
        self.timestep = dt
        self.solver = solver

    def step(self, timestep):
        self.timestep.assign(timestep)
        self.solver.solve()
        self.state.assign(self.next_state)


class SSPRK33(Integrator):
    def __init__(
        self,
        equation,
        state,
        solver_parameters=None,
        form_compiler_parameters=None
    ):
        r"""A third-order, three-stage, explicit Runge-Kutta scheme

        Parameters
        ----------
        equation
            A Python function that takes in the state variable and returns a
            firedrake.Form object describing the weak form of the PDE
        state : firedrake.Function
            The initial state of the system
        timestep : float
            The initial timestep to use for the method
        """
        z = state.copy(deepcopy=True)
        dt = firedrake.Constant(1.0)

        num_stages = 3
        zs = [state.copy(deepcopy=True) for stage in range(num_stages)]
        Fs = [equation(z), equation(zs[0]), equation(zs[1])]

        Z = z.function_space()
        w = firedrake.TestFunction(Z)
        forms = [
            inner(zs[0] - z, w) * dx - dt * Fs[0],
            inner(zs[1] - (3 * z + zs[0]) / 4, w) * dx - dt / 4 * Fs[1],
            inner(zs[2] - (z + 2 * zs[1]) / 3, w) * dx - 2 * dt / 3 * Fs[2]
        ]

        params = {"form_compiler_parameters": form_compiler_parameters}
        problems = [Problem(form, zk, **params) for form, zk in zip(forms, zs)]

        params = {"solver_parameters": solver_parameters or _solver_params(Z)}
        solvers = [Solver(problem, **params) for problem in problems]

        self.state = z
        self.stages = zs
        self.timestep = dt
        self.solvers = solvers

    def step(self, timestep):
        self.timestep.assign(timestep)
        for solver in self.solvers:
            solver.solve()
        self.state.assign(self.stages[-1])


class SSPRK34(Integrator):
    def __init__(
        self,
        equation,
        state,
        solver_parameters=None,
        form_compiler_parameters=None
    ):
        r"""A third-order, four-stage, explicit Runge-Kutta scheme

        This scheme has a larger region of absolute stability than the
        SSPRK33 scheme; the CFL limit is 2 rather than 1. While this
        scheme is more expensive, you can use half as many timesteps while
        preserving stability.

        Parameters
        ----------
        equation
            A Python function that takes in the state variable and returns
            a Form object describing the weak form of the PDE
        state : Function
            The initial state of the system
        timestep : float
            The initial timestep to use for the method
        """
        z = state.copy(deepcopy=True)
        dt = firedrake.Constant(1.0)

        num_stages = 4
        zs = [state.copy(deepcopy=True) for stage in range(num_stages)]
        Fs = [equation(z), equation(zs[0]), equation(zs[1]), equation(zs[2])]

        Z = z.function_space()
        w = firedrake.TestFunction(Z)
        forms = [
            inner(zs[0] - z, w) * dx - dt / 2 * Fs[0],
            inner(zs[1] - zs[0], w) * dx - dt / 2 * Fs[1],
            inner(zs[2] - (2 * z + zs[1]) / 3, w) * dx - dt / 6 * Fs[2],
            inner(zs[3] - zs[2], w) * dx - dt / 2 * Fs[3]
        ]

        params = {"form_compiler_parameters": form_compiler_parameters}
        problems = [Problem(form, zk, **params) for form, zk in zip(forms, zs)]

        params = {"solver_parameters": solver_parameters or _solver_params(Z)}
        solvers = [Solver(problem, **params) for problem in problems]

        self.state = z
        self.stages = zs
        self.timestep = dt
        self.solvers = solvers

    def step(self, timestep):
        self.timestep.assign(timestep)
        for solver in self.solvers:
            solver.solve()
        self.state.assign(self.stages[-1])


class ThetaMethod:
    def __init__(
        self,
        theta,
        equation,
        state,
        conserved_variables=lambda z: z,
        solver_parameters=None,
        form_compiler_parameters=None,
    ):
        z = state.copy(deepcopy=True)
        dt = firedrake.Constant(1.0)

        z_n = z.copy(deepcopy=True)
        Z = z.function_space()
        w = firedrake.TestFunction(Z)

        q_n = conserved_variables(z_n)
        q = conserved_variables(z)

        # The code below is a heinous dirty hack to work around the fact that
        # we'd like to (but cannot) write
        #
        #     F = equation((1 - theta) * z + theta * z_n)
        #
        # because `(1 - theta) * z + theta * z_n` doesn't have a well-defined
        # function space.
        F = firedrake.replace(equation(z), {z: (1 - theta) * z + theta * z_n})
        dQ = inner(q_n - q, w) * dx

        form = dQ - dt * F
        problem = Problem(form, z_n, form_compiler_parameters=form_compiler_parameters)
        solver = Solver(problem, solver_parameters=solver_parameters)

        self.state = z
        self.next_state = z_n
        self.timestep = dt
        self.solver = solver

    def step(self, timestep):
        self.timestep.assign(timestep)
        self.solver.solve()
        self.state.assign(self.next_state)


def ImplicitEuler(*args, **kwargs):
    return ThetaMethod(1.0, *args, **kwargs)


def ImplicitMidpoint(*args, **kwargs):
    return ThetaMethod(0.5, *args, **kwargs)
