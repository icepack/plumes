r"""Classes for integrating models forward in time"""

from abc import ABC, abstractmethod
import firedrake
from firedrake import (
    inner,
    dx,
    NonlinearVariationalProblem as Problem,
    NonlinearVariationalSolver as Solver
)

class Integrator(ABC):
    @abstractmethod
    def step(self, timestep):
        r"""Propagate the model forward by a single timestep"""
        pass


def _default_solver_parameters(Q):
    block_parameters = {
        'ksp_type': 'preonly',
        'pc_type': 'ilu',
        'sub_pc_type': 'bjacobi'
    }

    if not hasattr(Q, 'num_sub_spaces'):
        return block_parameters

    fieldsplits = {
        f'fieldsplit_{index}': block_parameters
        for index in range(Q.num_sub_spaces())
    }

    return dict(ksp_type='preonly', pc_type='fieldsplit', **fieldsplits)


class ExplicitEuler(Integrator):
    def __init__(
        self,
        equation,
        state,
        timestep,
        solver_parameters=None
    ):
        r"""A first-order explicit timestepping scheme

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
        F = equation(z)

        z_n = z.copy(deepcopy=True)
        Z = z.function_space()
        w = firedrake.TestFunction(Z)

        dt = firedrake.Constant(timestep)

        problem = Problem(inner(z_n - z, w) * dx - dt * F, z_n)

        if solver_parameters is None:
            solver_parameters = _default_solver_parameters(Z)
        solver = Solver(problem, solver_parameters=solver_parameters)

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
        timestep,
        solver_parameters=None
    ):
        r"""A third-order, three-stage, explicit Runge-Kutta scheme"""
        z = state.copy(deepcopy=True)
        dt = firedrake.Constant(timestep)

        num_stages = 3
        zs = [state.copy(deepcopy=True) for stage in range(num_stages)]
        Fs = [equation(z), equation(zs[0]), equation(zs[1])]

        Z = z.function_space()
        if solver_parameters is None:
            solver_parameters = _default_solver_parameters(Z)

        w = firedrake.TestFunction(Z)
        problems = [
            Problem(inner(zs[0] - z, w) * dx - dt * Fs[0], zs[0]),
            Problem(
                inner(zs[1] - (3 * z + zs[0]) / 4, w) * dx - dt / 4 * Fs[1], zs[1]
            ),
            Problem(
                inner(zs[2] - (z + 2 * zs[1]) / 3, w) * dx - 2 * dt / 3 * Fs[2], zs[2]
            )
        ]
        solvers = [
            Solver(problem, solver_parameters=solver_parameters)
            for problem in problems
        ]

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
        timestep,
        solver_parameters=None
    ):
        r"""A third-order, four-stage, explicit Runge-Kutta scheme"""
        z = state.copy(deepcopy=True)
        dt = firedrake.Constant(timestep)

        num_stages = 4
        zs = [state.copy(deepcopy=True) for stage in range(num_stages)]
        Fs = [equation(z), equation(zs[0]), equation(zs[1]), equation(zs[2])]

        Z = z.function_space()
        if solver_parameters is None:
            solver_parameters = _default_solver_parameters(Z)

        w = firedrake.TestFunction(Z)

        rhs = [
            inner(zs[0] - z, w) * dx - dt / 2 * Fs[0],
            inner(zs[1] - zs[0], w) * dx - dt / 2 * Fs[1],
            inner(zs[2] - (2 * z + zs[1]) / 3, w) * dx - dt / 6 * Fs[2],
            inner(zs[3] - zs[2], w) * dx - dt / 2 * Fs[3]
        ]

        problems = [Problem(rhs_k, z_k) for rhs_k, z_k in zip(rhs, zs)]
        solvers = [
            Solver(problem, solver_parameters=solver_parameters)
            for problem in problems
        ]

        self.state = z
        self.stages = zs
        self.timestep = dt
        self.solvers = solvers

    def step(self, timestep):
        self.timestep.assign(timestep)
        for solver in self.solvers:
            solver.solve()
        self.state.assign(self.stages[-1])
