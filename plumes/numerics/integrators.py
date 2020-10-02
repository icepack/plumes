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
