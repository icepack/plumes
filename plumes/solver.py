import firedrake
from firedrake import (
    inner,
    dx,
    LinearVariationalProblem,
    LinearVariationalSolver
)
from .model import MassTransport, MomentumTransport

_parameters = {
    'solver_parameters': {
        'ksp_type': 'preonly',
        'pc_type': 'bjacobi',
        'sub_pc_type': 'ilu'
    }
}

class PlumeSolver(object):
    def __init__(self, model, **kwargs):
        self.model = model

        field_names = ('thickness', 'velocity', 'temperature', 'salinity')
        self.fields = {
            name: kwargs[name].copy(deepcopy=True) for name in field_names
        }

        self.inflow = {
            name + '_inflow': kwargs[name + '_inflow']
            for name in field_names
        }

        self.inputs = {
            'ice_shelf_base': kwargs['ice_shelf_base']
        }

        self._init_mass_solver()
        self._init_momentum_solver()
        self._init_temperature_solver()
        self._init_salinity_solver()

    def _init_mass_solver(self):
        # Create the entrainment and melt rate fields as functions of the other
        # fields and input parameters
        e = self.model.entrainment(**self.fields, **self.inputs)
        m = self.model.melt(**self.fields, **self.inputs)
        sources = {'entrainment': e, 'melt': m}

        # Create the finite element mass matrix
        D = self.fields['thickness']
        Q = D.function_space()
        φ, ψ = firedrake.TestFunction(Q), firedrake.TrialFunction(Q)
        M = φ * ψ * dx

        # Create the right-hand side of the transport equation
        transport = self.model.mass_transport
        dD_dt = transport.dD_dt(**self.fields, **self.inflow, **sources)

        # Create a variable to store the change in thickness from one timestep
        # to the next; this is what the solver actually computes
        δD = firedrake.Function(Q)
        self.thickness_change = δD

        # Create a solver object to store
        problem = LinearVariationalProblem(M, dD_dt, δD)
        self.mass_solver = LinearVariationalSolver(problem, **_parameters)

    def _init_momentum_solver(self):
        pass

    def _init_temperature_solver(self):
        pass

    def _init_salinity_solver(self):
        pass

    def step(self, dt):
        r"""Advance the solution forward by a timestep of length `dt`"""
        D = self.fields['thickness']
        self.mass_solver.solve()
        δD = self.thickness_change
        D.assign(D + dt * δD)
