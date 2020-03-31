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

        # Create a solver object
        problem = LinearVariationalProblem(M, dD_dt, δD)
        self.mass_solver = LinearVariationalSolver(problem, **_parameters)

    def _init_momentum_solver(self):
        # Create the momentum sources (gravity and friction) as a function of
        # the other fields and input parameters
        g = self.model.gravity(**self.fields, **self.inputs)
        sources = {'gravity': g}

        # Create the finite element mass matrix
        V = self.fields['velocity'].function_space()
        u, v = firedrake.TestFunction(V), firedrake.TrialFunction(V)
        M = inner(u, v) * dx

        # Create the right-hand side of the momentum transport equation
        transport = self.model.momentum_transport
        du_dt = transport.du_dt(**self.fields, **self.inflow, **sources)

        # Create a variable to store the change in momentum from one timestep
        # to the next; this is what the solver actually computes
        δDu = firedrake.Function(V)
        self.momentum_change = δDu

        # Create a solver object
        problem = LinearVariationalProblem(M, du_dt, δDu)
        self.momentum_solver = LinearVariationalSolver(problem, **_parameters)

    def _init_temperature_solver(self):
        pass

    def _init_salinity_solver(self):
        pass

    def step(self, timestep):
        r"""Advance the solution forward by a timestep of length `dt`"""
        dt = firedrake.Constant(timestep)

        self.mass_solver.solve()
        self.momentum_solver.solve()

        D = self.fields['thickness']
        δD = self.thickness_change
        D_new = D + dt * δD

        u = self.fields['velocity']
        δDu = self.momentum_change
        u.project((D * u + dt * δDu) / D_new)

        D.assign(D_new)
