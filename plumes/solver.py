from enum import Flag, auto
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


class Component(Flag):
    Mass = auto()
    Momentum = auto()
    Heat = auto()
    Salt = auto()
    All = Mass | Momentum | Heat | Salt


class PlumeSolver(object):
    def __init__(self, model, components=Component.All, **kwargs):
        self.model = model
        self.components = components

        field_names = ('thickness', 'velocity', 'temperature', 'salinity')
        self.fields = {
            name: kwargs[name].copy(deepcopy=True) for name in field_names
        }

        self.inflow = {
            name + '_inflow': kwargs[name + '_inflow']
            for name in field_names
        }

        self.inputs = {
            'ice_shelf_base': kwargs['ice_shelf_base'],
            'salinity_ambient': kwargs['salinity_ambient'],
            'temperature_ambient': kwargs['temperature_ambient']
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
        e = self.model.entrainment(**self.fields, **self.inputs)
        m = self.model.melt(**self.fields, **self.inputs)
        T_f = self.model.freezing_temperature(**self.fields, **self.inputs)
        T_a = self.inputs['temperature_ambient']
        sources = {
            'entrainment': e,
            'melt': m,
            'temperature_ambient': T_a,
            'freezing_temperature': T_f
        }

        T = self.fields['temperature']
        Q = T.function_space()
        φ, ψ = firedrake.TestFunction(Q), firedrake.TrialFunction(Q)
        M = φ * ψ * dx

        transport = self.model.heat_transport
        dT_dt = transport.dT_dt(**self.fields, **self.inflow, **sources)

        δDT = firedrake.Function(Q)
        self.temperature_change = δDT

        problem = LinearVariationalProblem(M, dT_dt, δDT)
        self.heat_solver = LinearVariationalSolver(problem, **_parameters)

    def _init_salinity_solver(self):
        e = self.model.entrainment(**self.fields, **self.inputs)
        S_a = self.inputs['salinity_ambient']
        sources = {'entrainment': e, 'salinity_ambient': S_a}

        S = self.fields['salinity']
        Q = S.function_space()
        φ, ψ = firedrake.TestFunction(Q), firedrake.TrialFunction(Q)
        M = φ * ψ * dx

        # Create the right-hand side of the transport equation
        transport = self.model.salt_transport
        dS_dt = transport.dS_dt(**self.fields, **self.inflow, **sources)

        δDS = firedrake.Function(Q)
        self.salinity_change = δDS

        problem = LinearVariationalProblem(M, dS_dt, δDS)
        self.salt_solver = LinearVariationalSolver(problem, **_parameters)

    def step(self, timestep):
        r"""Advance the solution forward by a timestep of length `dt`"""
        dt = firedrake.Constant(timestep)

        if self.components & Component.Mass:
            self.mass_solver.solve()

        if self.components & Component.Momentum:
            self.momentum_solver.solve()

        if self.components & Component.Salt:
            self.salt_solver.solve()

        if self.components & Component.Heat:
            self.heat_solver.solve()

        D = self.fields['thickness']
        δD = self.thickness_change
        D_new = D + dt * δD

        if self.components & Component.Momentum:
            # TODO: Make projector objects for all of these operations
            u = self.fields['velocity']
            δDu = self.momentum_change
            u.project((D * u + dt * δDu) / D_new)

        if self.components & Component.Salt:
            S = self.fields['salinity']
            δDS = self.salinity_change
            S.project((D * S + dt * δDS) / D_new)

        if self.components & Component.Heat:
            T = self.fields['temperature']
            δDT = self.temperature_change
            T.project((D * T + dt * δDT) / D_new)

        D.assign(D_new)
