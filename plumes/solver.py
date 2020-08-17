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

        self._thickness_next = self.fields['thickness'].copy(deepcopy=True)
        self._velocity_next = self.fields['velocity'].copy(deepcopy=True)
        self._temperature_next = self.fields['temperature'].copy(deepcopy=True)
        self._salinity_next = self.fields['salinity'].copy(deepcopy=True)

        self._timestep = firedrake.Constant(1.)

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
        mass = φ * ψ * dx

        # Create the right-hand side of the transport equation
        transport = self.model.mass_transport
        dD_dt = transport.dD_dt(**self.fields, **self.inflow, **sources)

        dt = self._timestep
        rhs = D * φ * dx + dt * dD_dt

        # Create a solver object
        D_next = self._thickness_next
        problem = LinearVariationalProblem(mass, rhs, D_next)
        self.mass_solver = LinearVariationalSolver(problem, **_parameters)

    def _init_momentum_solver(self):
        # Create the momentum sources (gravity and friction) as a function of
        # the other fields and input parameters
        g = self.model.gravity(**self.fields, **self.inputs)
        sources = {'gravity': g}

        u = self.fields['velocity']
        D = self.fields['thickness']
        D_next = self._thickness_next

        # Create the finite element mass matrix
        V = u.function_space()
        v, w = firedrake.TestFunction(V), firedrake.TrialFunction(V)
        mass = D_next * inner(v, w) * dx

        # Create the right-hand side of the momentum transport equation
        transport = self.model.momentum_transport
        du_dt = transport.du_dt(**self.fields, **self.inflow, **sources)

        dt = self._timestep
        rhs = D * inner(u, v) * dx + dt * du_dt

        # Create a solver object
        u_next = self._velocity_next
        problem = LinearVariationalProblem(mass, rhs, u_next)
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
        D = self.fields['thickness']
        D_next = self._thickness_next

        Q = T.function_space()
        φ, ψ = firedrake.TestFunction(Q), firedrake.TrialFunction(Q)
        mass = D_next * φ * ψ * dx

        transport = self.model.heat_transport
        dT_dt = transport.dT_dt(**self.fields, **self.inflow, **sources)

        dt = self._timestep
        rhs = D * T * φ * dx + dt * dT_dt

        T_next = self._temperature_next
        problem = LinearVariationalProblem(mass, rhs, T_next)
        self.heat_solver = LinearVariationalSolver(problem, **_parameters)

    def _init_salinity_solver(self):
        e = self.model.entrainment(**self.fields, **self.inputs)
        S_a = self.inputs['salinity_ambient']
        sources = {'entrainment': e, 'salinity_ambient': S_a}

        S = self.fields['salinity']
        D = self.fields['thickness']
        D_next = self._thickness_next

        Q = S.function_space()
        φ, ψ = firedrake.TestFunction(Q), firedrake.TrialFunction(Q)
        mass = D_next * φ * ψ * dx

        # Create the right-hand side of the transport equation
        transport = self.model.salt_transport
        dS_dt = transport.dS_dt(**self.fields, **self.inflow, **sources)

        dt = self._timestep
        rhs = D * S * φ * dx + dt * dS_dt

        S_next = self._salinity_next
        problem = LinearVariationalProblem(mass, rhs, S_next)
        self.salt_solver = LinearVariationalSolver(problem, **_parameters)

    def step(self, timestep):
        r"""Advance the solution forward by a timestep of length `dt`"""
        dt = self._timestep
        dt.assign(timestep)

        if self.components & Component.Mass:
            self.mass_solver.solve()

        if self.components & Component.Momentum:
            self.momentum_solver.solve()

        if self.components & Component.Salt:
            self.salt_solver.solve()

        if self.components & Component.Heat:
            self.heat_solver.solve()

        self.fields['thickness'].assign(self._thickness_next)
        self.fields['velocity'].assign(self._velocity_next)
        self.fields['temperature'].assign(self._temperature_next)
        self.fields['salinity'].assign(self._salinity_next)
