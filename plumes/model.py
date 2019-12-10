import firedrake
from firedrake import inner, dx
from .physics import MassTransport, MomentumTransport

class PlumeModel(object):
    def __init__(self):
        self.parameters = {
            'solver_parameters': {
                'ksp_type': 'preonly',
                'pc_type': 'bjacobi',
                'sub_pc_type': 'ilu'
            }
        }

        self.mass_transport = MassTransport()
        self.momentum_transport = MomentumTransport()

    def mass_transport_solve(self, dt, D, u, e, m, D_inflow):
        Q = D.function_space()
        φ, ψ = firedrake.TestFunction(Q), firedrake.TrialFunction(Q)
        M = φ * ψ * dx

        # Explicit Euler scheme. TODO: SSPRK3
        dD_dt = self.mass_transport.dD_dt(D, u, e, m, D_inflow)
        F = D * φ * dx + dt * dD_dt
        firedrake.solve(M == F, D, **self.parameters)

    def solve(self, dt, D, u, e, m, g, D_inflow, u_inflow):
        D_n = D.copy(deepcopy=True)
        self.mass_transport_solve(dt, D, u, e, m, D_inflow)

        V = u.function_space()
        v, w = firedrake.TestFunction(V), firedrake.TrialFunction(V)
        M = D * inner(v, w) * dx

        du_dt = self.momentum_transport.du_dt(D, u, g, u_inflow)
        F = D_n * inner(u, v) * dx + dt * du_dt
        firedrake.solve(M == F, u, **self.parameters)
