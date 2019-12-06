import firedrake
from firedrake import dx
from .physics import MassTransport

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

    def mass_transport_solve(self, dt, D, u, e, m, D_inflow):
        Q = D.function_space()
        φ, ψ = firedrake.TestFunction(Q), firedrake.TrialFunction(Q)
        M = φ * ψ * dx

        # Explicit Euler scheme. TODO: SSPRK3
        F = D * φ * dx + dt * self.mass_transport.dD_dt(D, u, e, m, D_inflow)
        firedrake.solve(M == F, D, **self.parameters)
