import firedrake
from firedrake import inner, grad, dx, ds, dS, max_value, min_value

class PlumeModel(object):
    def __init__(self):
        self.parameters = {
            'solver_parameters': {
                'ksp_type': 'preonly',
                'pc_type': 'bjacobi',
                'sub_pc_type': 'ilu'
            }
        }

    def mass_transport_solve(self, dt, D, u, D_inflow):
        Q = D.function_space()
        mesh = Q.mesh()

        φ, ψ = firedrake.TestFunction(Q), firedrake.TrialFunction(Q)
        M = φ * ψ * dx

        n = firedrake.FacetNormal(mesh)
        u_n = 0.5 * (inner(u, n) + abs(inner(u, n)))
        f = D * u_n

        cell_flux = -inner(D * u, grad(φ)) * dx
        face_flux = (f('+') - f('-')) * (φ('+') - φ('-')) * dS
        flux_in = D_inflow * min_value(inner(u, n), 0) * φ * ds
        flux_out = D * max_value(inner(u, n), 0) * φ * ds

        F = D * φ * dx - dt * (cell_flux + face_flux + flux_in + flux_out)
        firedrake.solve(M == F, D, **self.parameters)
