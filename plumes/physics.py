import firedrake
from firedrake import inner, grad, dx, ds, dS, min_value, max_value

class MassTransport(object):
    def dD_dt(self, D, u, e, m, D_inflow):
        Q = D.function_space()
        φ = firedrake.TestFunction(Q)

        # Upwinding for stability
        mesh = Q.mesh()
        n = firedrake.FacetNormal(D.ufl_domain())
        u_n = 0.5 * (inner(u, n) + abs(inner(u, n)))
        f = D * u_n

        cell_flux = -inner(D * u, grad(φ)) * dx
        face_flux = (f('+') - f('-')) * (φ('+') - φ('-')) * dS
        flux_in = D_inflow * min_value(inner(u, n), 0) * φ * ds
        flux_out = D * max_value(inner(u, n), 0) * φ * ds

        sources = (e + m) * φ * dx

        return sources - (cell_flux + face_flux + flux_in + flux_out)
