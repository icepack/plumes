import firedrake
from firedrake import (inner, outer, grad, dx, ds, dS, sqrt,
                       min_value, max_value, Constant)

class MassTransport(object):
    def dD_dt(self, **kwargs):
        D = kwargs['thickness']
        u = kwargs['velocity']
        e = kwargs['entrainment']
        m = kwargs['melt']
        D_inflow = kwargs['thickness_inflow']

        Q = D.function_space()
        φ = firedrake.TestFunction(Q)

        # Upwinding for stability
        n = firedrake.FacetNormal(D.ufl_domain())
        u_n = 0.5 * (inner(u, n) + abs(inner(u, n)))
        f = D * u_n

        cell_flux = -inner(D * u, grad(φ)) * dx
        face_flux = (f('+') - f('-')) * (φ('+') - φ('-')) * dS
        flux_in = D_inflow * min_value(inner(u, n), 0) * φ * ds
        flux_out = D * max_value(inner(u, n), 0) * φ * ds

        sources = (e + m) * φ * dx

        return sources - (cell_flux + face_flux + flux_in + flux_out)


class MomentumTransport(object):
    def __init__(self, friction=2.5e-3):
        self.friction = friction

    def du_dt(self, **kwargs):
        D = kwargs['thickness']
        u = kwargs['velocity']
        g = kwargs['gravity']
        u_inflow = kwargs['velocity_inflow']

        V = u.function_space()
        v = firedrake.TestFunction(V)

        mesh = V.mesh()
        n = firedrake.FacetNormal(mesh)
        u_n = 0.5 * (inner(u, n) + abs(inner(u, n)))
        f = D * u * u_n

        cell_flux = -inner(D * outer(u, u), grad(v)) * dx
        face_flux = inner(f('+') - f('-'), v('+') - v('-')) * dS
        flux_in = D * inner(u_inflow, v) * min_value(inner(u, n), 0) * ds
        flux_out = D * inner(u, v) * max_value(inner(u, n), 0) * ds

        k = self.friction
        friction = -k * sqrt(inner(u, u)) * inner(u, v) * dx
        gravity = D * inner(g, v) * dx
        sources = friction + gravity

        return sources - (cell_flux + face_flux + flux_in + flux_out)


class PlumeModel(object):
    def __init__(
        self,
        mass_transport=MassTransport(),
        momentum_transport=MomentumTransport()
    ):
        self.mass_transport = mass_transport
        self.momentum_transport = momentum_transport

    def entrainment(self, **kwargs):
        u = kwargs['velocity']
        z_b = kwargs['ice_shelf_draft']
        return E_0 * inner(u, grad(z_b))

    def melt(self, **kwargs):
        # Miracle occurs...
        return Constant(0.)
