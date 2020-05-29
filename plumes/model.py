import firedrake
from firedrake import (inner, outer, grad, dx, ds, dS, sqrt,
                       min_value, max_value, Constant)
from . import coefficients

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
    def __init__(self, friction=coefficients.friction):
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

        # This term is very stiff, we should try splitting and doing it
        # implicitly.
        k = self.friction
        friction = -k * sqrt(inner(u, u)) * inner(u, v) * dx
        gravity = D * inner(g, v) * dx
        sources = friction + gravity

        return sources - (cell_flux + face_flux + flux_in + flux_out)


class HeatTransport(object):
    def __init__(self, friction=coefficients.friction):
        self.friction = friction

    def dT_dt(self, **kwargs):
        D = kwargs['thickness']
        T = kwargs['temperature']
        u = kwargs['velocity']

        Γ_T = coefficients.turbulent_heat_exchange

        D_inflow = kwargs['thickness_inflow']
        T_inflow = kwargs['temperature_inflow']

        e = kwargs['entrainment']
        T_a = kwargs['temperature_ambient']
        m = kwargs['melt']
        T_f = kwargs['freezing_temperature']

        Q = T.function_space()
        φ = firedrake.TestFunction(Q)

        mesh = Q.mesh()
        n = firedrake.FacetNormal(mesh)
        u_n = 0.5 * (inner(u, n) + abs(inner(u, n)))
        f = D * T * u_n

        cell_flux = -inner(D * T * u, grad(φ)) * dx
        face_flux = (f('+') - f('-')) * (φ('+') - φ('-')) * dS
        flux_in = D_inflow * T_inflow * min_value(inner(u, n), 0) * φ * ds
        flux_out = D * T * max_value(inner(u, n), 0) * φ * ds

        exchange = (e * T_a + m * T_f) * φ * dx
        friction = -Γ_T * sqrt(inner(u, u)) * (T - T_f) * φ * dx
        sources = exchange + friction

        return sources - (cell_flux + face_flux + flux_in + flux_out)


class SaltTransport(object):
    def dS_dt(self, **kwargs):
        D = kwargs['thickness']
        S = kwargs['salinity']
        u = kwargs['velocity']

        D_inflow = kwargs['thickness_inflow']
        S_inflow = kwargs['salinity_inflow']

        e = kwargs['entrainment']
        S_a = kwargs['salinity_ambient']

        Q = S.function_space()
        φ = firedrake.TestFunction(Q)

        mesh = Q.mesh()
        n = firedrake.FacetNormal(mesh)
        u_n = 0.5 * (inner(u, n) + abs(inner(u, n)))
        f = D * S * u_n

        cell_flux = -inner(D * S * u, grad(φ)) * dx
        face_flux = (f('+') - f('-')) * (φ('+') - φ('-')) * dS
        flux_in = D_inflow * S_inflow * min_value(inner(u, n), 0) * φ * ds
        flux_out = D * S * max_value(inner(u, n), 0) * φ * ds

        sources = e * S_a * φ * dx

        return sources - (cell_flux + face_flux + flux_in + flux_out)


class PlumeModel(object):
    def __init__(
        self,
        mass_transport=MassTransport(),
        momentum_transport=MomentumTransport(),
        heat_transport=HeatTransport(),
        salt_transport=SaltTransport()
    ):
        self.mass_transport = mass_transport
        self.momentum_transport = momentum_transport
        self.heat_transport = heat_transport
        self.salt_transport = salt_transport

    def entrainment(self, **kwargs):
        u = kwargs['velocity']
        z_b = kwargs['ice_shelf_base']
        E_0 = coefficients.entrainment
        return E_0 * inner(u, grad(z_b))

    def freezing_temperature(self, **kwargs):
        S = kwargs['salinity']
        z_b = kwargs['ice_shelf_base']
        λ1 = firedrake.Constant(coefficients.freezing_point_salinity)
        λ2 = firedrake.Constant(coefficients.freezing_point_offset)
        λ3 = firedrake.Constant(coefficients.freezing_point_depth)
        return λ1 * S + λ2 + λ3 * z_b

    def melt(self, **kwargs):
        T = kwargs['temperature']
        u = kwargs['velocity']

        # We use the simplified turbulent heat exchange parameterization from
        # Lazeroms et al. 2018, equation 6a). See also Jenkins et al. 2010.
        Γ_TS = coefficients.effective_heat_exchange
        L = coefficients.latent_heat
        C_W = coefficients.ocean_heat_capacity
        C_I = coefficients.ice_heat_capacity

        # Equation 6a) in Lazeroms 2018
        U = sqrt(inner(u, u))
        T_f = self.freezing_temperature(**kwargs)
        dT = T - T_f
        return C_W * Γ_TS * U * dT / (L + C_I * T_f)

    def density_contrast(self, **kwargs):
        T = kwargs['temperature']
        S = kwargs['salinity']

        β_T = firedrake.Constant(coefficients.thermal_expansion)
        β_S = firedrake.Constant(coefficients.haline_contraction)

        S_a = kwargs['salinity_ambient']
        T_a = kwargs['temperature_ambient']

        return β_S * (S_a - S) - β_T * (T_a - T)

    def gravity(self, **kwargs):
        z_b = kwargs['ice_shelf_base']
        δρ = self.density_contrast(**kwargs)
        g = coefficients.gravity
        return δρ * g * grad(z_b)
