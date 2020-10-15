import firedrake
from firedrake import Constant, sqrt, min_value, max_value, inner, grad, dx, ds
from . import forms, shallow_water
from .. import coefficients


class PlumePhysics:
    def __init__(self, **kwargs):
        r"""Describes all the source terms and physical coefficients of the
        plume model -- the melt and entrainment rates, density contrast, etc.

        Parameters
        ----------
        kwargs: dict, optional
            Alternative values for any coefficients (see the module
            `coefficients.py` for a full listing)
        """

        # Well this is pretty heinous
        self.coefficients = {
            name: Constant(kwargs.get(name, value))
            for name, value in coefficients.__dict__.items()
            if type(value) is float
        }

    def density_contrast(self, fields, **inputs):
        r"""Calculate the relative density contrast of the meltwater plume over
        the ambient ocean"""
        β_T = self.coefficients['thermal_expansion']
        β_S = self.coefficients['haline_contraction']
        S_a = inputs['salinity_ambient']
        T_a = inputs['temperature_ambient']

        D, q, E, S = firedrake.split(fields)
        T = E / D
        s = S / D

        return β_S * (S_a - S) - β_T * (T_a - T)

    def entrainment_rate(self, fields, **inputs):
        r"""Calculate the rate of entrainment of ambient ocean water"""
        E_0 = self.coefficients['entrainment']
        z_d = inputs['ice_shelf_draft']

        D, q = firedrake.split(fields)[:2]
        u = q / D

        return E_0 * max_value(0, inner(u, grad(z_d)))

    def melting_temperature(self, fields, **inputs):
        r"""Calculate the local melting temperature of the ice shelf, which
        depends on salinity and depth"""
        λ1 = self.coefficients['freezing_point_salinity']
        λ2 = self.coefficients['freezing_point_offset']
        λ3 = self.coefficients['freezing_point_depth']
        z_d = inputs['ice_shelf_draft']

        D, q, E, S = firedrake.split(fields)
        s = S / D

        return λ1 * s + λ2 + λ3 * z_d

    def melt_rate(self, fields, **inputs):
        r"""Calculate the rate of ice shelf melting"""
        c_w = self.coefficients['ocean_heat_capacity']
        c_i = self.coefficients['ice_heat_capacity']
        L = self.coefficients['latent_heat']
        γ_TS = self.coefficients['effective_heat_exchange']

        D, q, E = firedrake.split(fields)[:3]
        u = q / D
        T = E / D
        T_f = self.melting_temperature(fields, **inputs)

        U = sqrt(inner(u, u))
        return c_w * γ_TS * U * (T - T_f) / (L + c_i * T_f)


def make_transport_equation(physics=PlumePhysics(), **kwargs):
    r"""Create the part of the plume model equations describing the transport
    of mass, momentum, heat, and salt, excluding sources and sinks

    These terms are non-stiff and should be solve explicitly.
    """
    # Get the boundary conditions
    outflow_ids = kwargs.get('outflow_ids', ())
    inflow_ids = kwargs.get('inflow_ids', ())
    E_in = kwargs.get('energy_in', Constant(0.0))
    S_in = kwargs.get('salinity_in', Constant(0.0))

    def equation(fields):
        Z = fields.function_space()
        D, q, E, S = firedrake.split(fields)
        φ, v, ψ, η = firedrake.TestFunctions(Z)

        # Create the relative density contrast. This affects the effective
        # gravity of the plume.
        δlnρ = physics.density_contrast(fields, **kwargs)

        # Create the shallow water-like part of the dynamics.
        swe_bcs = {
            key: kwargs[key] for key in
            ['thickness_in', 'momentum_in', 'inflow_ids', 'outflow_ids']
        }

        z_d = kwargs['ice_shelf_draft']
        g = physics.coefficients['gravity']
        swe = shallow_water.make_equation(δlnρ * g, -z_d, **swe_bcs)

        mesh = Z.mesh()
        n = firedrake.FacetNormal(mesh)
        u = q / D
        c = abs(inner(u, n))

        # Add the terms for advection of thermal energy and salt.
        F_E = E * u
        heat_fluxes = (
            forms.cell_flux(F_E, ψ) +
            forms.central_facet_flux(F_E, ψ) +
            forms.lax_friedrichs_facet_flux(E, c, ψ) +
            E * max_value(0, inner(u, n)) * ψ * ds(outflow_ids) +
            E_in * min_value(0, inner(u, n)) * ψ * ds(inflow_ids)
        )

        F_S = S * u
        salt_fluxes = (
            forms.cell_flux(F_S, η) +
            forms.central_facet_flux(F_S, η) +
            forms.lax_friedrichs_facet_flux(S, c, η) +
            S * max_value(0, inner(u, n)) * η * ds(outflow_ids) +
            S_in * min_value(0, inner(u, n)) * η * ds(inflow_ids)
        )

        return swe(fields) - heat_fluxes - salt_fluxes

    return equation


def make_sink_equation(physics=PlumePhysics(), **kwargs):
    r"""Create the parts of the plume model equations describing sinks --
    friction and turbulent heat exchange

    These terms are stiff should be solved implicitly using an IMEX scheme.
    """
    k = physics.coefficients['friction']
    γ_T = physics.coefficients['turbulent_heat_exchange']

    def equation(fields):
        D, q, E = firedrake.split(fields)[:3]
        v, ψ = firedrake.TestFunctions(fields.function_space())[1:3]

        u = q / D
        U = sqrt(inner(u, u))
        momentum_sink = k * U * inner(u, v)

        T = E / D
        T_f = physics.melting_temperature(fields, **kwargs)
        heat_sink = γ_T * U * (T - T_f) * ψ

        return -(momentum_sink + heat_sink) * dx

    return equation


def make_source_equation(physics=PlumePhysics(), **kwargs):
    r"""Create the parts of the plume model describing sources -- entrainment
    of ambient ocean water, melting of the ice shelf, etc.

    These terms are non-stiff and should be treated explicitly.
    """
    def equation(fields):
        Z = fields.function_space()
        φ, v, ψ, η = firedrake.TestFunctions(Z)

        T_a = kwargs['temperature_ambient']
        s_a = kwargs['salinity_ambient']

        e = physics.entrainment_rate(fields, **kwargs)
        m = physics.melt_rate(fields, **kwargs)
        T_f = physics.melting_temperature(fields, **kwargs)

        mass_sources = (e + m) * φ
        heat_sources = (e * T_a + m * T_f) * ψ
        salt_sources = e * s_a * η
        return (mass_sources + heat_sources + salt_sources) * dx

    return equation
