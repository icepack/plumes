import firedrake
from firedrake import Constant, sqrt, min_value, max_value, inner, grad, dx, ds
from . import forms, thin_film
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
        β_T = self.coefficients["thermal_expansion"]
        β_S = self.coefficients["haline_contraction"]
        S_a = inputs["salinity_ambient"]
        T_a = inputs["temperature_ambient"]

        h, u, T, S = firedrake.split(fields)
        return β_S * (S_a - S) - β_T * (T_a - T)

    def entrainment_rate(self, fields, **inputs):
        r"""Return the rate at which the plume entrains ambient ocean water"""
        E_0 = self.coefficients["entrainment"]
        z_d = inputs["ice_shelf_draft"]

        h, u = firedrake.split(fields)[:2]
        return E_0 * max_value(0, inner(u, grad(z_d)))

    def melting_temperature(self, fields, **inputs):
        r"""Calculate the local melting temperature under the ice shelf, which
        depends on salinity and depth"""
        λ1 = self.coefficients["freezing_point_salinity"]
        λ2 = self.coefficients["freezing_point_offset"]
        λ3 = self.coefficients["freezing_point_depth"]
        z_d = inputs["ice_shelf_draft"]

        h, u, T, S = firedrake.split(fields)
        return λ1 * S + λ2 + λ3 * z_d

    def melt_rate(self, fields, **inputs):
        r"""Calculate the local rate of sub-ice shelf melt"""
        c_w = self.coefficients["ocean_heat_capacity"]
        c_i = self.coefficients["ice_heat_capacity"]
        L = self.coefficients["latent_heat"]
        γ_TS = self.coefficients["effective_heat_exchange"]

        h, u, T = firedrake.split(fields)[:3]
        T_f = self.melting_temperature(fields, **inputs)

        U = inner(u, u)**0.5
        return c_w * γ_TS * U * (T - T_f) / (L + c_i * T_f)


def conserved_variables(z):
    h, u, T, S = firedrake.split(z)
    return firedrake.as_vector((h, 0.0, 0.0, h * T, h * S))


def temperature_equation(physics=PlumePhysics(), **kwargs):
    inflow_ids = kwargs.get("inflow_ids", ())
    outflow_ids = kwargs.get("outflow_ids", ())
    h_in = kwargs.get("thickness_in")
    T_in = kwargs.get("temperature_in")
    T_a = kwargs["temperature_ambient"]

    def equation(z):
        Z = z.function_space()
        h, u, T, S = firedrake.split(z)
        φ = firedrake.TestFunctions(Z)[2]

        f_T = h * T * u
        n = firedrake.FacetNormal(Z.mesh())
        c = abs(inner(u, n))

        fluxes = (
            forms.cell_flux(f_T, φ)
            + forms.central_facet_flux(f_T, φ)
            + forms.lax_friedrichs_facet_flux(h * T, c, φ)
            + forms.lax_friedrichs_boundary_flux(h * T, h * T, c, φ, outflow_ids)
            + forms.lax_friedrichs_boundary_flux(h * T, h_in * T_in, c, φ, inflow_ids)
            + h * T * max_value(0, inner(u, n)) * φ * ds(outflow_ids)
            + h_in * T_in * min_value(0, inner(u, n)) * φ * ds(inflow_ids)
        )

        e = physics.entrainment_rate(z, **kwargs)
        m = physics.melt_rate(z, **kwargs)
        T_f = physics.melting_temperature(fields, **kwargs)
        sources = (e * T_a + m * T_f) * φ * dx

        γ_T = physics.coefficients["turbulent_heat_exchange"]
        U = inner(u, u)**0.5
        sinks = γ_T * U * (T - T_f) * φ * dx

        return sources - sinks - fluxes

    return equation


def salinity_equation(physics=PlumePhysics(), **kwargs):
    inflow_ids = kwargs.get("inflow_ids", ())
    outflow_ids = kwargs.get("outflow_ids", ())
    h_in = kwargs.get("thickness_in")
    S_in = kwargs.get("salinity_in")
    S_a = kwargs["salinity_ambient"]

    def equation(z):
        Z = z.function_space()
        h, u, T, S = firedrake.split(z)
        φ = firedrake.TestFunctions(Z)[3]

        f_S = h * S * u
        n = firedrake.FacetNormal(Z.mesh())
        c = abs(inner(u, n))

        fluxes = (
            forms.cell_flux(f_S, φ)
            + forms.central_facet_flux(f_S, φ)
            + forms.lax_friedrichs_facet_flux(h * S, c, φ)
            + forms.lax_friedrichs_boundary_flux(h * S, h * S, c, φ, outflow_ids)
            + forms.lax_friedrichs_boundary_flux(h * S, h_in * S_in, c, φ, inflow_ids)
            + h * S * max_value(0, inner(u, n)) * φ * ds(outflow_ids)
            + h_in * S_in * min_value(0, inner(u, n)) * φ * ds(inflow_ids)
        )

        e = physics.entrainment_rate(z, **kwargs)
        sources = e * S_a * φ * dx

        return sources - fluxes

    return equation


def make_equation(physics=PlumePhysics(), **kwargs):
    g = physics.coefficients["gravity"]
    μ = physics.coefficients["viscosity"]
    k = physics.coefficients["friction"]
    z_d = kwargs["ice_shelf_draft"]
    S_eqn = salinity_equation(physics=physics, **kwargs)
    T_eqn = salinity_equation(physics=physics, **kwargs)

    def equation(z):
        δlnρ = physics.density_contrast(z, **kwargs)
        e = physics.entrainment_rate(z, **kwargs)
        m = physics.melt_rate(z, **kwargs)
        # TODO: double-check the signs on the gravitational forcing
        # NOTE: We're using the *kinematic* viscosity and the *relative* density
        # over the background ocean rather than the absolute viscosity and
        # density. The kinematic viscosity and relative density have been
        # divided by the reference water density.
        thin_film_kwargs = {
            "gravity": g,
            "density": δlnρ,
            "bed": -z_d,
            "accumulation": e + m,
            "viscosity": Constant(μ),
            "friction": Constant(k),
            "friction_exponent": 1,
            "thickness_in": kwargs["thickness_in"],
            "velocity_in": kwargs["velocity_in"],
            "inflow_ids": kwargs["inflow_ids"],
            "outflow_ids": kwargs["outflow_ids"],
        }
        hu_eqn = thin_film.make_equation(physics=physics, **thin_film_kwargs)
        return hu_eqn(z) + T_eqn(z) + S_eqn(z)

    return equation
