r"""Model description for the 2D shallow water equations"""

import firedrake
from firedrake import sqrt, inner, outer, grad, dx, ds
from ..numerics import forms


def _fluxes(h, q, g):
    r"""Calculate the flux of mass and momentum for the shallow water
    equations"""
    I = firedrake.Identity(2)
    F_h = q
    F_q = outer(q, q) / h + 0.5 * g * h**2 * I
    return F_h, F_q


def _boundary_flux(z, h_ext, q_ext, g, boundary_ids):
    Z = z.function_space()
    n = firedrake.FacetNormal(Z.mesh())
    φ, v = firedrake.TestFunctions(Z)

    F_hx, F_qx = _fluxes(h_ext, q_ext, g)

    h, q = firedrake.split(z)
    F_h, F_q = _fluxes(h, q, g)

    return 0.5 * (
        inner(F_hx, φ * n) +
        inner(F_qx, outer(v, n)) +
        inner(F_h, φ * n) +
        inner(F_q, outer(v, n))
    ) * ds(boundary_ids)


def _wall_flux(z, g, boundary_ids):
    n = firedrake.FacetNormal(z.ufl_domain())
    h, q = firedrake.split(z)
    # Mirror value of the fluid momentum
    q_ex = q - 2 * inner(q, n) * n
    return _boundary_flux(z, h, q_ex, g, boundary_ids)


def make_equation(g, b, **kwargs):
    outflow_ids = kwargs.get('outflow_ids', ())
    inflow_ids = kwargs.get('inflow_ids', ())
    h_in = kwargs.get('h_in', firedrake.Constant(0.0))
    q_in = kwargs.get('q_in', firedrake.Constant((0.0, 0.0)))

    def equation(z):
        Z = z.function_space()
        φ, v = firedrake.TestFunctions(Z)
        h, q = firedrake.split(z)
        F_h, F_q = _fluxes(h, q, g)

        mesh = Z.mesh()
        n = firedrake.FacetNormal(mesh)
        c = abs(inner(q / h, n)) + sqrt(g * h)

        sources = -inner(g * h * grad(b), v) * dx

        fluxes = (
            forms.cell_flux(F_h, φ) +
            forms.central_facet_flux(F_h, φ) +
            forms.lax_friedrichs_facet_flux(h, c, φ) +
            forms.cell_flux(F_q, v) +
            forms.central_facet_flux(F_q, v) +
            forms.lax_friedrichs_facet_flux(q, c, v)
        )

        boundary_ids = set(mesh.exterior_facets.unique_markers)
        wall_ids = tuple(boundary_ids - set(outflow_ids) - set(inflow_ids))
        boundary_fluxes = (
            _boundary_flux(z, h, q, g, outflow_ids) +
            _boundary_flux(z, h_in, q_in, g, inflow_ids) +
            _boundary_flux(z, h, q - 2 * inner(q, n) * n, g, wall_ids)
        )

        return sources - fluxes - boundary_fluxes

    return equation
