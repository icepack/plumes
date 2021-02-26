r"""Model description for the 2D shallow water equations"""

import firedrake
from firedrake import sqrt, inner, outer, grad, dx, ds
from . import forms


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
    φ, v = firedrake.TestFunctions(Z)[:2]

    F_hx, F_qx = _fluxes(h_ext, q_ext, g)

    h, q = firedrake.split(z)[:2]
    F_h, F_q = _fluxes(h, q, g)

    return 0.5 * (
        inner(F_hx, φ * n) +
        inner(F_qx, outer(v, n)) +
        inner(F_h, φ * n) +
        inner(F_q, outer(v, n))
    ) * ds(boundary_ids)


def _wall_flux(z, g, boundary_ids):
    n = firedrake.FacetNormal(z.ufl_domain())
    h, q = firedrake.split(z)[:2]
    # Mirror value of the fluid momentum
    q_ex = q - 2 * inner(q, n) * n
    return _boundary_flux(z, h, q_ex, g, boundary_ids)


def make_equation(g, b, **kwargs):
    r"""Return a function that calculates the weak form of the nonlinear
    shallow water equations

    Parameters
    ----------
    g : float, Constant or Function
        The acceleration due to gravity
    b : Function or expression
        The seafloor bathymetry
    thickness_in : expression, optional
        The thickness at the inflow boundary
    momentum_in : expression, optional
        The momentum at the inflow boundary
    inflow_ids : tuple of int, optional
        The numeric IDs of the boundary segments where fluid is flowing in
    outflow_ids : tuple of int, optional
        The numeric IDs of the boundary segments where fluid is flowing out

    Returns
    -------
    equation
        A function that takes in an element of the mixed thickness-momentum
        function space and returns a Form object describing the weak form
    """
    outflow_ids = kwargs.get('outflow_ids', ())
    inflow_ids = kwargs.get('inflow_ids', ())
    h_in = kwargs.get('thickness_in', firedrake.Constant(0.0))
    q_in = kwargs.get('momentum_in', firedrake.Constant((0.0, 0.0)))

    def equation(z):
        Z = z.function_space()
        φ, v = firedrake.TestFunctions(Z)[:2]
        h, q = firedrake.split(z)[:2]
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
        q_wall = q - 2 * inner(q, n) * n
        q_out = firedrake.max_value(0, inner(q, n)) * n
        boundary_fluxes = (
            _boundary_flux(z, h, q_out, g, outflow_ids) +
            _boundary_flux(z, h_in, q_in, g, inflow_ids) +
            _boundary_flux(z, h, q_wall, g, wall_ids) +
            forms.lax_friedrichs_boundary_flux(h, h, c, φ, outflow_ids) +
            forms.lax_friedrichs_boundary_flux(q, q_out, c, v, outflow_ids) +
            forms.lax_friedrichs_boundary_flux(h, h_in, c, φ, inflow_ids) +
            forms.lax_friedrichs_boundary_flux(q, q_in, c, v, inflow_ids) +
            forms.lax_friedrichs_boundary_flux(h, h, c, φ, wall_ids) +
            forms.lax_friedrichs_boundary_flux(q, q_wall, c, v, wall_ids)
        )

        return sources - fluxes - boundary_fluxes

    return equation
