r"""Model description for the 2D shallow water equations"""

import firedrake
from firedrake import sqrt, inner, outer, grad, dx, ds
from . import forms


class MomentumForm:
    def fluxes(self, h, u, g):
        r"""Calculate the flux of mass and momentum for the shallow water
        equations"""
        I = firedrake.Identity(2)
        F_h = u
        F_u = outer(u, u) / h + 0.5 * g * h**2 * I
        return F_h, F_u

    def boundary_flux(self, z, h_ext, u_ext, g, boundary_ids):
        Z = z.function_space()
        n = firedrake.FacetNormal(Z.mesh())
        φ, v = firedrake.TestFunctions(Z)[:2]

        F_hx, F_ux = self.fluxes(h_ext, u_ext, g)

        h, u = firedrake.split(z)[:2]
        F_h, F_u = self.fluxes(h, u, g)

        return 0.5 * (
            inner(F_h + F_hx, φ * n) +
            inner(F_u + F_ux, outer(v, n))
        ) * ds(boundary_ids)

    def wave_speed(self, h, u, g, n):
        return abs(inner(u / h, n)) + sqrt(g * h)


class VelocityForm:
    def fluxes(self, h, u, g):
        I = firedrake.Identity(2)
        F_h = h * u
        F_u = h * outer(u, u) + 0.5 * g * h ** 2 * I
        return F_h, F_u

    def boundary_flux(self, z, h_ext, u_ext, g, boundary_ids):
        Z = z.function_space()
        n = firedrake.FacetNormal(Z.mesh())
        φ, v = firedrake.TestFunctions(Z)[:2]

        F_hx, F_ux = self.fluxes(h_ext, u_ext, g)

        h, u = firedrake.split(z)[:2]
        F_h, F_u = self.fluxes(h, u, g)

        return 0.5 * (
            inner(F_h + F_hx, φ * n) +
            inner(F_u + F_ux, outer(v, n))
        ) * ds(boundary_ids)

    def wave_speed(self, h, u, g, n):
        return abs(inner(u, n)) + sqrt(g * h)


def make_equation(g, b, form="momentum", **kwargs):
    r"""Return a function that calculates the weak form of the nonlinear
    shallow water equations

    Parameters
    ----------
    g : float, Constant or Function
        The acceleration due to gravity
    b : Function or expression
        The seafloor bathymetry
    form : str, optional
        Either "velocity" or "momentum"; which form of the problem to use
    thickness_in : expression, optional
        The thickness at the inflow boundary
    momentum_in : expression, optional
        The momentum at the inflow boundary
    velocity_in : expression, optional
        The velocity at the inflow boundary
    inflow_ids : tuple of int, optional
        The numeric IDS of the boundary segments where fluid is flowing in
    outflow_ids : tuple of int, optional
        The numeric IDs of the boundary segments where fluid is flowing out

    Returns
    -------
    equation
        A function that takes in an element of the mixed thickness-momentum
        or velocity function space and returns a Form object describing the
        weak form
    """
    outflow_ids = kwargs.get("outflow_ids", ())
    inflow_ids = kwargs.get("inflow_ids", ())
    h_in = kwargs.get("thickness_in", firedrake.Constant(0.0))

    if form == "momentum":
        problem_form = MomentumForm()
        u_in = kwargs.get("momentum_in", firedrake.Constant((0.0, 0.0)))
    else:
        problem_form = VelocityForm()
        u_in = kwargs.get("velocity_in", firedrake.Constant((0.0, 0.0)))

    def equation(z):
        Z = z.function_space()
        φ, v = firedrake.TestFunctions(Z)[:2]
        h, u = firedrake.split(z)[:2]
        F_h, F_u = problem_form.fluxes(h, u, g)

        mesh = Z.mesh()
        n = firedrake.FacetNormal(mesh)
        c = problem_form.wave_speed(h, u, g, n)

        sources = -inner(g * h * grad(b), v) * dx

        fluxes = (
            forms.cell_flux(F_h, φ) +
            forms.central_facet_flux(F_h, φ) +
            forms.lax_friedrichs_facet_flux(h, c, φ) +
            forms.cell_flux(F_u, v) +
            forms.central_facet_flux(F_u, v) +
            forms.lax_friedrichs_facet_flux(u, c, v)
        )

        boundary_ids = set(mesh.exterior_facets.unique_markers)
        wall_ids = tuple(boundary_ids - set(outflow_ids) - set(inflow_ids))
        u_wall = u - 2 * inner(u, n) * n
        boundary_fluxes = (
            problem_form.boundary_flux(z, h, u, g, outflow_ids) +
            problem_form.boundary_flux(z, h_in, u_in, g, inflow_ids) +
            problem_form.boundary_flux(z, h, u_wall, g, wall_ids) +
            forms.lax_friedrichs_boundary_flux(h, h, c, φ, outflow_ids) +
            forms.lax_friedrichs_boundary_flux(u, u, c, v, outflow_ids) +
            forms.lax_friedrichs_boundary_flux(h, h_in, c, φ, inflow_ids) +
            forms.lax_friedrichs_boundary_flux(u, u_in, c, v, inflow_ids) +
            forms.lax_friedrichs_boundary_flux(h, h, c, φ, wall_ids) +
            forms.lax_friedrichs_boundary_flux(u, u_wall, c, v, wall_ids)
        )

        return sources - fluxes - boundary_fluxes

    return equation
