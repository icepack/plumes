r"""Model description for the 2D shallow water equations"""

import firedrake
from firedrake import sqrt, inner, outer, grad, dx, ds
from . import forms


class MomentumForm:
    def fluxes(self, h, q, g):
        r"""Calculate the flux of mass and momentum for the shallow water
        equations"""
        I = firedrake.Identity(2)
        F_h = q
        F_q = outer(q, q) / h + 0.5 * g * h**2 * I
        return F_h, F_q

    def boundary_flux(self, z, h_ext, q_ext, g, boundary_ids):
        Z = z.function_space()
        n = firedrake.FacetNormal(Z.mesh())
        φ, v = firedrake.TestFunctions(Z)

        F_hx, F_qx = self.fluxes(h_ext, q_ext, g)

        h, q = firedrake.split(z)
        F_h, F_q = self.fluxes(h, q, g)

        return 0.5 * (
            inner(F_hx, φ * n) +
            inner(F_qx, outer(v, n)) +
            inner(F_h, φ * n) +
            inner(F_q, outer(v, n))
        ) * ds(boundary_ids)

    def wall_flux(self, z, g, boundary_ids):
        n = firedrake.FacetNormal(z.ufl_domain())
        h, q = firedrake.split(z)
        # Mirror value of the fluid momentum
        q_ex = q - 2 * inner(q, n) * n
        return _boundary_flux(z, h, q_ex, g, boundary_ids)

    def wave_speed(self, h, q, g, n):
        return abs(inner(q / h, n)) + sqrt(g * h)


class VelocityForm:
    def fluxes(self, h, u, g):
        I = firedrake.Identity(2)
        F_h = h * u
        F_u = h * outer(u, u) + 0.5 * g * h ** 2 * I
        return F_h, F_u

    def boundary_flux(self, z, h_ext, u_ext, g, boundary_ids):
        Z = z.function_space()
        n = firedrake.FacetNormal(Z.mesh())
        φ, v = firedrake.TestFunctions(Z)

        F_hx, F_ux = self.fluxes(h_ext, u_ext, g)

        h, u = firedrake.split(z)
        F_h, F_u = self.fluxes(h, u, g)

        return 0.5 * (
            inner(F_hx, φ * n) +
            inner(F_ux, outer(v, n)) +
            inner(F_h, φ * n) +
            inner(F_u, outer(v, n))
        ) * ds(boundary_ids)

    def wall_flux(self, z, g, boundary_ids):
        n = firedrake.FacetNormal(z.ufl_domain())
        h, u = firedrake.split(z)
        # Mirror value of the fluid momentum
        u_ext = u - 2 * inner(u, n) * n
        return _boundary_flux(z, h, u_ext, g, boundary_ids)

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
        q_in = kwargs.get("momentum_in", firedrake.Constant((0.0, 0.0)))
    else:
        problem_form = VelocityForm()
        q_in = kwargs.get("velocity_in", firedrake.Constant((0.0, 0.0)))

    def equation(z):
        Z = z.function_space()
        φ, v = firedrake.TestFunctions(Z)
        h, q = firedrake.split(z)
        F_h, F_q = problem_form.fluxes(h, q, g)

        mesh = Z.mesh()
        n = firedrake.FacetNormal(mesh)
        c = problem_form.wave_speed(h, q, g, n)

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
        boundary_fluxes = (
            problem_form.boundary_flux(z, h, q, g, outflow_ids) +
            problem_form.boundary_flux(z, h_in, q_in, g, inflow_ids) +
            problem_form.boundary_flux(z, h, q_wall, g, wall_ids) +
            forms.lax_friedrichs_boundary_flux(h, h, c, φ, outflow_ids) +
            forms.lax_friedrichs_boundary_flux(q, q, c, v, outflow_ids) +
            forms.lax_friedrichs_boundary_flux(h, h_in, c, φ, inflow_ids) +
            forms.lax_friedrichs_boundary_flux(q, q_in, c, v, inflow_ids) +
            forms.lax_friedrichs_boundary_flux(h, h, c, φ, wall_ids) +
            forms.lax_friedrichs_boundary_flux(q, q_wall, c, v, wall_ids)
        )

        return sources - fluxes - boundary_fluxes

    return equation
