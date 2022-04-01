import numpy as np
from numpy.linalg import norm
import firedrake
from firedrake import Constant, inner, dot, sym, tr, grad, dx, ds, max_value, min_value
from . import forms


def conserved_variables(z):
    h, u = firedrake.split(z)
    return firedrake.as_vector((h, 0.0, 0.0))


def ε(u):
    return sym(grad(u))


def minimum_angle(mesh):
    coords = mesh.coordinates.dat.data_ro
    cells = mesh.coordinates.cell_node_map().values

    θ = np.inf
    for cell in cells:
        for k in range(3):
            x, y, z = coords[np.roll(cell, k)]
            ζ, ξ = y - x, z - x
            angle = np.arccos(np.inner(ζ, ξ) / (norm(ζ) * norm(ξ)))
            θ = min(angle, θ)

    return θ


def make_equation(**kwargs):
    """Return a function that calculates the weak form of the viscous thin-film
    equation

    Parameters
    ----------
    gravity : float
        The acceleration due to gravity
    bed : Function or expression
        The bed topography
    accumulation : Function or Constant
        The accumulation rate
    viscosity : Function or Constant
        The viscosity of the fluid
    friction : Function or Constant
        The bed friction coefficient
    friction_exponent : int
        The nonlinearity in the friction law
    thickness_in : expression, optional
        Thickness at the inflow boundary
    velocity_in : expression, optional
        Velocity at the inflow boundary
    inflow_ids : tuple of int, optional
        The numeric IDs of the boundary segments where fluid is flowing in
    outflow_ids : tuple of int, optional
        The numeric IDs of the boundary segments where fluid is flowing out

    Returns
    -------
    equation
        A function that takes in an element of the mixed thickness-momentum or
        velocity function space and returns a Form object describing the weak form
    """
    keywords = ("gravity", "bed", "accumulation", "viscosity", "density", "friction")
    g, b, a, μ, ρ, C = map(kwargs.__getitem__, keywords)

    m = kwargs.get("friction_exponent", 0)
    inflow_ids = kwargs.get("inflow_ids", ())
    outflow_ids = kwargs.get("outflow_ids", ())
    h_in = kwargs.get("thickness_in", firedrake.Constant(0.0))
    u_in = kwargs.get("velocity_in", firedrake.Constant((0.0, 0.0)))

    def equation(z):
        Z = z.function_space()
        mesh = Z.mesh()
        φ, v = firedrake.TestFunctions(Z)[:2]
        h, u = firedrake.split(z)[:2]

        mass_sources = a * φ * dx
        cell_flux = -h * inner(u, grad(φ)) * dx
        n = firedrake.FacetNormal(mesh)
        boundary_flux_out = h * max_value(0, inner(u, n)) * φ * ds
        boundary_flux_in = h_in * min_value(0, inner(u, n)) * φ * ds
        equation_h = mass_sources - (cell_flux + boundary_flux_out + boundary_flux_in)

        viscosity = h * μ * (inner(ε(u), ε(v)) + tr(ε(u)) * tr(ε(v))) * dx
        friction = C * inner(u, u)**(m / 2) * inner(u, v) * dx
        s = b + h
        gravity = -ρ * g * h * inner(grad(s), v) * dx

        # Use Nitsche's method for Dirichlet BCs
        I = firedrake.Identity(2)
        boundary_flux = μ * h * (
            inner(dot(ε(v) + tr(ε(v)) * I, n), u - u_in)
            + inner(dot(ε(u) + tr(ε(u)) * I, n), v)
        ) * ds(inflow_ids)

        p = Z.sub(1).ufl_element().degree()
        θ = minimum_angle(mesh)
        α = Constant(4 * p * (p + 1) / (np.sin(θ) * np.tan(θ / 2)))
        λ = firedrake.CellDiameter(mesh)
        boundary_penalty = α * h * μ / λ * inner(u - u_in, v) * ds(inflow_ids)

        equation_u = (
            viscosity
            + friction
            - gravity
            + boundary_flux
            + boundary_penalty
        )

        return equation_h + equation_u

    return equation
