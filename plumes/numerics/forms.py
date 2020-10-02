r"""Functions to calculate the weak forms of hyperbolic PDE"""

from firedrake import inner, outer, avg, grad, dx, dS, FacetNormal

def cell_flux(F, v):
    r"""Create the weak form of the fluxes through the cell interior

    Parameters
    ----------
    F : ufl.Expr
        A symbolic expression for the flux
    v : firedrake.TestFunction
        A test function from the state space

    Returns
    -------
    f : firedrake.Form
        A 1-form that discretizes the residual of the flux
    """
    return -inner(F, grad(v)) * dx


def central_facet_flux(F, v):
    r"""Create the weak form of the central numerical flux through cell facets

    This numerical flux, by itself, is unstable. The full right-hand side of
    the problem requires an additional facet flux term for stability.

    Parameters
    ----------
    F : ufl.Expr
        A symbolic expression for the flux
    v : firedrake.TestFunction
        A test function from the state space

    Returns
    -------
    f : firedrake.Form
        A 1-form that discretizes the residual of the flux
    """
    mesh = v.ufl_domain()
    n = FacetNormal(mesh)
    return inner(avg(F), outer(v('+'), n('+')) + outer(v('-'), n('-'))) * dS


def central_inflow_flux(F_in, v, boundary_ids):
    r"""Create the weak form of the central numerical flux through the domain
    boundary"""
    mesh = v.ufl_domain()
    n = FacetNormal(mesh)
    return inner(F_in, outer(v, n)) * ds(boundary_ids)


def lax_friedrichs_facet_flux(s, c, v):
    r"""Create the Lax-Friedrichs numerical flux through cell facets

    This is a diffusive numerical flux that can correct for the instability of
    the central facet flux. You can think of it as being like upwinding but
    for systems of PDE.

    Parameters
    ----------
    s : firedrake.Function
        The state variable being solved for
    c : ufl.Expr
        An expression for the maximum outward wave speed through a facet

    Returns
    -------
    f : firedrake.Form
        A 1-form that discretizes the residual of the flux
    """
    return avg(c) * inner(s('+') - s('-'), v('+') - v('-')) * dS
