r"""Model description for the 2D shallow water equations"""

import firedrake
from firedrake import sqrt, inner, outer, grad, dx
from ..numerics import forms

def make_equation(g, b):
    def equation(z):
        Z = z.function_space()
        φ, v = firedrake.TestFunctions(Z)
        h, q = firedrake.split(z)

        F_h = q

        I = firedrake.Identity(2)
        F_q = outer(q, q) / h + 0.5 * g * h**2 * I

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

        return sources - fluxes

    return equation
