r"""Model description for the advection equation"""

import firedrake
from firedrake import max_value, min_value, inner, dx, ds
from ..numerics import forms

def make_equation(u, s, q_in=firedrake.Constant(0)):
    def equation(q):
        Q = q.function_space()
        φ = firedrake.TestFunction(Q)

        F = q * u

        mesh = Q.mesh()
        n = firedrake.FacetNormal(mesh)
        c = abs(inner(u, n))

        sources = s * φ * dx
        fluxes = (
            forms.cell_flux(F, φ) +
            forms.central_facet_flux(F, φ) +
            forms.lax_friedrichs_facet_flux(q, c, φ) +
            q * max_value(0, inner(u, n)) * φ * ds +
            q_in * min_value(0, inner(u, n)) * φ * ds
        )

        return sources - fluxes

    return equation
