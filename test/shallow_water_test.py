import pytest
import numpy as np
from scipy.optimize import root_scalar
import firedrake
from firedrake import (
    sqrt, assemble, Constant, as_vector, inner, max_value, dx
)
import plumes
from plumes.coefficients import gravity
from plumes.numerics import *

schemes = [ExplicitEuler, SSPRK34, RosenbrockMidpoint]
multipliers = {ExplicitEuler: 1/8, SSPRK34: 1/4, RosenbrockMidpoint: 1}
@pytest.mark.parametrize('scheme', schemes)
def test_linear_bed(scheme):
    start = 16
    finish = 2 * start
    incr = 4
    num_points = np.array(list(range(start, finish + incr, incr)))
    errors = np.zeros_like(num_points, dtype=np.float64)

    # TODO: Test with different degrees + function spaces
    degree = 1

    for k, nx in enumerate(num_points):
        Lx, Ly = 20.0, 20.0
        mesh = firedrake.RectangleMesh(nx, nx, Lx, Ly, diagonal='crossed')
        Q = firedrake.FunctionSpace(mesh, 'DG', degree)
        V = firedrake.VectorFunctionSpace(mesh, 'DG', degree)
        Z = Q * V

        x = firedrake.SpatialCoordinate(mesh)

        b_0 = Constant(0.2)
        δb = Constant(0.0)
        b = b_0 - δb * x[0] / Lx

        g = Constant(gravity)
        h_in = Constant(0.1)
        u_in = Constant(2.)
        q_in = h_in * u_in

        # Make the exact steady-state thickness.
        h = firedrake.project(h_in, Q)
        φ = firedrake.TestFunction(Q)
        F = (
            h + 0.5 * q_in**2 / (g * h**2) -
            (h_in + 0.5 * q_in**2 / (g * h_in**2)) -
            δb * x[0] / Lx
        ) * φ * dx
        firedrake.solve(F == 0, h)

        # The exact steady-state momentum is a constant.
        q = firedrake.project(as_vector((q_in, 0.0)), V)

        # Get the maximum wave speed and choose a timestep that will satisfy
        # the CFL condition.
        c = firedrake.project(q[0] / h + sqrt(g * h), Q)
        max_speed = c.dat.data_ro[:].max()
        min_diameter = mesh.cell_sizes.dat.data_ro[:].min()
        multiplier = multipliers[scheme]
        timestep = multiplier * min_diameter / max_speed / (2 * degree + 1)

        # Run the simulation for a full residence time of the system.
        final_time = 2 * Lx / float(u_in)
        num_steps = int(final_time / timestep)
        dt = final_time / num_steps

        z_0 = firedrake.Function(Z)
        h_0, q_0 = z_0.split()
        q_0.assign(q)

        # Add a small perturbation to the initial state.
        δh = Constant(h_in / 10)
        r = Constant(Lx / 8)
        y = Constant((Lx / 2, Ly / 2))
        h_0.project(h + δh * max_value(0, 1 - inner(x - y, x - y) / r**2))

        bcs = {
            'thickness_in': h_in,
            'momentum_in': as_vector((q_in, 0)),
            'inflow_ids': (1,),
            'outflow_ids': (2,)
        }
        equation = plumes.models.shallow_water.make_equation(g, b, **bcs)
        integrator = scheme(equation, z_0)

        for step in range(num_steps):
            integrator.step(dt)

        h_n, q_n = integrator.state.split()
        errors[k] = assemble(abs(h_n - h) * dx) / assemble(h * dx)

    slope, intercept = np.polyfit(np.log2(1 / num_points), np.log2(errors), 1)
    print(f'log(error) ~= {slope:5.3f} * log(dx) {intercept:+5.3f}')
    assert slope > degree - 0.05
