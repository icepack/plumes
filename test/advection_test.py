import pytest
import numpy as np
from numpy import pi as π
import firedrake
from firedrake import (
    assemble, exp, Constant, as_vector, as_tensor, inner, max_value, dx
)
import plumes
from plumes import numerics

@pytest.mark.parametrize(
    'scheme',
    [numerics.ExplicitEuler, numerics.SSPRK33, numerics.SSPRK34]
)
def test_rotating_bump(scheme):
    start = 16
    finish = 3 * start
    incr = 4
    num_points = np.array(list(range(start, finish + incr, incr)))
    errors = np.zeros_like(num_points, dtype=np.float64)

    for k, nx in enumerate(num_points):
        mesh = firedrake.PeriodicUnitSquareMesh(nx, nx, diagonal='crossed')
        degree = 1
        Q = firedrake.FunctionSpace(mesh, family='DG', degree=degree)

        # The velocity field is uniform solid-body rotation about the
        # center of a square
        x = firedrake.SpatialCoordinate(mesh)
        y = Constant((0.5, 0.5))
        w = x - y
        u = as_vector((-w[1], +w[0]))

        # There are no sources
        s = Constant(0.0)

        origin = Constant((1/2, 1/2))
        delta = Constant((1/6, 1/6))
        z_0 = Constant(origin - delta)
        r = Constant(1/6)
        expr_0 = max_value(0, 1 - inner(x - z_0, x - z_0) / r**2)

        θ = 4 * π / 3
        min_diameter = mesh.cell_sizes.dat.data_ro[:].min()
        max_speed = 1 / np.sqrt(2)
        # Choose a timestep that will satisfy the CFL condition
        timestep = (min_diameter / 8) / max_speed / (2 * degree + 1)
        num_steps = int(θ / timestep)
        dt = θ / num_steps

        q_0 = firedrake.project(expr_0, Q)
        equation = plumes.models.advection.make_equation(u, s)
        parameters = {
            'solver_parameters': {
                'snes_type': 'ksponly',
                'ksp_type': 'preonly',
                'pc_type': 'ilu',
                'sub_pc_type': 'bjacobi'
            }
        }
        integrator = scheme(equation, q_0, **parameters)

        for step in range(num_steps):
            integrator.step(dt)

        # Compute the exact solution
        R = as_tensor([[np.cos(θ), -np.sin(θ)], [np.sin(θ), np.cos(θ)]])
        z_1 = firedrake.dot(R, z_0 - origin) + origin
        expr_1 = max_value(0, 1 - inner(x - z_1, x - z_1) / r**2)

        q = integrator.state
        errors[k] = assemble(abs(q - expr_1) * dx) / assemble(abs(expr_1) * dx)

    slope, intercept = np.polyfit(np.log2(1 / num_points), np.log2(errors), 1)
    print(f'log(error) ~= {slope:5.3f} * log(dx) {intercept:+5.3f}')
    assert slope > degree - 0.95


@pytest.mark.parametrize(
    'scheme',
    [numerics.ExplicitEuler, numerics.SSPRK33, numerics.SSPRK34]
)
def test_inflow_boundary(scheme):
    start = 16
    finish = 3 * start
    incr = 4
    num_points = np.array(list(range(start, finish + incr, incr)))
    errors = np.zeros_like(num_points, dtype=np.float64)

    for k, nx in enumerate(num_points):
        mesh = firedrake.UnitSquareMesh(nx, nx, diagonal='crossed')
        degree = 1
        Q = firedrake.FunctionSpace(mesh, family='DG', degree=1)

        x = firedrake.SpatialCoordinate(mesh)
        U = Constant(1.)
        u = as_vector((U, 0.))

        q_in = Constant(1.)
        s = Constant(0.)

        final_time = 0.5
        min_diameter = mesh.cell_sizes.dat.data_ro[:].min()
        max_speed = 1.
        timestep = (min_diameter / 16) / max_speed / (2 * degree + 1)
        num_steps = int(final_time / timestep)
        dt = final_time / num_steps

        q_0 = firedrake.project(q_in - x[0], Q)
        equation = plumes.models.advection.make_equation(u, s, q_in)
        integrator = scheme(equation, q_0)

        for step in range(num_steps):
            integrator.step(dt)

        q = integrator.state
        T = Constant(final_time)
        expr = q_in - firedrake.conditional(x[0] > U * T, x[0] - U * T, 0)
        errors[k] = assemble(abs(q - expr) * dx) / assemble(abs(expr) * dx)

    slope, intercept = np.polyfit(np.log2(1 / num_points), np.log2(errors), 1)
    print(f'log(error) ~= {slope:5.3f} * log(dx) {intercept:+5.3f}')
    assert slope > degree - 0.1


def test_imex():
    start = 16
    finish = 3 * start
    incr = 4

    num_points = np.array(list(range(start, finish + incr, incr)))
    errors = np.zeros_like(num_points, dtype=np.float64)

    for k, nx in enumerate(num_points):
        mesh = firedrake.PeriodicUnitSquareMesh(nx, nx, diagonal='crossed')
        degree = 1
        Q = firedrake.FunctionSpace(mesh, family='DG', degree=degree)

        # The velocity field is uniform solid-body rotation about the
        # center of a square
        x = firedrake.SpatialCoordinate(mesh)
        y = Constant((0.5, 0.5))
        w = x - y
        u = as_vector((-w[1], +w[0]))

        # There are no sources
        s = Constant(0.0)

        origin = Constant((1/2, 1/2))
        delta = Constant((1/6, 1/6))
        z_0 = Constant(origin - delta)
        r = Constant(1/6)
        expr_0 = max_value(0, 1 - inner(x - z_0, x - z_0) / r**2)

        θ = 4 * π / 3
        min_diameter = mesh.cell_sizes.dat.data_ro[:].min()
        max_speed = 1 / np.sqrt(2)
        # Choose a timestep that will satisfy the CFL condition
        timestep = (min_diameter / 8) / max_speed / (2 * degree + 1)
        num_steps = int(θ / timestep)
        dt = θ / num_steps

        q_0 = firedrake.project(expr_0, Q)
        advection_equation = plumes.models.advection.make_equation(u, s)

        λ = Constant(0.1)
        def decay_equation(q):
            φ = firedrake.TestFunction(q.function_space())
            return -λ * q * φ * dx

        integrator = numerics.IMEX(advection_equation, decay_equation, q_0)

        for step in range(num_steps):
            integrator.step(dt)

        # Compute the exact solution
        R = as_tensor([[np.cos(θ), -np.sin(θ)], [np.sin(θ), np.cos(θ)]])
        z_1 = firedrake.dot(R, z_0 - origin) + origin
        expr_1 = exp(-λ * θ) * max_value(0, 1 - inner(x - z_1, x - z_1) / r**2)

        q = integrator.state
        errors[k] = assemble(abs(q - expr_1) * dx) / assemble(abs(expr_1) * dx)

    slope, intercept = np.polyfit(np.log2(1 / num_points), np.log2(errors), 1)
    print(f'log(error) ~= {slope:5.3f} * log(dx) {intercept:+5.3f}')
    assert slope > degree - 0.95
