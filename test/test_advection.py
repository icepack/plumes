import numpy as np
from numpy import pi as π
import firedrake
from firedrake import assemble, Constant, as_vector, inner, max_value, dx
import plumes

def test_rotating_bump():
    start = 32
    finish = 2 * start
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

        z = Constant((1/3, 1/3))
        r = Constant(1/6)
        expr = max_value(0, 1 - inner(x - z, x - z) / r**2)

        final_time = 2 * π
        min_diameter = mesh.cell_sizes.dat.data_ro[:].min()
        max_speed = 1 / np.sqrt(2)
        # Choose a timestep that will satisfy the CFL condition
        timestep = (min_diameter / 8) / max_speed / (2 * degree + 1)
        num_steps = int(final_time / timestep)
        dt = final_time / num_steps

        q_0 = firedrake.project(expr, Q)
        equation = plumes.models.advection.make_equation(u, s)
        integrator = plumes.numerics.ExplicitEuler(equation, q_0, dt)

        for step in range(num_steps):
            integrator.step(dt)

        q = integrator.state
        errors[k] = assemble(abs(q - expr) * dx) / assemble(abs(expr) * dx)

    slope, intercept = np.polyfit(np.log2(1 / num_points), np.log2(errors), 1)
    print(f'log(error) ~= {slope:5.3f} * log(dx) {intercept:+5.3f}')
    assert slope > degree - 0.95


def test_inflow_boundary():
    start = 32
    finish = 2 * start
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
        integrator = plumes.numerics.ExplicitEuler(equation, q_0, dt)

        for step in range(num_steps):
            integrator.step(dt)

        q = integrator.state
        T = Constant(final_time)
        expr = q_in - firedrake.conditional(x[0] > U * T, x[0] - U * T, 0)
        errors[k] = assemble(abs(q - expr) * dx) / assemble(abs(expr) * dx)

    slope, intercept = np.polyfit(np.log2(1 / num_points), np.log2(errors), 1)
    print(f'log(error) ~= {slope:5.3f} * log(dx) {intercept:+5.3f}')
    assert slope > degree - 0.1
