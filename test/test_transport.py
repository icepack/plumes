from numpy import pi as π
import firedrake
from firedrake import inner, dx
import plumes

def norm(q):
    return firedrake.assemble(abs(q) * dx)

def test_mass_transport():
    nx, ny = 64, 64
    mesh = firedrake.UnitSquareMesh(nx, ny)
    x = firedrake.SpatialCoordinate(mesh)
    y = firedrake.as_vector((1/2, 1/2))

    v = x - y
    u = firedrake.as_vector((-v[1], v[0]))

    Q = firedrake.FunctionSpace(mesh, family='DG', degree=1)
    final_time = 2 * π
    num_steps = 3600
    timestep = final_time / num_steps
    dt = firedrake.Constant(timestep)

    z = firedrake.as_vector((1/2, 3/4))
    r = firedrake.Constant(1/8)
    D_expr = firedrake.max_value(1 - inner(x - z, x - z) / r**2, 0)
    D0 = firedrake.project(D_expr, Q)
    D = D0.copy(deepcopy=True)

    e = firedrake.Constant(0)
    m = firedrake.Constant(0)

    model = plumes.PlumeModel()
    for step in range(num_steps):
        model.mass_transport_solve(dt, D=D, u=u, e=e, m=m, D_inflow=D0)

    assert norm(D - D0) / norm(D0) < 1.
