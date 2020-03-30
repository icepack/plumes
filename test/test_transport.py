from numpy import pi as π
import sympy
import firedrake
from firedrake import sqrt, inner, grad, dx, Constant, as_vector
import plumes, plumes.coefficients

def norm(q):
    if len(q.ufl_shape) == 0:
        return firedrake.assemble(abs(q) * dx)
    return firedrake.assemble(sqrt(inner(q, q)) * dx)


def test_mass_transport():
    nx, ny = 64, 64
    mesh = firedrake.UnitSquareMesh(nx, ny)
    x = firedrake.SpatialCoordinate(mesh)
    y = as_vector((1/2, 1/2))

    Q = firedrake.FunctionSpace(mesh, family='DG', degree=1)
    V = firedrake.VectorFunctionSpace(mesh, family='DG', degree=1)

    v = x - y
    u = firedrake.interpolate(as_vector((-v[1], v[0])), V)

    z = as_vector((1/2, 3/4))
    r = Constant(1/8)
    D_expr = firedrake.max_value(1 - inner(x - z, x - z) / r**2, 0)
    D0 = firedrake.project(D_expr, Q)

    fields = {
        'thickness': D0,
        'velocity': u,
        'temperature': firedrake.Function(Q),
        'salinity': firedrake.Function(Q)
    }

    inflow = {
        'thickness_inflow': D0,
        'velocity_inflow': u,
        'temperature_inflow': None,
        'salinity_inflow': None
    }

    inputs = {
        'ice_shelf_base': Constant(0)
    }

    final_time = 2 * π
    num_steps = 3600
    timestep = final_time / num_steps
    dt = Constant(timestep)

    class MassTransportTestingModel(plumes.PlumeModel):
        def entrainment(self, **kwargs):
            return Constant(0., domain=mesh)

        def melt(self, **kwargs):
            return Constant(0., domain=mesh)

    model = MassTransportTestingModel()
    solver = plumes.PlumeSolver(model, **fields, **inflow, **inputs)
    for step in range(num_steps):
        solver.step(dt)

    D = solver.fields['thickness']
    assert norm(D - D0) / norm(D0) < 1.


def make_initial_plume_state(Lx):
    X = sympy.symbols('X')
    D_in, δD = .5, 30.
    u_in, δu = .01, .04
    D_sym = D_in + δD * X / Lx
    u_sym = u_in + δu * X / Lx

    return D_sym, u_sym


def make_parameters(friction):
    # To calculate the density contrast, we need to know the temperature and
    # salinity of high-salinity shelf water (HSSW) and ice shelf water (ISW).
    # See Lazeroms 2018 for the values used here.
    T_hssw, S_hssw = -1.91, 34.65
    T_isw, S_isw = -3.5, 0.
    β_T = 3.87e-5  # 1 / temperature
    β_S = 7.86e-4
    δρ = β_S * (S_hssw - S_isw) - β_T * (T_hssw - T_isw)
    E_0 = 0.036
    return {
        'friction': friction,
        'entrainment': E_0,
        'density_contrast': δρ
    }


def make_steady_plume_inputs(D, u, **kwargs):
    r"""Given the plume thickness and velocity in 1D, return the ice shelf
    slope and melt rate that make this thickness and velocity a steady state"""
    k = kwargs['friction']
    E_0 = kwargs['entrainment']
    δρ = kwargs['density_contrast']
    g = 9.81

    symbols = list(D.free_symbols)
    assert len(symbols) == 1
    x = symbols[0]
    assert list(u.free_symbols)[0] == x

    flux = (D * u**2).diff(x)
    friction = -k * u**2
    force = flux - friction
    slope = force / (D * δρ * g)

    entrainment = E_0 * u * slope
    melt = (D * u).diff(x) - entrainment

    return slope, melt


def test_momentum_transport_steady():
    pass
    Lx, Ly = 20e3, 20e3
    model = plumes.PlumeModel()

    # Create a synthetic plume thickness and velocity.
    D_sym, u_sym = make_initial_plume_state(Lx)

    # Compute the slope of the ice shelf draft and melt rate that make the
    # plume thickness and velocity we just defined in steady state.
    parameters = make_parameters(model.momentum_transport.friction)
    dZ_dX, m_sym = make_steady_plume_inputs(D_sym, u_sym, **parameters)

    # Integrate the slope to get the ice shelf draft.
    z_in = -100.
    X = list(D_sym.free_symbols)[0]
    z_sym = sympy.integrate(dZ_dX, (X, 0, X)) - z_in

    # Create a mesh and function spaces to represent the solution.
    nx, ny = 64, 64
    mesh = firedrake.RectangleMesh(nx, ny, Lx, Ly)
    P = firedrake.FunctionSpace(mesh, family='CG', degree=1)
    Q = firedrake.FunctionSpace(mesh, family='DG', degree=1)
    V = firedrake.VectorFunctionSpace(mesh, family='DG', degree=1)
    x = firedrake.SpatialCoordinate(mesh)

    # Convert the sympy expressions above into UFL expressions and project them
    # into finite element spaces.
    D0 = firedrake.project(sympy.lambdify(X, D_sym)(x[0]), Q)
    u_expr = as_vector((sympy.lambdify(X, u_sym)(x[0]), 0))
    u0 = firedrake.project(u_expr, V)

    fields = {
        'D': D0.copy(deepcopy=True),
        'u': u0.copy(deepcopy=True)
    }

    m = firedrake.project(sympy.lambdify(X, m_sym)(x[0]), Q)
    E_0 = parameters['entrainment']
    e = firedrake.project(sympy.lambdify(X, E_0 * u_sym * dZ_dX)(x[0]), Q)

    # The expression for the ice shelf draft includes a natural logarithm. To
    # convert a sympy expression involving the natural logarithm into a UFL
    # expression, we need to tell sympy to map all instances of this function
    # to the corresponding function from UFL.
    z_expr = sympy.lambdify(X, z_sym, modules={'log': firedrake.ln})(x[0])
    z_b = firedrake.project(z_expr, P)

    g = 9.81
    δρ = parameters['density_contrast']
    inputs = {
        'e': e,
        'm': m,
        'g': δρ * g * grad(z_b),
        'D_inflow': D0,
        'u_inflow': u0
    }

    final_time = 24 * 60 * 60
    num_steps = 2400
    dt = final_time / num_steps

    r"""
    for step in range(num_steps):
        model.solve(dt, **fields, **inputs)

    D = fields['D']
    assert norm(D - D0) / norm(D) < 1 / nx

    u = fields['u']
    assert norm(u - u0) / norm(u0) < 1 / nx
    """


def test_momentum_transport_unsteady():
    model = plumes.PlumeModel()
    Lx, Ly = 20e3, 20e3

    D_sym, u_sym = make_initial_plume_state(Lx)
    parameters = make_parameters(model.momentum_transport.friction)
    dZ_dX, m_sym = make_steady_plume_inputs(D_sym, u_sym, **parameters)
    z_in = -100.
    X = list(D_sym.free_symbols)[0]
    z_sym = sympy.integrate(dZ_dX, (X, 0, X)) - z_in

    nx, ny = 64, 64
    mesh = firedrake.RectangleMesh(nx, ny, Lx, Ly)
    P = firedrake.FunctionSpace(mesh, family='CG', degree=1)
    Q = firedrake.FunctionSpace(mesh, family='DG', degree=1)
    V = firedrake.VectorFunctionSpace(mesh, family='DG', degree=1)
    x = firedrake.SpatialCoordinate(mesh)

    D0 = firedrake.project(sympy.lambdify(X, D_sym)(x[0]), Q)
    u_expr = as_vector((sympy.lambdify(X, u_sym)(x[0]), 0))
    u0 = firedrake.project(u_expr, V)

    # The only difference between this unsteady test and the steady test is
    # that we use a slight perturbation to the initial plume thickness in the
    # middle of the domain and check that it gets propagated out.
    x0 = as_vector((3 * Lx / 4, Ly / 2))
    r = Constant(min(Lx, Ly) / 8)
    δD = Constant(30.)
    Dp = δD / 2 * firedrake.max_value(0, 1 - inner(x - x0, x - x0) / r**2)

    fields = {
        'D': firedrake.project(D0 + Dp, Q),
        'u': u0.copy(deepcopy=True)
    }

    m = firedrake.project(sympy.lambdify(X, m_sym)(x[0]), Q)
    E_0 = parameters['entrainment']
    e = firedrake.project(sympy.lambdify(X, E_0 * u_sym * dZ_dX)(x[0]), Q)

    z_expr = sympy.lambdify(X, z_sym, modules={'log': firedrake.ln})(x[0])
    z_b = firedrake.project(z_expr, P)

    g = 9.81
    δρ = parameters['density_contrast']
    inputs = {
        'e': e,
        'm': m,
        'g': δρ * g * grad(z_b),
        'D_inflow': D0,
        'u_inflow': u0
    }

    final_time = 3 * 24 * 60 * 60
    num_steps = 2400
    dt = final_time / num_steps

    r"""
    for step in range(num_steps):
        model.solve(dt, **fields, **inputs)

    D = fields['D']
    assert norm(D - D0) / norm(D) < 1 / nx

    u = fields['u']
    assert norm(u - u0) / norm(u0) < 1 / nx
    """
