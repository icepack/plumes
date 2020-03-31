from numpy import pi as π
import sympy
import firedrake
from firedrake import sqrt, inner, grad, dx, Constant, as_vector
import plumes, plumes.coefficients

def norm(q):
    if len(q.ufl_shape) == 0:
        return firedrake.assemble(abs(q) * dx)
    return firedrake.assemble(sqrt(inner(q, q)) * dx)


def make_initial_plume_state(Lx):
    X = sympy.symbols('X')
    D_in, δD = .5, 30.
    u_in, δu = .01, .04
    D_sym = D_in + δD * X / Lx
    u_sym = u_in + δu * X / Lx

    return D_sym, u_sym


def make_steady_plume_inputs(D, u):
    r"""Given the plume thickness and velocity in 1D, return the ice shelf
    slope and melt rate that make this thickness and velocity a steady state"""
    k = plumes.coefficients.friction
    E_0 = plumes.coefficients.entrainment

    # To calculate the density contrast, we need to know the temperature and
    # salinity of high-salinity shelf water (HSSW) and ice shelf water (ISW).
    # See Lazeroms 2018 for the values used here.
    T_hssw, S_hssw = -1.91, 34.65
    T_isw, S_isw = -3.5, 0.
    β_T = 3.87e-5  # 1 / temperature
    β_S = 7.86e-4
    δρ = β_S * (S_hssw - S_isw) - β_T * (T_hssw - T_isw)
    g = 9.81

    symbols = list(D.free_symbols)
    assert len(symbols) == 1
    x = symbols[0]
    assert list(u.free_symbols)[0] == x

    flux = (D * u**2).diff(x)
    friction = -k * u**2
    force = flux - friction
    slope = force / (D * δρ * g)
    gravity = δρ * g * slope

    entrainment = E_0 * u * slope
    melt = (D * u).diff(x) - entrainment

    return {
        'slope': slope,
        'gravity': gravity,
        'entrainment': entrainment,
        'melt': melt
    }


def test_momentum_transport_steady():
    Lx, Ly = 20e3, 20e3

    # Create a synthetic plume thickness and velocity.
    D_sym, u_sym = make_initial_plume_state(Lx)

    # Compute the slope of the ice shelf draft and melt rate that make the
    # plume thickness and velocity we just defined in steady state.
    plume_inputs = make_steady_plume_inputs(D_sym, u_sym)

    # Integrate the slope to get the ice shelf draft.
    dZ_dX = plume_inputs['slope']
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
    u0 = firedrake.project(as_vector((sympy.lambdify(X, u_sym)(x[0]), 0)), V)

    m_sym = plume_inputs['melt']
    m = firedrake.project(sympy.lambdify(X, m_sym)(x[0]), Q)
    E_0 = plumes.coefficients.entrainment
    e_sym = plume_inputs['entrainment']
    e = firedrake.project(sympy.lambdify(X, e_sym)(x[0]), Q)

    # The expression for the ice shelf draft includes a natural logarithm. To
    # convert a sympy expression involving the natural logarithm into a UFL
    # expression, we need to tell sympy to map all instances of this function
    # to the corresponding function from UFL.
    z_expr = sympy.lambdify(X, z_sym, modules={'log': firedrake.ln})(x[0])
    z_b = firedrake.project(z_expr, P)

    g_sym = plume_inputs['gravity']
    g_expr = sympy.lambdify(X, g_sym)(x[0])
    g = firedrake.project(as_vector((g_expr, 0)), V)

    class SteadyMomentumTransportTestingModel(plumes.PlumeModel):
        def entrainment(self, **kwargs):
            return e

        def melt(self, **kwargs):
            return m

        def gravity(self, **kwargs):
            return g

    fields = {
        'thickness': D0,
        'velocity': u0,
        'temperature': firedrake.Function(Q),
        'salinity': firedrake.Function(Q)
    }

    inflow = {
        'thickness_inflow': D0,
        'velocity_inflow': u0,
        'temperature_inflow': firedrake.Function(Q),
        'salinity_inflow': firedrake.Function(Q)
    }

    inputs = {'ice_shelf_base': z_b}

    final_time = 24 * 60 * 60
    num_steps = 2400
    dt = final_time / num_steps

    model = SteadyMomentumTransportTestingModel()
    solver = plumes.PlumeSolver(model, **fields, **inflow, **inputs)
    for step in range(num_steps):
        solver.step(dt)

    D = solver.fields['thickness']
    assert norm(D - D0) / norm(D) < 1 / nx

    u = solver.fields['velocity']
    assert norm(u - u0) / norm(u0) < 1 / nx


def test_momentum_transport_unsteady():
    Lx, Ly = 20e3, 20e3

    # Create a synthetic plume thickness and velocity.
    D_sym, u_sym = make_initial_plume_state(Lx)

    # Compute the slope of the ice shelf draft and melt rate that make the
    # plume thickness and velocity we just defined in steady state.
    plume_inputs = make_steady_plume_inputs(D_sym, u_sym)

    # Integrate the slope to get the ice shelf draft.
    dZ_dX = plume_inputs['slope']
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
    u0 = firedrake.project(as_vector((sympy.lambdify(X, u_sym)(x[0]), 0)), V)

    m_sym = plume_inputs['melt']
    m = firedrake.project(sympy.lambdify(X, m_sym)(x[0]), Q)
    E_0 = plumes.coefficients.entrainment
    e_sym = plume_inputs['entrainment']
    e = firedrake.project(sympy.lambdify(X, e_sym)(x[0]), Q)

    # The expression for the ice shelf draft includes a natural logarithm. To
    # convert a sympy expression involving the natural logarithm into a UFL
    # expression, we need to tell sympy to map all instances of this function
    # to the corresponding function from UFL.
    z_expr = sympy.lambdify(X, z_sym, modules={'log': firedrake.ln})(x[0])
    z_b = firedrake.project(z_expr, P)

    g_sym = plume_inputs['gravity']
    g_expr = sympy.lambdify(X, g_sym)(x[0])
    g = firedrake.project(as_vector((g_expr, 0)), V)

    class SteadyMomentumTransportTestingModel(plumes.PlumeModel):
        def entrainment(self, **kwargs):
            return e

        def melt(self, **kwargs):
            return m

        def gravity(self, **kwargs):
            return g

    # The only difference between this unsteady test and the steady test is
    # that we use a slight perturbation to the initial plume thickness in the
    # middle of the domain and check that it gets propagated out.
    x0 = as_vector((3 * Lx / 4, Ly / 2))
    r = Constant(min(Lx, Ly) / 8)
    δD = Constant(15.)
    perturbation = δD * firedrake.max_value(0, 1 - inner(x - x0, x - x0) / r**2)

    fields = {
        'thickness': firedrake.project(D0 + perturbation, Q),
        'velocity': u0,
        'temperature': firedrake.Function(Q),
        'salinity': firedrake.Function(Q)
    }

    inflow = {
        'thickness_inflow': D0,
        'velocity_inflow': u0,
        'temperature_inflow': firedrake.Function(Q),
        'salinity_inflow': firedrake.Function(Q)
    }

    inputs = {'ice_shelf_base': z_b}

    final_time = 24 * 60 * 60
    num_steps = 2400
    dt = final_time / num_steps

    model = SteadyMomentumTransportTestingModel()
    solver = plumes.PlumeSolver(model, **fields, **inflow, **inputs)
    for step in range(num_steps):
        solver.step(dt)

    D = solver.fields['thickness']
    assert norm(D - D0) / norm(D) < 1 / nx

    u = solver.fields['velocity']
    assert norm(u - u0) / norm(u0) < 1 / nx
