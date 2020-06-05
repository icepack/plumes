import pytest
import numpy as np
import sympy
import firedrake
from firedrake import sqrt, inner, grad, dx, Constant, as_vector
import plumes, plumes.coefficients


# For several of the synthetic test cases we use some realistic values for the
# temperature and salinity of high-salinity shelf water (HSSW) and ice-shelf
# water (ISW). See Lazeroms 2018 for the values used here.
T_hssw, S_hssw = -1.91, 34.65
T_isw, S_isw = -3.5, 0.


def norm(q):
    if len(q.ufl_shape) == 0:
        return firedrake.assemble(abs(q) * dx)
    return firedrake.assemble(sqrt(inner(q, q)) * dx)


def make_initial_plume_state(Lx, D_in, δD, u_in, δu):
    X = sympy.symbols('X')
    D_sym = D_in + δD * X / Lx
    u_sym = u_in + δu * X / Lx

    return D_sym, u_sym


def make_steady_plume_inputs(D, u):
    r"""Given the plume thickness and velocity in 1D, return the ice shelf
    slope and melt rate that make this thickness and velocity a steady state"""
    k = plumes.coefficients.friction
    E_0 = plumes.coefficients.entrainment

    β_T = plumes.coefficients.thermal_expansion
    β_S = plumes.coefficients.haline_contraction
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

    entrainment = E_0 * u * slope
    melt = (D * u).diff(x) - entrainment

    return {
        'slope': slope,
        'entrainment': entrainment,
        'melt': melt
    }


def momentum_transport_run(steady, nx, ny):
    Lx, Ly = 20e3, 20e3

    # Create a synthetic plume thickness and velocity.
    D_in, δD = .5, 30.
    u_in, δu = .01, .04
    D_sym, u_sym = make_initial_plume_state(Lx, D_in, δD, u_in, δu)

    # Compute the slope of the ice shelf draft and melt rate that make the
    # plume thickness and velocity we just defined in steady state.
    plume_inputs = make_steady_plume_inputs(D_sym, u_sym)

    # Integrate the slope to get the ice shelf draft.
    dZ_dX = plume_inputs['slope']
    z_in = -100.
    X = list(D_sym.free_symbols)[0]
    z_sym = sympy.integrate(dZ_dX, (X, 0, X)) - z_in

    # Create a mesh and function spaces to represent the solution.
    mesh = firedrake.RectangleMesh(nx, ny, Lx, Ly, quadrilateral=True)
    P = firedrake.FunctionSpace(mesh, family='CG', degree=2)
    Q = firedrake.FunctionSpace(mesh, family='DQ', degree=1)
    V = firedrake.VectorFunctionSpace(mesh, family='DQ', degree=1)
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

    class MomentumTransportTestingModel(plumes.PlumeModel):
        def entrainment(self, **kwargs):
            return e

        def melt(self, **kwargs):
            return m

        def density_contrast(self, **kwargs):
            β_T = plumes.coefficients.thermal_expansion
            β_S = plumes.coefficients.haline_contraction
            return β_S * (S_hssw - S_isw) - β_T * (T_hssw - T_isw)

    if steady:
        δD = Constant(0.)
    else:
        δD = Constant(1.)

    # The only difference between this unsteady test and the steady test is
    # that we use a slight perturbation to the initial plume thickness in
    # the middle of the domain and check that it gets propagated out.
    x0 = as_vector((3 * Lx / 4, Ly / 2))
    r = Constant(min(Lx, Ly) / 8)
    perturbation = δD * firedrake.max_value(0, 1 - inner(x - x0, x - x0) / r**2)

    fields = {
        'thickness': firedrake.project(D0 + perturbation, Q),
        'velocity': u0.copy(deepcopy=True),
        'temperature': firedrake.Function(Q),
        'salinity': firedrake.Function(Q)
    }

    inflow = {
        'thickness_inflow': D0,
        'velocity_inflow': u0,
        'temperature_inflow': firedrake.Function(Q),
        'salinity_inflow': firedrake.Function(Q)
    }

    inputs = {
        'ice_shelf_base': z_b,
        'salinity_ambient': S_hssw,
        'temperature_ambient': T_hssw
    }

    final_time = 2 * 24 * 60 * 60
    timestep = Lx / nx / (u_in + δu) / 48
    num_steps = int(final_time / timestep)
    dt = final_time / num_steps

    model = MomentumTransportTestingModel()
    solver = plumes.PlumeSolver(model, **fields, **inflow, **inputs)
    for step in range(num_steps):
        solver.step(dt)

    D = solver.fields['thickness']
    u = solver.fields['velocity']
    return norm(D - D0) / norm(D0), norm(u - u0) / norm(u0)


@pytest.mark.parametrize('steady', [True, False])
def test_momentum_transport(steady):
    Ns = np.array([32, 48, 64, 72, 84, 96])
    D_errors = np.zeros(len(Ns))
    u_errors = np.zeros(len(Ns))
    for k, N in enumerate(Ns):
        D_error, u_error = momentum_transport_run(steady, N, N)
        D_errors[k] = D_error
        u_errors[k] = u_error

    D_slope, D_intercept = np.polyfit(np.log10(1/Ns), np.log10(D_errors), 1)
    u_slope, u_intercept = np.polyfit(np.log10(1/Ns), np.log10(u_errors), 1)

    print('Thickness/velocity convergence rate: {}, {}'.format(D_slope, u_slope))

    assert D_slope > 0.5
    assert u_slope > 0.5


def material_transport_run(component, nx, ny):
    Lx, Ly = 20e3, 20e3

    D_in, δD = .5, 30.
    u_in, δu = .01, .04
    D_sym, u_sym = make_initial_plume_state(Lx, D_in, δD, u_in, δu)

    plume_inputs = make_steady_plume_inputs(D_sym, u_sym)

    dZ_dX = plume_inputs['slope']
    z_in = -100.
    X = list(D_sym.free_symbols)[0]
    z_sym = sympy.integrate(dZ_dX, (X, 0, X)) - z_in

    mesh = firedrake.RectangleMesh(nx, ny, Lx, Ly, quadrilateral=True)
    P = firedrake.FunctionSpace(mesh, family='CG', degree=2)
    Q = firedrake.FunctionSpace(mesh, family='DQ', degree=1)
    V = firedrake.VectorFunctionSpace(mesh, family='DQ', degree=1)
    x = firedrake.SpatialCoordinate(mesh)

    D0 = firedrake.project(sympy.lambdify(X, D_sym)(x[0]), Q)
    S0 = firedrake.project(firedrake.Constant(S_isw), Q)
    T0 = firedrake.project(firedrake.Constant(T_isw), Q)
    u0 = firedrake.project(as_vector((sympy.lambdify(X, u_sym)(x[0]), 0)), V)

    m_sym = plume_inputs['melt']
    m = firedrake.project(sympy.lambdify(X, m_sym)(x[0]), Q)
    E_0 = plumes.coefficients.entrainment
    e_sym = plume_inputs['entrainment']
    e = firedrake.project(sympy.lambdify(X, e_sym)(x[0]), Q)

    z_expr = sympy.lambdify(X, z_sym, modules={'log': firedrake.ln})(x[0])
    z_b = firedrake.project(z_expr, P)

    fields = {
        'thickness': D0.copy(deepcopy=True),
        'velocity': u0.copy(deepcopy=True),
        'temperature': T0.copy(deepcopy=True),
        'salinity': S0.copy(deepcopy=True)
    }

    inflow = {
        'thickness_inflow': D0,
        'velocity_inflow': u0,
        'temperature_inflow': T0,
        'salinity_inflow': S0
    }

    inputs = {
        'ice_shelf_base': z_b,
        'salinity_ambient': S_hssw,
        'temperature_ambient': T_hssw
    }

    final_time = 2 * 24 * 60 * 60
    timestep = Lx / nx / (u_in + δu) / 48
    num_steps = int(final_time / timestep)
    dt = final_time / num_steps

    model = plumes.PlumeModel()
    components = plumes.Component.Mass | component
    solver = plumes.PlumeSolver(model, components, **fields, **inflow, **inputs)
    name = {plumes.Component.Salt: 'salinity', plumes.Component.Heat: 'temperature'}
    for step in range(num_steps):
        if step % 50 == 0:
            q = solver.fields[name[component]]
            print('    {} {}'.format(q.dat.data_ro.min(), q.dat.data_ro.max()))
        solver.step(dt)

    q = solver.fields[name[component]]
    return q.dat.data_ro.min(), q.dat.data_ro.max()


def test_salt_transport():
    Ns = np.array([32, 48, 64, 72, 84, 96])
    for n in Ns:
        s_min, s_max = material_transport_run(plumes.Component.Salt, n, n)
        print(s_min, s_max)


def test_heat_transport():
    Ns = np.array([32, 48, 64, 72, 84, 96])
    for n in Ns:
        T_min, T_max = material_transport_run(plumes.Component.Heat, n, n)
        print(T_min, T_max)
