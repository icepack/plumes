import firedrake
from firedrake import as_vector, Constant, inner, interpolate, dx
import plumes
from plumes.numerics import ImplicitEuler

def test_linear_thin_film():
    Lx, Ly = 20.0, 20.0
    nx = 64
    ny = int(nx * Ly / Lx)
    mesh = firedrake.RectangleMesh(nx, ny, Lx, Ly, diagonal="crossed")

    degree = 1
    Q = firedrake.FunctionSpace(mesh, "CG", degree)
    V = firedrake.VectorFunctionSpace(mesh, "CG", degree)

    μ = Constant(1.0)
    ρ = Constant(1.0)
    g = Constant(1.0)

    h_0, δh = Constant(1.0), Constant(0.5)
    b_0, δb = Constant(0.0), Constant(0.25)

    u_0 = Constant(0.1)
    α = Constant(0.5)
    δu = α * Constant(Lx) / (4 * μ) * ρ * g * (h_0 - δh) * (1 + δb / δh)

    x, y = firedrake.SpatialCoordinate(mesh)
    H = h_0 - δh * x / Constant(Lx)
    B = b_0 - δb * x / Constant(Lx)
    S = B + H
    U = u_0 + δu * x / Constant(Lx)

    h = interpolate(H, Q)
    b = interpolate(B, Q)
    s = interpolate(S, Q)
    u = interpolate(as_vector((U, 0)), V)

    C = interpolate(((4 * H * μ * U.dx(0)).dx(0) - ρ * g * H * S.dx(0)) / U, Q)
    assert C.dat.data_ro[:].min() > 0
    a = interpolate((H * U).dx(0), Q)

    equation = plumes.models.thin_film.make_equation(
        density=ρ,
        gravity=g,
        bed=b,
        accumulation=a,
        viscosity=μ,
        friction=C,
        thickness_in=h.copy(deepcopy=True),
        velocity_in=u.copy(deepcopy=True),
        inflow_ids=(1,),
        outflow_ids=(2,),
    )

    def G(z):
        h, u = firedrake.split(z)
        return as_vector((h, 0.0, 0.0))

    params = {
        "solver_parameters": {
            "mat_type": "aij",
            "snes_type": "ksponly",
            "ksp_type": "gmres",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }
    }

    Z = Q * V
    z = firedrake.Function(Z)
    z.sub(0).assign(h)
    z.sub(1).assign(u)

    integrator = ImplicitEuler(equation, z, conserved_variables=G, **params)
    δx = mesh.cell_sizes.dat.data_ro[:].min()
    δt = δx / (float(u_0 + δu))
    final_time = 20.0
    num_steps = int(final_time / δt)

    for step in range(num_steps):
        integrator.step(δt)

    h = integrator.state.sub(0).copy(deepcopy=True)
    assert h.dat.data_ro[:].max() < 10.0
