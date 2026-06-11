using TensorMixedStates, .Qubits
using Printf

N = 6
B = 1.0
t = 10.

system = System(N, Qubit())

# Initial state: normalized x|up up ... up> + (1 - x)|down down ... down>
psi0 = State{Pure}(system, "Up")

# Uniform magnetic field in the x direction: H = B * sum_i Sx_i
H = sum(B * Sx(i) for i in 1:N)

# WII evolution in a single finite time step.
psi_t = approx_W(
    -im * H, t, psi0;
    nsweeps = 1,
    order = 1,
    w = 2,
    limits = Limits(cutoff = 1e-14, maxdim = 4),
)

mz = real.(expect1(psi_t, Z))
mz_exact = fill(cos(B * t), N)
print("<Z_i> = ", mz, "\n")
@printf("max_i |<Z_i> - exact| = %.3e\n", maximum(abs.(mz .- mz_exact)))