using TensorMixedStates, .Qubits 
using ITensorMPS: linkdims

function lindbladian_xx(n, mu, γ,Γ)
    # XX spin chain Hamiltonian
    h = sum(X(i) * X(i + 1) + Y(i) * Y(i + 1) for i in 1:(n - 1))
    # Boundary driving / coupling to baths at the edges 
    baths =
        Dissipator(sqrt((1 - mu)*Γ*2) * Sp)(1) +
        Dissipator(sqrt((1 + mu)*Γ*2) * Sm)(1) +
        Dissipator(sqrt((1 + mu)*Γ*2) * Sp)(n) +
        Dissipator(sqrt((1 - mu)*Γ*2) * Sm)(n)

    # Dephasing in the bulk, with rate γ
    dephasing = sum(Dissipator(sqrt(γ) * Z)(i) for i in 1:n)
    return Evolver(-im * h) + baths + dephasing
    
end

# Exact NESS magnetization profile for the XX chain with boundary driving and bulk dephasing:
# M. Žnidarič J. Stat. Mech. (2010) L05002 (https://arxiv.org/abs/1005.1271)
function sigma_z_steady(i::Integer, n::Integer, γ::Real, Γ::Real, mu::Real)
    b = -mu / (Γ + 1/Γ + (n - 1) * γ)
    if i == 1
        return -b / Γ - mu
    elseif i == n
        return -b * (1/Γ + 2Γ + 2(n - 1) * γ) - mu
    else
        return -b * (1/Γ + Γ + 2(i - 1) * γ) - mu
    end
end

n = 20 # number of spins in the chain
mu = 1.0 #magnetization bias at the edges
γ = 1.0 # dephasing rate in the bulk
Γ = 1.0 # coupling strength to the baths at the edges
t = 25 # final time for the time evolution of the density matrix

rho0 = State{Mixed}(System(n, Qubit()), "FullyMixed")
L = lindbladian_xx(n, mu, γ,Γ)

L_mpo = make_mpo(rho0, L) # MPO representation of the Lindbladian superoperator
println("Lindbladian MPO bond dimensions = ", linkdims(L_mpo))

#Time evolution of the density matrix
rho_t = approx_W(
    L,
    t, rho0; # final time and initial state
    nsweeps = 100, # number of time steps to reach time t
    w = 2,      # W^{II} scheme [Zaletel et al. (2015)]
    order = 3,  # discretization error is of order (dt)^(order+1)
    limits = Limits(cutoff = 1e-7, maxdim = 30), # truncation parameters for the MPS representation of |rho>>
    n_hermitianize = 10, # Make rho exactly hermitian every 10 steps to limit numerical errors
)

println("rho_t max bond dimension = ", maxlinkdim(rho_t))

mz = real.(expect1(rho_t, Z)) # magnetization <Z_i>
mz_exact = [sigma_z_steady(i, n, γ, Γ, mu) for i in 1:n] # exact NESS magnetization

println("n = ", n, ", mu = ", mu, ", γ = ", γ, ", t = ", t)

#print the magnetization profile and compare to the exact NESS result:
println("i, <Z_i>(t), <Z_i>(Exact NESS)")
for i in 1:n
    println(i, ", ", mz[i], ", ", mz_exact[i])
end
println("max |<Z_i>(t) - <Z_i>(NESS)| = ", maximum(abs.(mz - mz_exact)))
