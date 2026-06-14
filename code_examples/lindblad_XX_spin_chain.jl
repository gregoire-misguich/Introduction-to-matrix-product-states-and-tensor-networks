using TensorMixedStates, .Qubits 
using ITensorMPS: linkdims

function lindbladian_xx(N, mu, γ,Γ)
    # XX spin chain Hamiltonian
    h = sum(X(i) * X(i + 1) + Y(i) * Y(i + 1) for i in 1:(N - 1))
    # Boundary driving / coupling to baths at the edges 
    baths =
        Dissipator(sqrt((1 - mu)*Γ*2) * Sp)(1) +
        Dissipator(sqrt((1 + mu)*Γ*2) * Sm)(1) +
        Dissipator(sqrt((1 + mu)*Γ*2) * Sp)(N) +
        Dissipator(sqrt((1 - mu)*Γ*2) * Sm)(N)

    # Dephasing in the bulk, with rate γ
    dephasing = sum(Dissipator(sqrt(γ) * Z)(i) for i in 1:N)
    return Evolver(-im * h) + baths + dephasing
    
end

# Exact NESS magnetization profile for this model: M. Žnidarič J. Stat. Mech. (2010) L05002
function sigma_z_steady(i::Integer, N::Integer, γ::Real, Γ::Real, mu::Real)
    b = -mu / (Γ + 1/Γ + (N - 1) * γ)
    if i == 1
        return -b / Γ - mu #magnetization at the left edge
    elseif i == N
        return -b * (1/Γ + 2Γ + 2(N - 1) * γ) - mu # magnetization at the right edge
    else
        return -b * (1/Γ + Γ + 2(i - 1) * γ) - mu # magnetization in the bulk
    end
end

N = 8 # number of spins in the chain
mu = 1.0 # magnetization bias at the edges
γ = 1.0 # dephasing rate in the bulk
Γ = 1.0 # coupling strength to the baths at the edges
t = 10 # final time
nsteps = 60 # number of time steps to reach time t
rho0 = State{Mixed}(System(N, Qubit()), "FullyMixed") # initial state
maxdim = 10 # maximum bond dimension for the MPS representation of |rho>>
L = lindbladian_xx(N, mu, γ,Γ)

L_mpo = make_mpo(rho0, L) # MPO of the Lindbladian superoperator
println("Lindbladian MPO bond dimensions = ", linkdims(L_mpo))
W=true
if (W)
    rho_t = approx_W( #Lindblad evolution with W^{II} MPO method
        L,
        t, rho0; # final time and initial state
        nsweeps = nsteps,
        w = 2,      # W^{II} scheme [Zaletel et al. (2015)]
        order = 4,  # discretization error is of order (dt)^(order+1)
        limits = Limits(cutoff = 1e-20, maxdim = maxdim), # truncation parameters for the MPS of |rho>>
        n_hermitianize = 10, # Make rho exactly hermitian every 10 steps to limit numerical errors
    )
else    
    rho_t = tdvp(#Lindblad evolution with TDVP
        L, t, rho0;
        limits = Limits(cutoff = 1e-20, maxdim = maxdim),
        nsteps=nsteps,
        nsite=2, #two-site TDVP allows for bond dimension to grow
    )
end
println("rho_t max bond dimension = ", maxlinkdim(rho_t))

mz = real.(expect1(rho_t, Z)) # magnetization <Z_i>
mz_exact = [sigma_z_steady(i, N, γ, Γ, mu) for i in 1:N] # exact NESS magnetization

println("N = ", N, ", mu = ", mu, ", γ = ", γ, ", t = ", t, ", dt = ", t/nsteps, ", max dim. = ", maxdim)

#print the magnetization profile and compare to the exact NESS :
println("i, <Z_i>(t), <Z_i>(Exact NESS)")
for i in 1:N
    println(i, ", ", mz[i], ", ", mz_exact[i])
end
println("max |<Z_i>(t) - <Z_i>(NESS)| = ", maximum(abs.(mz - mz_exact)))
