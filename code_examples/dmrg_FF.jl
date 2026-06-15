using ITensors
using ITensorMPS

# Exact free-fermion energy (open boundary conditions)
function exact_tightbinding_energy(N::Int)
    Nf = N ÷ 2
    eps = [-2.0 * cos(pi * m / (N + 1)) for m in 1:N]
    return sum(eps[1:Nf])
end

# DMRG benchmark for
# H = -t Σ_j (c†_j c_{j+1} + c†_{j+1} c_j)
function dmrg_free_fermion(N::Int)

    # Spinless fermions with conserved total fermion number
    sites = siteinds("Fermion", N; conserve_qns = true)
    #sites = siteinds("Fermion", N; conserve_qns = false)

    # Nearest-neighbor hopping Hamiltonian
    ampo = OpSum()
    for j in 1:(N - 1)
        ampo += -1.0, "Cdag", j, "C", j + 1
        ampo += -1.0, "Cdag", j + 1, "C", j
    end
    # Represent the Hamiltonian as an MPO
    H = MPO(ampo, sites)

    # Half-filled initial product state.
    init_state = [isodd(j) ? "1" : "0" for j in 1:N]  # Néel-like: 101010...
    #init_state = [j <= N ÷ 2 ? "1" : "0" for j in 1:N] # domain wall: 111...000 => slower convergence

    # Initial product MPS for the DMRG sweeps.
    psi0 = MPS(sites, init_state)

    # DMRG sweeps with gradual increase in bond dimension
    nsweeps = 8
    maxdim = [20, 40, 80, 120, 200]
    cutoff = [1e-6,1e-8,1e-10]
    energy_dmrg, psi = dmrg(H, psi0;
        nsweeps,
        maxdim,
        cutoff,
    )
    
    energy_exact = exact_tightbinding_energy(N)

    # Check the energy variance in the final state
    H2 = inner(H, psi, H, psi)
    E = inner(psi', H, psi)
    variance = H2 - E^2

    return (
        energy_dmrg = energy_dmrg,
        energy_exact = energy_exact,
        abs_error = abs(energy_dmrg - energy_exact),
        variance = variance,
        psi = psi
    )
end

# half filling on a 40-site chain
N=40
println("Running DMRG for free fermions on a chain of $N sites...")
result = dmrg_free_fermion(N)
println("Final bond dimension = ", maxlinkdim(result.psi))
println("DMRG energy   = ", result.energy_dmrg)
println("Exact energy  = ", result.energy_exact)
println("Absolute error = ", result.abs_error)
println("Variance       = ", result.variance)
