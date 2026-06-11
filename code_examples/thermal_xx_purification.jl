using ITensors
using ITensorMPS
using Printf

# Finite-temperature XX chain from purification and imaginary-time evolution with TDVP

const N = 8
const J = 1.0
const BETA_VALUES = 0.0:1.0:5.0
const IMAG_TIME_STEP = 0.1
const CUTOFF = 1e-12
const MAXDIM = 10 # Maximum bond dimension for TDVP evolution

# Physical sites are placed on odd positions of the doubled chain.
# Ancilla sites are placed on even positions:
physical(i) = 2i - 1
ancilla(i) = 2i

function xx_hamiltonian(sites, n; J=1.0, site=physical)
    os = OpSum()
    for i in 1:(n - 1)
        p = site(i)
        q = site(i + 1)
        os += J / 2, "S+", p, "S-", q
        os += J / 2, "S-", p, "S+", q
    end
    return MPO(os, sites)
end

function infinite_temperature_purification(sites, n)
    psi = MPS(sites, "0")

    # The infinite-temperature purification is a product of Bell pairs (|00> + |11>) / sqrt(2)
    # between physical and ancilla sites
    gates = ITensor[] #empty array whose element type is ITensor
    for i in 1:n
        push!(gates, op("H", sites[physical(i)])) # Hadamard gate 
        push!(gates, op("CX", sites[physical(i)], sites[ancilla(i)])) # CNOT=Controlled-X gate
    end
    psi = apply(gates, psi; cutoff=CUTOFF, maxdim=2)
    normalize!(psi)
    return psi
end

function thermal_purification_tdvp(psi0, hamiltonian, beta)
    # Computes |Psi(beta)> = exp(-beta H / 2) |Psi(0)>
    if iszero(beta)
        return copy(psi0)
    end
    tau = beta / 2
    # ITensorMPS TDVP requires tau / time_step to be an integer.
    nsteps = max(1, ceil(Int, tau / IMAG_TIME_STEP))
    # Imaginary-time evolution with TDVP
    psi_beta = tdvp(
        -hamiltonian,
        tau,
        psi0;
        time_step=tau / nsteps,
        cutoff=CUTOFF,maxdim=MAXDIM,
        nsite=2,#two-site TDVP allows for bond dimension growth
        normalize=true,
        outputlevel=0,
    )
    return psi_beta
end

function energy(psi, hamiltonian)
    return real(inner(psi', hamiltonian, psi) / inner(psi, psi))
end

function exact_xx_energy(n, betaJ)
    # The Jordan-Wigner transformation maps the XX chain to free fermions with
    # single-particle energies eps_m = J * cos(pi m / (n + 1)).
    return sum(1:n) do m
        eps = cos(pi * m / (n + 1))
        eps / (1 + exp(betaJ * eps))
    end
end

sites = siteinds("Qubit", 2N)
h = xx_hamiltonian(sites, N; J)
psi0 = infinite_temperature_purification(sites, N)

@printf("Open XX chain with N = %d spins\n", N)
@printf("%8s  %16s  %16s  %12s  %8s\n",
        "beta", "E_TDVP(beta)", "E_exact(beta)", "abs error", "maxdim")

for beta in BETA_VALUES
    psi_beta = thermal_purification_tdvp(psi0, h, beta)
    energy_tdvp = energy(psi_beta, h)
    energy_exact = exact_xx_energy(N, beta*J)

    @printf("%8.3f  %16.10f  %16.10f  %12.3e  %8d\n",
            beta,
            energy_tdvp,
            energy_exact,
            abs(energy_tdvp - energy_exact),
            maxlinkdim(psi_beta))
end