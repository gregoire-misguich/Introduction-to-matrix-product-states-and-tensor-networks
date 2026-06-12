using LinearAlgebra
using ITensors
using ITensorMPS

# Snake (zig-zag) numbering
snake_index(x::Int, y::Int, Lx::Int) = isodd(y) ? (y - 1) * Lx + x : y * Lx - x + 1

function nearest_neighbor_bonds(Lx::Int, Ly::Int)
	horizontal = ((snake_index(x, y, Lx), snake_index(x + 1, y, Lx))
				  for y in 1:Ly for x in 1:(Lx - 1))
	vertical = ((snake_index(x, y, Lx), snake_index(x, y + 1, Lx))
				for y in 1:(Ly - 1) for x in 1:Lx)
	return Iterators.flatten((horizontal, vertical))
end
# Matrix for the single-particle tight-binding Hamiltonian
# H = -t sum_{<i,j>} (c_i^† c_j + h.c.) + open boundary conditions
function single_particle_hamiltonian(Lx::Int, Ly::Int)
	N = Lx * Ly
	H = zeros(Float64, N, N)
	for (i, j) in nearest_neighbor_bonds(Lx, Ly)
		H[i, j] = H[j, i] = -1.0
	end
	return H
end
# Compute the exact ground state at half filling by diagonalizing the single-particle Hamiltonian
# and summing the lowest N/2 single-particle energies
function exact_ground_state(Lx::Int, Ly::Int)
    N = Lx * Ly
    H = single_particle_hamiltonian(Lx, Ly)
    evals, evecs = eigen(H)
    Nf = div(N, 2)
    energy_gs = sum(evals[1:Nf])
    return energy_gs, evals, evecs
end
# Compute <n(i1)>, <n(i2)> and <n(i1)n(i2)>
# from the single-particle occupied modes provided by the function exact_ground_state
function density_density_observables_free(evecs::Matrix{Float64}, i::Int, j::Int, N::Int)
	Nf = div(N, 2)
	occupied_modes = @view evecs[:, 1:Nf]
	# One-body correlation matrix C_ij = <c_i^† c_j> for the Slater determinant
	# built from the N/2 lowest single-particle orbitals.
	C = occupied_modes * occupied_modes'
	ni = C[i, i]
	nj = C[j, j]
	ninj = i == j ? ni : ni * nj - C[i, j] * C[j, i]
	return ni, nj, ninj
end
# Compute <n(i1)>, <n(i2)> and <n(i1)n(i2)> from an MPS
# Uses ITensorMPS.correlation_matrix
function density_density_observables_mps(psi_mps::MPS, N::Int, i1::Int, i2::Int)
	Cnn = correlation_matrix(psi_mps, "N", "N"; ishermitian = true)
	n_  = expect(psi_mps, "N")
	return real(n_[i1]), real(n_[i2]), real(Cnn[i1, i2])
end
# von Neumann entanglement entropy across bond b
function entanglement_entropy!(psi::MPS, b::Int)
	# Move the orthogonality center of the MPS to site j.
	orthogonalize!(psi, b)
	U, S, V = svd(psi[b], (linkind(psi, b - 1), siteind(psi, b)))
	SvN = 0.0
	for n in 1:dim(commonind(U, S))
		p = S[n, n]^2
		if p > 1e-14
			SvN -= p * log(p)
		end
	end
	return real(SvN)
end
# Print the absolute and relative differences between exact and approximate values
function print_diff(exact_value::Real, approx_value::Real)
	abs_diff = abs(exact_value - approx_value)
	rel_diff = iszero(exact_value) ? NaN : abs_diff / abs(exact_value)
	println("\tabsolute difference   = $(round(abs_diff, sigdigits=5))")
	println("\trelative difference   = $(round(rel_diff, sigdigits=5))")
end

connected_correlation(obs) = obs[3] - obs[1] * obs[2]

function print_density_observables(i::Int, j::Int, obs)
	ni, nj, ninj = obs
	c = connected_correlation(obs)
	println("\t<n($i)> = $ni\t<n($j)> = $nj")
	println("\t<n($i) n($j)> = $ninj")
	println("\t<n($i) n($j)>^c = $c")
	return c
end
# Find the ground state using DMRG
function dmrg_ground_state(Lx::Int, Ly::Int; maxdim::Int, cutoff::Float64)
	N = Lx * Ly
	Nf = div(N, 2)
	sites = siteinds("Fermion", N; conserve_qns = true)
	os = OpSum() # ITensorMPS.OpSum for building the Hamiltonian as an MPO
	for (i, j) in nearest_neighbor_bonds(Lx, Ly)
		os += -1.0, "Cdag", i, "C", j
		os += -1.0, "Cdag", j, "C", i
	end
	H = MPO(os, sites) # actual construction of the Hamiltonian as an MPO
	# Start with a half-filled state
	state = [n <= Nf ? "Occ" : "Emp" for n in 1:N]
	psi0 = random_mps(sites, state; linkdims = min(maxdim, 10))
	nsweeps = 12
	energy, psi_gs = dmrg(H, psi0; nsweeps, maxdim = maxdim, cutoff)
	return energy, psi_gs
end
# Example
function main()
	Lx, Ly = 5, 6
	energy_exact, evals, evecs = exact_ground_state(Lx, Ly)
	N = Lx * Ly
	println("Square lattice: $Lx x $Ly = $N sites")
	println("Hilbert space dim. D = $(2^N), half filling N_fermions = $(N ÷ 2)")
	nzero = count(x -> isapprox(x, 0.0; atol = 1e-12), evals)
	if nzero > 0
		println("Warning: single-particle spectrum with $nzero zero modes => ground state degeneracy.")
	end

	# Exact density-density observables for two sites at opposite corners of the lattice
	i1, i2 = snake_index(1, 1, Lx), snake_index(Lx, Ly, Lx)
	println("\nExact observables:")
	println("\tground state energy = $energy_exact")
	obs_exact = density_density_observables_free(evecs, i1, i2, N)
	c_exact = print_density_observables(i1, i2, obs_exact)

	# Compute the ground state with "snake" DMRG 
	for maxdim in [50, 100, 200]
		println("\n== Computing the ground state with DMRG, max. bond dim= $maxdim ==")
		energy_dmrg, gs_dmrg = dmrg_ground_state(Lx, Ly; maxdim = maxdim, cutoff = 1e-10)

		# Compare the DMRG results with the exact results, for the energy
		println("\nDMRG ground state energy = $energy_dmrg with maxdim = $maxdim")
		println("Energy:")
		print_diff(energy_exact, energy_dmrg)

		# ... and for the density-density observables
		obs_dmrg = density_density_observables_mps(gs_dmrg, N, i1, i2)
		println("\nDMRG observables:")
		c_dmrg = print_density_observables(i1, i2, obs_dmrg)
		println("\tS_vN(center bond) = $(entanglement_entropy!(gs_dmrg, N ÷ 2))")
		println("\nConnected correlation <n($i1) n($i2)>^c:")
		print_diff(c_exact, c_dmrg)
	end
end

main()
