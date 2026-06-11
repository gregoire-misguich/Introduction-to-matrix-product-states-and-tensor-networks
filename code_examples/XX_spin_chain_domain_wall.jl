using LinearAlgebra
using LaTeXStrings
using Plots
using ITensors
using ITensorMPS

J = 1.0 # coupling strength
dt = 0.25 # time step 
cutoff = 1e-10 # MPS truncation cutoff
maxdim = 6 # maximum bond dimension of the MPS

# XX Hamiltonian with open boundaries: # H = (J/2) * sum_j (σ^x_j σ^x_{j+1} + σ^y_j σ^y_{j+1})
function exact_sigma_z(L::Int, t::Real)
	h = zeros(Float64, L, L)
	for j in 1:(L-1)
		h[j, j+1] = J
		h[j+1, j] = J
	end
	F = eigen(h) # diagonalize the single-particle Hamiltonian
	V = F.vectors
	ε = F.values
	U = V * Diagonal(exp.(-im .* ε .* t)) * V'
	# n[i] = <c†_i c_i> = sum_{j in occupied} |U[i, j]|^2
	n = [sum(abs2.(U[i, 1:div(L, 2)])) for i in 1:L]
	return 2 .* n .- 1
end

# <sigma^z> from MPS + TDVP
function tdvp_sigma_z(L::Int, t::Real)
	# Hilbert space for the MPS: spin-1/2 chain with S^z_{tot} conservation
	sites = siteinds("S=1/2", L; conserve_qns = true)

	# XX Hamiltonian as MPO
	os = OpSum()
	for j in 1:(L-1)
		os += J, "S+", j, "S-", j + 1
		os += J, "S-", j, "S+", j + 1
	end
	H = MPO(os, sites)

	# Initial state: domain-wall product state. left half: ↑, right half: ↓
	dw_states = [j <= div(L, 2) ? "↑" : "↓" for j in 1:L]
	psi0 = MPS(sites, dw_states)

	#time evolution: call to tdvp from ITensorMPS.jl
	psi_t = tdvp(-im * H, t, psi0;
		time_step = dt, cutoff = cutoff, maxdim = maxdim, nsite = 2)

	# Compute <σ^z> at each site
	mz = expect(psi_t, "Z")
	return real.(mz)
end

# Compute and plot exact magnetization vs TDVP
function plot_exact_vs_tdvp(L::Int, t::Real)

	mz_exact = exact_sigma_z(L, t)
	mz_tdvp = tdvp_sigma_z(L, t)

	p = plot(1:L, mz_exact;linewidth = 3,
        label = L"\mathrm{Exact\ (free\ fermions)}",
		xlabel = L"\mathrm{site}\ i",
		ylabel = L"\langle \sigma^z_i(t) \rangle",
		title = latexstring("\\mathrm{XX\\ domain-wall\\ quench},\\ Jt = ", t * J),
		size = (500, 330),
		dpi = 300,
	)
	scatter!(p, 1:L, mz_tdvp;
		markersize = 4,
		label = latexstring("\\mathrm{MPS + TDVP}\\ (\\mathrm{max.\\ dim.} = ",
			maxdim, ",\\ \\delta t = ", dt, ")"),
	)

	return p, mz_exact, mz_tdvp
end

# Example run

L = 40 # Lenth of the spin chain
t = 5.0 # time at which to evaluate <σ^z>

p, mz_exact, mz_tdvp = plot_exact_vs_tdvp(L, t)
display(p)
#savefig(p, "xx_domain_wall_exact_vs_tdvp.png")

# Maximum pointwise difference
err = maximum(abs.(mz_exact .- mz_tdvp))
println("max_i |mz_exact(i) - mz_tdvp(i)| = ", err)
