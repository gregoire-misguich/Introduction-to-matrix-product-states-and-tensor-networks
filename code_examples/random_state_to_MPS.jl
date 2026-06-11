using LinearAlgebra
using ITensors
using ITensorMPS

N = 14
psi = randn(2^N)
psi ./= norm(psi)

ITensors.disable_warn_order()
sites = siteinds(2, N)
psi_mps = MPS(psi, sites)

bond_dims = [dim(commonind(psi_mps[j], psi_mps[j + 1])) for j in 1:(N - 1)]
println("bond dimensions = ", bond_dims)
