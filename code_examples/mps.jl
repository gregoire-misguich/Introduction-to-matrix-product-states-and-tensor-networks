using ITensors
using ITensorMPS
N = 10
sites = siteinds("S=1/2", N)
states = [isodd(j) ? "Up" : "Dn" for j in 1:N]
psi_neel = MPS(sites, states)
println("max. bond dim. = ", maxlinkdim(psi_neel))