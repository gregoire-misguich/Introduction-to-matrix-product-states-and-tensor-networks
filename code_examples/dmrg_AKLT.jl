using ITensors, ITensorMPS
let
N = 10
J = 1.0
J2 = J / 3 # AKLT point
sites = siteinds("S=1", N)
os = OpSum()
spin_ops = ("Sx", "Sy", "Sz")

for j in 1:(N - 1)
    for Sa in spin_ops
        os += J, Sa, j, Sa, j + 1 #Heisenberg term
        for Sb in spin_ops
            os += J2, Sa, j, Sa, j + 1, Sb, j, Sb, j + 1 # biquadratic term
        end
    end
end

H = MPO(os, sites)
psi0 = random_mps(sites; linkdims = 3)
nsweeps = 6

# Below we allow for a maximal bond dimension = 10, but the actual bond
# dimension at the end of the DMRG sweeps is only 2 for the zero-field AKLT
# ground state, since it is exactly an MPS with bond dimension 2.
maxdim = [10]
cutoff = [1E-10]
energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff)
energy_exact = -2 / 3 * (N - 1) * J
println("DMRG energy: ", energy, " Exact energy: ", energy_exact)
return
end
