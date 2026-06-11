using ITensors, ITensorMPS
let
N = 50
sites = siteinds("S=1/2",N)
os = OpSum()
for j=1:N-1
os += "Sz",j,"Sz",j+1
os += 1/2,"S+",j,"S-",j+1
os += 1/2,"S-",j,"S+",j+1
end
H = MPO(os,sites)
psi0 = random_mps(sites;linkdims=10)
nsweeps = 5
maxdim = [10,20,100,100,200]
cutoff = [1E-10]
energy,psi = dmrg(H,psi0;nsweeps,maxdim,cutoff)

# Compare with the exact ground state energy per bondin the thermodynamic limit (Bethe ansatz) e0 = -ln(2) + 1/4
e0 =- log(2) + 1/4
print("DMRG energy per bond: ", energy / (N-1), "\n")
print("Difference with 1/4-ln(2):", energy / (N-1) - e0, "\n")


# Compute the von Neumann entanglement entropy across the central bond.
i = N ÷ 2
orthogonalize!(psi, i)
# the virtual bond on the left of site i and the physical index (on site i) are grouped to define the matrix to be decomposed by SVD:
U, S, V = svd(psi[i], (linkind(psi, i - 1), siteind(psi, i)))
SvN = 0.0
for n in 1:dim(commonind(U, S))
    p = S[n, n]^2
    SvN -= p > 1e-14 ? p * log(p) : 0.0
end
print("Entanglement entropy of subsystem [1..",i,"] : ", SvN, "\n")
return
end