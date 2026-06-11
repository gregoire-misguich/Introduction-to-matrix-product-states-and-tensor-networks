using ITensors, ITensorMPS

N = 12
k = 4 # Ising couplings up to distance k
# 1/r^6 van der Waals interactions up to distance k
J = [1.0 / r^6 for r in 1:k]

h = 0.7 # transverse field

sites = siteinds("S=1/2", N)

os = OpSum()
for r in 1:k
    for j in 1:(N - r)
        global os += J[r], "Sz", j, "Sz", j + r
    end
end
for j in 1:N
    global os += h, "Sx", j
end

H = MPO(os, sites)
bond_dims = [dim(linkind(H, j)) for j in 1:(N - 1)]
println("MPO bond dimensions = ", bond_dims)
