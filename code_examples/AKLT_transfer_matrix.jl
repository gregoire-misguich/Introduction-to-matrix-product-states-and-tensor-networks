using LinearAlgebra
using ITensors

chi = 2 # bond dimension of the AKLT MPS

l = Index(chi, "left") # left virtual index
r = Index(chi, "right") # right virtual index
s = Index(3, "spin-1") # physical index

# Physical basis: s = 1, 2, 3 corresponds to |+>, |0>, |->.
A = ITensor(l, s, r) # three-leg MPS tensor with indices  (l, s, r)

# Tensor entries for the AKLT state
A[l => 1, s => 1, r => 2] = 1/sqrt(2) 
A[l => 1, s => 2, r => 1] = -1/2 
A[l => 2, s => 2, r => 2] = 1/2 
A[l => 2, s => 3, r => 1] = -1/sqrt(2) 

# Contract only the physical index, leaving the two doubled virtual indices open.
E = A * dag(prime(A, l, r))

row = combiner(l, prime(l); tags = "row") # combines the 2 left indices (l and l') into one row index  
col = combiner(r, prime(r); tags = "col") # combines the 2 right indices (r and r') into one column index
E_reshaped = row * E * col # reshape E into a two-leg tensor 
E_matrix = Array(E_reshaped, combinedind(row), combinedind(col)) # Convert reshaped E into a matrix

lambda = sort(eigvals(E_matrix); by = abs, rev = true)
println("largest two eigenvalues of E = ", lambda[1:2], " ratio |λ2/λ1| = ", abs(lambda[2] / lambda[1]))
println("correlation length ξ = -1/log(|λ2/λ1|)= ", -1 / log(abs(lambda[2] / lambda[1])))
