using ITensors
using ITensorMPS

N = 6 # number of qubits
k = 3 # site where we want to compute <sigma^z_k>

sites = siteinds("Qubit", N)
psi0 = MPS(sites, fill("0", N)) # |000000>
psi1 = MPS(sites, fill("1", N)) # |111111>

x=0.25 # arbitrary mixing parameter
psi = x*psi0 + (1-x)*psi1 # MPS with bond dimension 2, but not in canonical form
normalize!(psi)

A = psi[k]            # MPS tensor at site k
Z = op("Z", sites[k]) # tensor representing the Pauli sigma^z operator on site k
wrong_expect_Z_k = scalar(dag(prime(A, sites[k])) * Z * A) 
println("wrong <sigma^z_$k> = ", real(wrong_expect_Z_k))

# Bring the MPS into mixed canonical with orthogonality center at site k
orthogonalize!(psi, k)
A = psi[k]
# Compute <sigma^z_k> using only A = psi[k] (thanks to the orthogonality center property)
expect_Z_k = scalar(dag(prime(A, sites[k])) * Z * A)
println("<sigma^z_$k> = ", real(expect_Z_k), "... should be ", (2*x-1)/(2*x*x-2*x+1))
