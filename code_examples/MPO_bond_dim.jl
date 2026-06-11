using TensorMixedStates, .Qubits 
using ITensorMPS: linkdims
n = 6
# MPO bond dimensions for a few examples of operators acting on pure or mixed states
h = sum(X(i) * X(i + 1) + Y(i) * Y(i + 1) for i in 1:(n - 1)) # XX spin chain Hamiltonian
psi0 = State{Pure}(System(n, Qubit()), "Up")
H_mpo = make_mpo(psi0, h) # psi0 needed so that make_mpo (from TensorMixedStates) knows the physical dim. of the tensors
println("XX Hamiltonian acting on a pure state: MPO bond dimensions = ", linkdims(H_mpo)) #output: 4

rho0 = State{Mixed}(System(n, Qubit()), "FullyMixed")
L = Evolver(-im * h) # Hamiltonian-only Lindbladian: d(rho)/dt = -i [H, rho]
L_mpo = make_mpo(rho0, L)
println("XX Hamiltonian acting on a mixed state density matrix: MPO bond dimensions = ", linkdims(L_mpo)) #output: 6 

L = sum(Dissipator(1.0 * Z)(i) for i in 1:n)
L_mpo = make_mpo(rho0, L)
println("Local dephasing Lindbladian: MPO bond dimensions = ", linkdims(L_mpo)) #output: 2

L = -im*h + sum(Dissipator(1.0 * Z)(i) for i in 1:n)
L_mpo = make_mpo(rho0, L)
println("XX Hamiltonian with local dephasing: MPO bond dimensions = ", linkdims(L_mpo)) #output: 6

# Example with two-site dephasing jump operators.
dephase2 = 1.0 * (Z ⊗ Z)
dissipators = sum(Dissipator(dephase2)(i, i+1) for i in 1:n-1)
L = -im*h + dissipators
L_mpo = make_mpo(rho0, L)
println("XX Hamiltonian with two-site dephasing: MPO bond dimensions = ", linkdims(L_mpo)) #output: 7
