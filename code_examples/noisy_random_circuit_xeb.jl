#using ITensors
#using ITensorMPS
using TensorMixedStates, .Qubits
using LinearAlgebra
using Printf
using Random
using Statistics

# Noisy random-circuit/XEB simulation example
# The state is represented as a TensorMixedStates mixed state (MPS)
# For p = 0 we use a pure-state MPS representation of the ideal circuit output

const N = 8
const D = 2
const PNOISE = 0.5
const NCIRCUITS = 25
const NSAMPLES = 100
const RNG = MersenneTwister(time_ns())
const LIMITS = Limits(cutoff = 1e-20, maxdim = 200)

# One-qubit depolarizing (noise) channel:
#   rho -> (1 - 3p/4) rho + (p/4) X rho X + (p/4) Y rho Y + (p/4) Z rho Z.
depolarizing(p) =
    (1 - 0.75p) * Gate(Id) + 0.25p * Gate(X) + 0.25p * Gate(Y) + 0.25p * Gate(Z)

# Generate a Haar-random two-qubit gate using a QR decomposition of a random complex matrix.
# Method: F. Mezzadri, https://arxiv.org/abs/math-ph/0609050
function random_two_qubit_gate(rng, name)
    z = randn(rng, 4, 4) + im * randn(rng, 4, 4)
    q, r = qr(z)
    phases = diag(r) ./ abs.(diag(r))
    u = Matrix(q) * Diagonal(phases)
    return Operator{2}(name, u, plain_op)
end

# Make the list of gates making up the random brick-wall circuit
function make_circuit(n, depth; rng)
    circuit = []
    gate_number = 0
    for layer in 1:depth
        first_site = isodd(layer) ? 1 : 2
        pairs = [(i, i + 1) for i in first_site:2:(n - 1)]
        gates = map(pairs) do _
            gate_number += 1
            random_two_qubit_gate(rng, "U$(gate_number)")
        end
        push!(circuit, (pairs = pairs, gates = gates))
    end
    return circuit
end

circuit_unitary_layer(layer, ::State{Pure}) =
    prod(layer.gates[k](i, j) for (k, (i, j)) in enumerate(layer.pairs))

# For custom two-site matrix operators, TensorMixedStates' mixed Gate representation
# uses the opposite site order from the pure-state representation.
circuit_unitary_layer(layer, ::State{Mixed}) =
    prod(layer.gates[k](j, i) for (k, (i, j)) in enumerate(layer.pairs))

# Compute the final state rho after applying the circuit and noise layers.
function run_circuit(circuit, n, pnoise)
    if pnoise > 0
        # rho below is a density matrix encoded as an MPO.
        rho = State{Mixed}(System(n, Qubit()), "0")
    else
        # rho below is of the form |psi><psi| and |psi> is encoded as an MPS.
        rho = State{Pure}(System(n, Qubit()), "0")
    end
    for layer in circuit
        unitary_layer = circuit_unitary_layer(layer, rho)
        rho = apply(unitary_layer, rho; limits = LIMITS)
        if pnoise > 0
            noise_layer = prod(depolarizing(pnoise)(i) for i in 1:n)
            rho = apply(noise_layer, rho; limits = LIMITS)
        end
    end
    return normalize(hermitianize(rho; limits = LIMITS))
end

# The probability of a given bitstring is the expectation value of this projector.
bit_projector(bits) = prod(Proj(b == 0 ? "Up" : "Dn")(i) for (i, b) in enumerate(bits))

basis_probability(rho, bits) =
    isempty(bits) ? 1.0 : max(real(expect(rho, bit_projector(bits))), 0.0)

function sample_bitstring(rho; rng)
    bits = Int[]
    for _ in 1:length(rho)
        p0 = basis_probability(rho, [bits; 0])
        p1 = basis_probability(rho, [bits; 1])
        norm = p0 + p1
        prob0 = norm > 0 ? clamp(p0 / norm, 0.0, 1.0) : 0.5
        push!(bits, rand(rng) < prob0 ? 0 : 1)
    end
    return bits
end

sample_bitstrings(rho, nsamples; rng) =
    [sample_bitstring(rho; rng = rng) for _ in 1:nsamples]

linear_xeb(sampled_ideal_probs, n) = 2^n * mean(sampled_ideal_probs) - 1

sampled_ideal_probs = Float64[]
center_cut = N ÷ 2
osee_values = Float64[]
svn_values = Float64[]

@printf("\nN = %d qubits, circuit depth D = %d, p = %.4f\n", N, D, PNOISE)
@printf("Using %d random circuits and %d samples per circuit\n", NCIRCUITS, NSAMPLES)

for circuit_number in 1:NCIRCUITS
    @printf("\nCircuit %d/%d\n", circuit_number, NCIRCUITS)
    circuit = make_circuit(N, D; rng = RNG)

    rho = run_circuit(circuit, N, PNOISE)
    print("Finished running noisy circuit -> rho\n")
    rho0 = run_circuit(circuit, N, 0.0)
    print("Finished running ideal circuit -> rho0 = |psi><psi|\n")

    @printf("trace(rho) = %.12f, purity(rho) = %.6f\n", real(trace(rho)), trace2(rho))
    @printf("trace(rho0) = %.12f, purity(rho0) = %.6f\n", real(trace(rho0)), trace2(rho0))
    @printf("max bond dimension of rho = %d\n", maxlinkdim(rho))
    @printf("max bond dimension of rho0 = %d\n", maxlinkdim(rho0))
    osee, _ = entanglement_entropy(rho, center_cut)
    svn_psi, _ = entanglement_entropy(rho0, center_cut)
    push!(osee_values, osee)
    push!(svn_values, svn_psi)
    @printf("OSEE of rho across center bond = %.6f\n", osee)
    @printf("entanglement entropy of |psi> across center bond = %.6f\n", svn_psi)

    @printf("Sampling %d bitstrings from the noisy state...\n", NSAMPLES)
    sampled_bitstrings = sample_bitstrings(rho, NSAMPLES; rng = RNG)
    append!(sampled_ideal_probs, [basis_probability(rho0, bits) for bits in sampled_bitstrings])
end

sampled_xeb = linear_xeb(sampled_ideal_probs, N)
@printf("\nlinear XEB from %d circuits and %d noisy samples per circuit = %.6f\n", NCIRCUITS, NSAMPLES, sampled_xeb)
@printf("average OSEE of rho across center bond = %.6f\n", mean(osee_values))
@printf("average entanglement entropy of |psi> across center bond = %.6f\n", mean(svn_values))
