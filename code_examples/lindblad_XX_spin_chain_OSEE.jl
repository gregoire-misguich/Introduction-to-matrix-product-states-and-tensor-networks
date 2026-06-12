using TensorMixedStates, .Qubits
using DelimitedFiles, LaTeXStrings, Plots

N = 10 ; Γ = 0.5 ; tmax = 4.0
dir = "lindblad_results" # simulation name, used to create a directory with output files
limits = Limits(cutoff = 1e-10, maxdim = 20) # MPS truncation parameters

initial_state = [isodd(i) ? "Up" : "Dn" for i in 1:N] # Néel spin configuration
Lindbladian =
    -im * sum(X(i) * X(i + 1) + Y(i) * Y(i + 1) for i in 1:(N - 1)) + # XX spin chain Hamiltonian
    sum(Dissipator(sqrt(Γ) * Sm)(i) for i in 1:N) + # Jump operators: lowering operators σ⁻, rate Γ
    sum(Dissipator(sqrt(Γ) * Sp)(i) for i in 1:N)   # Jump operators: raising operators σ⁺, rate Γ
# 3 observables: center OSEE, center magnetization <σ^z_{N/2}>, TraceError=Tr[ρ(t)]-1
obs = ["osee.dat" => EE(N ÷ 2), "zmid.dat" => Z(N ÷ 2), "TraceError.dat" => TraceError] 
sim = SimData( # High-level simulation interface of TensorMixedStates
    name = dir,
    phases = [
        CreateState(# Initial state preparation
            type = Mixed(), system = System(N, Qubit()), state = initial_state), 
        Evolve(algo = ApproxW(order = 4, w = 2), # W^{II} scheme
        limits = limits, duration = tmax, time_step = 0.1,
           evolver = Lindbladian, measures = obs),
])
runTMS(sim; restart = true) # High-level simulation interface of TensorMixedStates
function read_data(file)
    data = readdlm(joinpath(dir, file), '\t')
    return data[:, 2], data[:, 3]
end

# Plot the results (from the output files)
t, osee = read_data("osee.dat")
_, zmid = read_data("zmid.dat")
p1 = plot(t, osee; label = false, xlabel = L"t", ylabel = L"\mathrm{OSEE}")
p2 = plot(t, zmid; label = false, xlabel = L"t", ylabel = L"\langle \sigma^z_{N/2}\rangle")
plot(p1, p2; layout = (1, 2), size = (500, 250), dpi = 300)
savefig(joinpath(dir, "lindblad_xx_OSEE.png"))