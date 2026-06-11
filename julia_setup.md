# Running Julia, ITensors, and TensorMixedStates

## Option 1: Install Julia on a Linux PC

A first option is to install Julia directly on a personal computer. On a Linux
PC, open a terminal and run:

```bash
curl -fsSL https://install.julialang.org | sh
```

After the installation, close and reopen the terminal, then start Julia with:

```bash
julia
```

The packages used below can then be installed from the Julia prompt with the
same `Pkg.add` commands as in the Colab instructions.

## Option 2: Use Google Colab

Google Colab runs in a web browser and does not require a local Julia
installation. A notebook is made of cells. To run the code present in a cell,
use the play button on the left, or press `Shift+Enter`.

### 1. Create a new Colab notebook

1. Go to <https://colab.research.google.com/>.
2. Click `New notebook`.
3. Colab opens a notebook with a first code cell.


### 2. Switch the notebook from Python to Julia

By default, a new Colab notebook uses Python. For these exercises, the notebook
runtime should be switched from Python to Julia.

In the Colab menu:

1. Click `Execution` or `Runtime`, depending on the interface language.
2. Click `Change runtime type`.
3. In `Runtime type`, select `Julia`.
4. Click `Save`.


### 3. Check that Julia is running

Run this in a new cell:

```julia
VERSION
```

Then run:

```julia
println("Hello from Julia!")
```

### 4. Install ITensors, ITensorMPS, and TensorMixedStates

In a Julia cell, run:

```julia
using Pkg

Pkg.add(["ITensors", "ITensorMPS", "TensorMixedStates"])
```

This installs:

- `ITensors`: the core tensor library;
- `ITensorMPS`: matrix-product-state and matrix-product-operator tools;
- `TensorMixedStates`: tools for pure and mixed quantum states, especially open
  quantum systems.

The first installation can take several minutes. The first package import may
also take some time, because Julia compiles the packages.

Colab virtual machines are temporary. If Colab gives the notebook a fresh
machine, the packages may need to be reinstalled.

### 5. Import the packages in the notebook

After installation, run:

```julia
using ITensors
using ITensorMPS
using TensorMixedStates, .Qubits
```

These `using` lines should be run once after starting or reconnecting the
notebook.



### 6. Quick ITensorMPS test

Run this cell to check that ITensors and ITensorMPS are working:

```julia
using ITensors
using ITensorMPS

sites = siteinds("S=1/2", 4)
psi = random_mps(sites; linkdims=2)

println("Maximum MPS bond dimension = ", maxlinkdim(psi))
```

The output should say that the maximum MPS bond dimension is `2`.

### 7. Quick TensorMixedStates test

Run this cell to check that TensorMixedStates is working:

```julia
using TensorMixedStates, .Qubits

n = 4
system = System(n, Qubit())
rho0 = State{Mixed}(system, "FullyMixed")

println("Created a fully mixed state for ", length(system), " qubits.")
```
