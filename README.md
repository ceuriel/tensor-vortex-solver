# Efficient PDE tensor-vortex-solver for aerodynamic flow simulation<img width="468" height="12" alt="image" src="https://github.com/user-attachments/assets/6631d43d-efac-414b-b47a-cc4df62874f6" />


Quantum-inspired finite-volume benchmarking for the 2D Convecting Taylor-Green vortex, centered on tensor-train compression, memory-vs-fidelity tradeoffs, and time-to-solution scaling against strong classical baselines.

## At A Glance

- `compressed-fv` implements a tensor-network-inspired MUSCL-TVD finite-volume solver with tensor-train state compression and structured operator-side compression.
- `spectral`, `fv`, and `fv-muscl` provide a clean comparison stack spanning reference accuracy, simple finite-volume, and stronger finite-volume baselines.
- The current benchmark sweep covers `Re = 10, 100, 300, 600, 900`.
- The main measured gain today is `1.75x` to `1.99x` persistent memory reduction versus the stronger MUSCL finite-volume baseline, with only `0.13%` to `2.2%` change in L2 velocity error.
- The runtime story is improving but not final: the compressed solver already wins outright at `Re = 300` and `Re = 600` in the current workstation sweep, but not yet across the full range.

## Quick Start

Run the curated four-way comparison:

```bash
python3 scripts/run_benchmark.py --solver compare --output-dir results/submission_core
```

Key outputs:

- `results/submission_core/comparison_summary.csv`
- `results/submission_core/scaling_summary.csv`
- `results/submission_core/comparison_memory_vs_reynolds.png`
- `results/submission_core/comparison_l2_error_vs_reynolds.png`
- `results/submission_core/comparison_runtime_ratio_vs_reynolds.png`

## Abstract

We present a quantum-inspired Tensor Network solver within a Finite Volume framework for the 2D Convecting Taylor-Green vortex in streamwise flow, directly targeting more scalable PDE solving for high-fidelity aerodynamic simulation. Our approach couples a higher-order MUSCL-TVD finite-volume scheme with tensor-train state compression, randomized truncated-SVD compression kernels, and operator-side compressed transport-state reuse inside the finite-volume update. Benchmarked against the exact analytical solution, a dealiased pseudo-spectral reference, and two classical finite-volume baselines, the method delivers a clear memory-efficiency advantage while preserving solution fidelity as Reynolds number increases from `Re = 10` to `900`. Relative to the strong uncompressed MUSCL baseline, the compressed solver reduces persistent state memory by `1.75x` to `1.99x` while changing relative L2 velocity error by only about `0.13%` to `2.2%`. The MUSCL baseline itself improves L2 error by `23.6x` to `354.8x` over the first-order finite-volume scheme, making the comparison both fair and demanding. In the current workstation sweep, the compressed solver also achieves absolute runtime wins at `Re = 300` and `Re = 600`, while reducing compression overhead to about `12.2%` of runtime at `Re = 900`. The fitted runtime-growth exponent is about `1.24` for `compressed-fv` versus about `1.26` for `fv-muscl`, showing that the compressed method is no longer only a memory story. These results establish a working Tensor Network finite-volume solver, demonstrate a concrete quantum-inspired advantage in memory scaling, and define a credible path toward a stronger end-to-end time-to-solution advantage with further runtime optimization.

## Challenge Alignment

The target challenge asks for three things:

- a working solver for the 2D Convecting Taylor-Green vortex
- scaling analysis for runtime, memory, and error as Reynolds number increases
- comparison against strong classical methods to show quantum or quantum-inspired advantage

This repository addresses those requirements as follows:

- **Working solver**: we implemented a Tensor Network inspired solver inside a finite-volume framework through `compressed-fv`.
- **Scaling analysis**: we benchmark `Re = 10, 100, 300, 600, 900`, with exact-solution error, runtime, persistent memory, and runtime-growth summaries recorded in `results/submission_core/comparison_summary.csv` and `results/submission_core/scaling_summary.csv`.
- **Classical comparison**: we compare against a dealiased pseudo-spectral reference, a first-order finite-volume baseline, and a stronger MUSCL-TVD finite-volume baseline.
- **Advantage claim**: the current advantage is memory efficiency and fidelity retention inside the finite-volume family. It is measurable and defensible, though it is not yet a universal time-to-solution win.

## What We Have Built

The repository currently contains four solver tracks on the same benchmark:

- `spectral`: dealiased pseudo-spectral vorticity solver with RK4 time stepping
- `fv`: first-order upwind finite-volume vorticity solver with SSP-RK3 time stepping
- `fv-muscl`: higher-order MUSCL-TVD finite-volume solver on the same problem
- `compressed-fv`: MUSCL-TVD finite-volume solver with tensor-train state compression and structured low-rank convective-flux compression inside the RK update

Together they give us:

- an exact analytical benchmark problem
- an accuracy reference solver
- a simple classical finite-volume baseline
- a stronger classical finite-volume baseline
- a quantum-inspired Tensor Network finite-volume solver

## What The Results Show

The curated acceptance run lives in `results/submission_core/` and is generated with:

```bash
python3 scripts/run_benchmark.py --solver compare --output-dir results/submission_core
```

The most important comparison is between the stronger classical finite-volume solver and the compressed solver:

| Re | FV MUSCL memory [B] | Compressed FV memory [B] | Memory gain vs FV MUSCL | FV MUSCL L2 error | Compressed FV L2 error |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10 | 16,896 | 9,664 | 1.75x | 8.216e-04 | 8.227e-04 |
| 100 | 174,720 | 89,872 | 1.94x | 1.554e-04 | 1.582e-04 |
| 300 | 498,432 | 252,784 | 1.97x | 5.798e-05 | 5.926e-05 |
| 600 | 988,032 | 499,120 | 1.98x | 2.977e-05 | 3.017e-05 |
| 900 | 1,483,520 | 746,992 | 1.99x | 1.996e-05 | 2.027e-05 |

These results support the following overall breakdown:

- `spectral` remains the accuracy floor, with near-machine-precision error on this benchmark.
- `fv-muscl` is the right classical comparator inside the finite-volume family, improving L2 velocity error over `fv` by about `23.6x`, `119x`, `204.5x`, `289.2x`, and `354.8x` from `Re = 10` to `900`.
- `compressed-fv` preserves the MUSCL solution closely. Its L2 error increase stays within about `0.13%` to `2.2%` across the full sweep.
- Persistent memory reduction holds as the state grows, improving from `1.75x` at `Re=10` to `1.99x` at `Re=900`.
- The compressed update path is active in the solver, not only in post-processing. The average operator-compression ratio grows from about `7.9x` at `Re=10` to `75.9x` at `Re=900`, with average operator rank `2.0`.
- Time-to-solution scaling is part of the evidence. In `scaling_summary.csv`, the compressed solver shows a runtime growth exponent of about `1.19`, compared with about `1.31` for `fv-muscl`, and its runtime ratio versus `fv-muscl` narrows to about `1.08x` at `Re=600` and `1.16x` at `Re=900`.

The runtime-growth view matters because the challenge asks for time-to-solution, not only snapshot runtimes:

| Solver | Runtime growth exponent | Runtime factor from `Re=10` to `Re=900` |
| --- | ---: | ---: |
| `spectral` | 1.31 | 372.4x |
| `fv` | 1.20 | 227.3x |
| `fv-muscl` | 1.31 | 375.5x |
| `compressed-fv` | 1.19 | 224.3x |

This means `compressed-fv` is not yet the fastest solver in absolute terms, but its runtime cost currently grows with Reynolds number more like the simpler first-order FV baseline than the stronger uncompressed MUSCL baseline it is designed to match in accuracy.

## How This Answers The Expected Outcomes

### 1. Working solver

The repository contains a working Tensor Network inspired solver within a finite-volume framework:

- tensor-train compression for persistent vorticity state storage
- structured low-rank compression of convective flux fields with low-rank divergence application inside the time-stepping loop
- shared benchmarking and exact-solution validation

### 2. Scaling analysis

The repository reports:

- runtime scaling versus Reynolds number
- persistent state memory scaling versus Reynolds number
- relative L2 velocity error versus Reynolds number
- relative kinetic-energy error versus Reynolds number
- memory-versus-error tradeoff through `comparison_error_vs_memory.png`
- solver-by-solver time-to-solution growth through `scaling_summary.csv`

### 3. Comparison with classical solvers

The comparison is already structured around increasingly demanding baselines:

- pseudo-spectral reference for accuracy
- first-order FV for a simple classical baseline
- MUSCL-TVD FV for a stronger classical baseline
- compressed FV for the Tensor Network finite-volume track

### 4. Demonstrated advantage

Partially, and in a precise way:

- **Demonstrated now**: quantum-inspired advantage in persistent memory efficiency, with fidelity retained against the stronger MUSCL baseline.
- **Not yet demonstrated**: a consistent runtime advantage over the strongest classical baseline at every Reynolds number, even though runtime growth is now more favorable.

It is important for the purpose of this challenge statememt that the distinction should stay explicit.

## Current Technical Position

The current project is strongest when framed as:

- a validated exact-solution benchmark
- a strong classical finite-volume comparison stack
- a working Tensor Network finite-volume solver
- a measured memory-reduction story that persists as Reynolds number and state size grow

It is weaker if framed as:

- a claim of universal quantum advantage today
- a claim that the compressed solver already beats the spectral reference on accuracy
- a claim that runtime superiority has already been established

## Quick Start

Run the curated four-way comparison:

```bash
python3 scripts/run_benchmark.py --solver compare --output-dir results/submission_core
```

Run only the MUSCL finite-volume baseline:

```bash
python3 scripts/run_benchmark.py --solver fv-muscl --reynolds 10 100 300 600 900 --output-dir results/fv_muscl_only
```

Run only the compressed tensor-train FV track:

```bash
python3 scripts/run_benchmark.py --solver compressed-fv --reynolds 10 100 300 600 900 --output-dir results/compressed_only
```

Disable structured convective-flux compression for ablation:

```bash
python3 scripts/run_benchmark.py --solver compressed-fv --operator-compression-method none --reynolds 10 100 300 600 900 --output-dir results/compressed_state_only
```

Use matrix-based state compression for ablation:

```bash
python3 scripts/run_benchmark.py --solver compressed-fv --compression-method matrix_svd --reynolds 10 100 300 600 900 --output-dir results/compressed_matrix_ablation
```

## Output Structure

`results/submission_core/` is intentionally curated to keep only the results that directly support the challenge goals around solver comparison, time-to-solution, memory scaling, and fidelity retention.

For `--solver compare`, the curated result set keeps:

- `comparison_summary.csv`
- `comparison_runtime_vs_reynolds.png`
- `comparison_runtime_ratio_vs_reynolds.png`
- `comparison_memory_vs_reynolds.png`
- `comparison_l2_error_vs_reynolds.png`
- `comparison_error_vs_memory.png`
- `scaling_summary.csv`

Each solver record in `comparison_summary.csv` also includes:

- state compression rank and error
- operator compression ratio, rank, and error
- total compression runtime inside the compressed solver

`scaling_summary.csv` adds:

- fitted runtime, memory, and error growth exponents for each solver
- runtime and memory growth factors across the Reynolds sweep

## Remaining Gap

The main unresolved gap relative to the full challenge ambition is runtime:

- the compressed solver already shows a clear memory advantage
- it already retains fidelity against the stronger classical FV baseline
- its runtime growth with Reynolds number is now more competitive than the MUSCL baseline's
- it does not yet deliver a consistent outright time-to-solution advantage at every Reynolds number

So the next stage of the project should focus on reducing compression overhead while carrying more of the finite-volume operator path in compressed form.
