# Challenge Plan

## Purpose

This document summarizes the project in terms of the target enterprise challenge itself: what has been built, what has been demonstrated, how the current results map to the stated objective and expected outcomes, and what remains to close the gap.

## Challenge Mapping

### Challenge context

The target enterprise problem is looking for more scalable PDE solvers for aerodynamic simulation, motivated by the cost of current high-performance computing workflows and the limits of classical scaling for demanding operating conditions.

### Challenge objective

The challenge objective is to demonstrate an advantage over classical techniques on the Convecting Taylor-Green vortex benchmark, either through a fault-tolerant quantum algorithm or through a Tensor Network solver within a Finite Volume framework.

### Expected outcomes

The expected outcomes are:

1. a working solver for the 2D Convecting Taylor-Green vortex
2. scaling analysis for runtime, memory, and error as Reynolds number increases
3. comparison with strong classical solvers that demonstrates a quantum or quantum-inspired advantage

## What We Have Done So Far

The project currently delivers a complete benchmark stack around the same exact-solution problem:

- `spectral`: dealiased pseudo-spectral vorticity solver used as the reference accuracy floor
- `fv`: first-order finite-volume vorticity solver used as the simplest classical baseline
- `fv-muscl`: higher-order MUSCL-TVD finite-volume solver used as the stronger classical baseline
- `compressed-fv`: Tensor Network inspired finite-volume solver using tensor-train state compression, randomized truncated-SVD compression kernels, and operator-side compressed transport-state reuse through the finite-volume update

This is important because the project does not only present a compressed method in isolation. It presents:

- a benchmark with a known analytical solution
- an accuracy reference
- a simple classical comparison point
- a stronger classical finite-volume comparison point
- a Tensor Network finite-volume solver evaluated against all of the above

## How The Current Work Answers The Expected Outcomes

### 1. Working solver

Achieved.

The repository contains a working Tensor Network inspired finite-volume solver for the 2D Convecting Taylor-Green vortex:

- tensor-train compression is used for persistent state storage
- the compressed method operates inside a finite-volume MUSCL-TVD framework
- randomized truncated-SVD acceleration reduces compression overhead
- an operator-side compressed transport-state projection is reused across the convective and diffusive branches of the time-stepping loop

So the project satisfies the "Tensor Network within a Finite Volume framework" route in the challenge statement.

### 2. Scaling analysis

Achieved.

The curated benchmark sweep in `results/submission_core/comparison_summary.csv` covers:

- `Re = 10`
- `Re = 100`
- `Re = 300`
- `Re = 600`
- `Re = 900`

For each solver and Reynolds number, the benchmark reports:

- runtime
- persistent state memory
- relative L2 velocity error
- relative kinetic-energy error
- compression rank and compression error
- operator-compression behavior for the compressed solver
- runtime-growth exponents and time-to-solution factors in `results/submission_core/scaling_summary.csv`

This directly answers the required scaling study on runtime, memory, and error beyond `Re = 10` and `100`.

In the `results/submission_core/` folder we have only the challenge-facing results needed for our claim: aggregate solver comparison, runtime scaling, memory scaling, error scaling, and the memory-versus-fidelity tradeoff.

### 3. Comparison with classical techniques

Achieved, with an important nuance.

The current comparison is stronger than a single baseline comparison because it includes:

- a pseudo-spectral reference for near-best benchmark accuracy
- a simple first-order finite-volume baseline
- a stronger MUSCL-TVD finite-volume baseline
- the compressed Tensor Network finite-volume solver

The key nuance is:

- the project currently demonstrates a **quantum-inspired memory advantage**
- it does **not yet** demonstrate a universal runtime advantage over the strongest classical baseline


## What The Current Results Prove

The most important evidence is the comparison between `fv-muscl` and `compressed-fv`:

| Re | FV MUSCL memory [B] | Compressed memory [B] | Memory gain | FV MUSCL L2 error | Compressed L2 error |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10 | 16,896 | 9,664 | 1.75x | 8.216e-04 | 8.227e-04 |
| 100 | 174,720 | 89,872 | 1.94x | 1.554e-04 | 1.582e-04 |
| 300 | 498,432 | 252,784 | 1.97x | 5.798e-05 | 5.926e-05 |
| 600 | 988,032 | 499,120 | 1.98x | 2.977e-05 | 3.017e-05 |
| 900 | 1,483,520 | 746,992 | 1.99x | 1.996e-05 | 2.027e-05 |

From these results, we can state the following confidently:

- the stronger classical finite-volume baseline is significantly better than the first-order FV baseline
- the compressed solver stays very close to the stronger MUSCL baseline in error
- the memory advantage persists as the state size grows
- the solver remains stable and accurate beyond the minimal Reynolds points named in the challenge statement

More specifically:

- `fv-muscl` improves L2 error over `fv` by about `23.6x`, `119x`, `204.5x`, `289.2x`, and `354.8x` from `Re = 10` to `900`
- `compressed-fv` changes the MUSCL L2 error by only about `0.13%` to `2.2%`
- persistent memory reduction improves from `1.75x` at `Re=10` to `1.99x` at `Re=900`
- runtime growth with Reynolds number is now explicitly measured, with a fitted runtime-growth exponent of about `1.24` for `compressed-fv` versus about `1.26` for `fv-muscl`
- compression overhead has been reduced to about `12.2%` of total compressed runtime at `Re=900`, with the high-Re overhead now much smaller than in the earlier builds
- in the current single-run workstation sweep, `compressed-fv` is faster than `fv-muscl` at `Re=300` and `Re=600`, but not yet at `Re=10`, `100`, or `900`
- absolute wall-clock results should therefore be read together with the runtime-ratio and scaling summaries, rather than as a claim of a universal win already achieved

## What The Current Results Do Not Yet Prove

The project remains disciplined about what is and is not established:

- `spectral` still remains the reference accuracy floor
- `compressed-fv` has not surpassed the spectral solver on accuracy
- `compressed-fv` does not yet show a consistent runtime win over `fv-muscl` across the full Reynolds sweep, even though the runtime-growth fit is now slightly more favorable and the current run crosses the MUSCL baseline at mid-range Reynolds number

So the current project is best described as:

- a working Tensor Network finite-volume solver
- with validated memory savings
- with strong fidelity retention
- with scaling evidence beyond the minimum Reynolds points
- but without a final time-to-solution advantage yet

## Technical Story For The Submission

The final submission is framed in this order:

1. Starts from the exact analytical solution and the spectral reference to establish benchmark credibility.
2. Shows the simple and strong classical finite-volume baselines, so the comparison is honest.
3. Positions `compressed-fv` as the Tensor Network finite-volume method under evaluation.
4. Emphasizes that the current advantage is persistent memory reduction with fidelity retention inside the finite-volume family.
5. Adds the time-to-solution analysis explicitly: the compressed solver has a slightly better fitted growth trend than the MUSCL baseline, and the current run reaches absolute runtime wins at `Re=300` and `600`, but the win is not yet uniform.
6. Is explicit that the remaining gap is not whether the method works, but whether the runtime win can be made consistent and stronger at the top of the Reynolds sweep.

This gives us a clean submission narrative:

- validated benchmark
- strong classical reference
- working Tensor Network finite-volume solver
- memory scaling advantage
- improving time-to-solution scaling and beginning to cross the stronger FV baseline in the current sweep
- honest discussion of what remains

## Current Repository Role

The repository now serves four purposes relevant to the challenge:

- implementation of the solver stack
- reproducible scaling analysis
- reproducible classical versus compressed comparison
- submission-ready evidence for a quantum-inspired Tensor Network finite-volume approach

## Next Technical Priorities

The next steps should continue from the current evidence:

1. Reduce the remaining top-end compression overhead so the runtime gains seen at `Re=300` and `600` become more stable at `Re=900` and beyond.
2. Carry still more of the convective and flux operator path in compressed form, beyond the current compressed transport-state reuse.
3. Replace the remaining dense parts of operator application with more structured compressed updates where possible.
4. Extend the Reynolds and resolution sweep again after further runtime improvements so the next time-to-solution claim is both stronger and more consistent than the current `Re = 900` result.

## Conclussion

So far, the project already satisfies the challenge route of a Tensor Network solver within a Finite Volume framework, provides the requested scaling study, and demonstrates a clear quantum-inspired advantage in persistent memory usage while retaining finite-volume fidelity. The runtime analysis now also shows a slightly better fitted growth trend than the stronger MUSCL baseline and a partial absolute runtime advantage in the current mid-range sweep. The principal remaining gap is turning that partial win into a stronger and more consistent end-to-end time-to-solution advantage across the full Reynolds range.
