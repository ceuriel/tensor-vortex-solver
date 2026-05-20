from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .compression import CompressionConfig
from .fv_solver import FvSimulationConfig, run_fv_simulation
from .problem import TaylorGreenParameters, recommended_resolution
from .spectral_solver import SimulationConfig, SimulationResult, run_simulation


COMPARISON_SOLVERS = ("spectral", "fv", "fv_muscl", "compressed_fv")

SOLVER_LABELS = {
    "spectral": "Spectral Baseline",
    "fv": "FV Upwind",
    "fv_muscl": "FV MUSCL-TVD",
    "compressed_fv": "Compressed FV (TT)",
}


def _normalize_solver_name(solver: str) -> str:
    normalized = solver.strip().lower().replace("-", "_")
    if normalized not in {"spectral", "fv", "fv_muscl", "compressed_fv", "compare"}:
        raise ValueError(f"Unsupported solver '{solver}'.")
    return normalized


def _write_csv(results: list[SimulationResult], output_path: Path) -> None:
    fieldnames = list(results[0].to_record().keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_record())


def _plot_metric(
    results: list[SimulationResult],
    output_path: Path,
    y_attr: str,
    y_label: str,
) -> None:
    reynolds = [result.reynolds for result in results]
    values = [getattr(result, y_attr) for result in results]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(reynolds, values, marker="o", linewidth=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Reynolds number")
    ax.set_ylabel(y_label)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_comparison_metric(
    results: list[SimulationResult],
    output_path: Path,
    y_attr: str,
    y_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    for solver in COMPARISON_SOLVERS:
        solver_results = [result for result in results if result.solver == solver]
        if not solver_results:
            continue
        solver_results.sort(key=lambda result: result.reynolds)
        ax.plot(
            [result.reynolds for result in solver_results],
            [getattr(result, y_attr) for result in solver_results],
            marker="o",
            linewidth=2,
            label=SOLVER_LABELS.get(solver, solver),
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Reynolds number")
    ax.set_ylabel(y_label)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_error_memory_tradeoff(results: list[SimulationResult], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    for solver in COMPARISON_SOLVERS:
        solver_results = [result for result in results if result.solver == solver]
        if not solver_results:
            continue
        solver_results.sort(key=lambda result: result.reynolds)
        x_values = [result.effective_state_bytes or result.state_bytes for result in solver_results]
        y_values = [result.l2_velocity_error for result in solver_results]
        ax.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=2,
            label=SOLVER_LABELS.get(solver, solver),
        )
        for result in solver_results:
            ax.annotate(
                f"Re={int(result.reynolds)}",
                (result.effective_state_bytes or result.state_bytes, result.l2_velocity_error),
                fontsize=7,
                xytext=(4, 4),
                textcoords="offset points",
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Persistent state memory [bytes]")
    ax.set_ylabel("Relative L2 velocity error")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_runtime_ratio(
    results: list[SimulationResult],
    output_path: Path,
    numerator_solver: str = "compressed_fv",
    denominator_solver: str = "fv_muscl",
) -> None:
    denominator = {
        result.reynolds: result
        for result in results
        if result.solver == denominator_solver
    }
    paired_results = [
        (result.reynolds, result.runtime_seconds / denominator[result.reynolds].runtime_seconds)
        for result in results
        if result.solver == numerator_solver and result.reynolds in denominator
    ]
    paired_results.sort(key=lambda item: item[0])

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot([item[0] for item in paired_results], [item[1] for item in paired_results], marker="o", linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel("Reynolds number")
    ax.set_ylabel(f"Runtime ratio: {SOLVER_LABELS[numerator_solver]} / {SOLVER_LABELS[denominator_solver]}")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_compression_runtime_share(results: list[SimulationResult], output_path: Path) -> None:
    compressed_results = [result for result in results if result.solver == "compressed_fv"]
    compressed_results.sort(key=lambda result: result.reynolds)
    shares = [
        (result.compression_runtime_seconds or 0.0) / max(result.runtime_seconds, 1e-12)
        for result in compressed_results
    ]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot([result.reynolds for result in compressed_results], shares, marker="o", linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel("Reynolds number")
    ax.set_ylabel("Compression runtime share")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _fit_loglog_growth(x_values: list[float], y_values: list[float]) -> float:
    if len(x_values) < 2:
        return 0.0
    safe_x = np.asarray([max(value, 1e-12) for value in x_values], dtype=float)
    safe_y = np.asarray([max(value, 1e-12) for value in y_values], dtype=float)
    slope, _intercept = np.polyfit(np.log(safe_x), np.log(safe_y), 1)
    return float(slope)


def _write_scaling_summary(results: list[SimulationResult], output_path: Path) -> None:
    fieldnames = [
        "solver",
        "min_reynolds",
        "max_reynolds",
        "runtime_growth_exponent",
        "memory_growth_exponent",
        "l2_error_growth_exponent",
        "runtime_growth_factor",
        "memory_growth_factor",
        "compression_runtime_share_at_max_re",
    ]

    rows: list[dict[str, float | str]] = []
    for solver in COMPARISON_SOLVERS:
        solver_results = [result for result in results if result.solver == solver]
        if not solver_results:
            continue
        solver_results.sort(key=lambda result: result.reynolds)
        reynolds = [result.reynolds for result in solver_results]
        runtimes = [result.runtime_seconds for result in solver_results]
        memories = [float(result.effective_state_bytes or result.state_bytes) for result in solver_results]
        errors = [result.l2_velocity_error for result in solver_results]

        rows.append(
            {
                "solver": solver,
                "min_reynolds": reynolds[0],
                "max_reynolds": reynolds[-1],
                "runtime_growth_exponent": _fit_loglog_growth(reynolds, runtimes),
                "memory_growth_exponent": _fit_loglog_growth(reynolds, memories),
                "l2_error_growth_exponent": _fit_loglog_growth(reynolds, errors),
                "runtime_growth_factor": runtimes[-1] / max(runtimes[0], 1e-12),
                "memory_growth_factor": memories[-1] / max(memories[0], 1e-12),
                "compression_runtime_share_at_max_re": (
                    (solver_results[-1].compression_runtime_seconds or 0.0) / max(solver_results[-1].runtime_seconds, 1e-12)
                ),
            }
        )

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _run_single_solver(
    solver: str,
    reynolds_values: list[float],
    base_resolution: int,
    t_end: float,
    compression_rank: int,
    compression_energy: float,
    compress_every: int,
    compression_method: str,
    operator_compression_method: str,
    operator_rank: int,
    operator_energy: float,
    operator_compress_every: int,
) -> list[SimulationResult]:
    results: list[SimulationResult] = []

    for reynolds in reynolds_values:
        params = TaylorGreenParameters(reynolds=reynolds)
        nx = recommended_resolution(reynolds=reynolds, base_resolution=base_resolution)

        if solver == "spectral":
            config = SimulationConfig(nx=nx, t_end=t_end)
            result = run_simulation(params=params, config=config)
        elif solver == "fv":
            config = FvSimulationConfig(nx=nx, t_end=t_end, reconstruction="first_order")
            result = run_fv_simulation(params=params, config=config)
        elif solver == "fv_muscl":
            config = FvSimulationConfig(nx=nx, t_end=t_end, reconstruction="muscl")
            result = run_fv_simulation(params=params, config=config)
        elif solver == "compressed_fv":
            config = FvSimulationConfig(
                nx=nx,
                t_end=t_end,
                reconstruction="muscl",
                compression=CompressionConfig(
                    target_rank=compression_rank,
                    energy_capture=compression_energy,
                    compress_every=compress_every,
                    method=compression_method,
                    operator_method=operator_compression_method,
                    operator_target_rank=operator_rank,
                    operator_energy_capture=operator_energy,
                    operator_compress_every=operator_compress_every,
                    compress_operators=operator_compression_method != "none",
                ),
            )
            result = run_fv_simulation(params=params, config=config)
        else:
            raise ValueError(f"Unsupported solver '{solver}'.")

        results.append(result)

    return results


def _write_single_solver_outputs(results: list[SimulationResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(results, output_dir / "benchmark_summary.csv")
    _plot_metric(results, output_dir / "runtime_vs_reynolds.png", "runtime_seconds", "Runtime [s]")
    _plot_metric(results, output_dir / "memory_vs_reynolds.png", "effective_state_bytes", "Persistent state memory [bytes]")
    _plot_metric(results, output_dir / "l2_error_vs_reynolds.png", "l2_velocity_error", "Relative L2 velocity error")
    _plot_metric(results, output_dir / "energy_error_vs_reynolds.png", "relative_energy_error", "Relative kinetic energy error")


def run_benchmark(
    reynolds_values: list[float],
    output_dir: str | Path = "results",
    base_resolution: int = 32,
    t_end: float = 1.0,
    solver: str = "spectral",
    compression_rank: int = 4,
    compression_energy: float = 0.9999,
    compress_every: int = 1,
    compression_method: str = "tensor_train",
    operator_compression_method: str = "matrix_power",
    operator_rank: int = 2,
    operator_energy: float = 0.999,
    operator_compress_every: int = 3,
) -> list[SimulationResult]:
    output_dir = Path(output_dir)
    solver = _normalize_solver_name(solver)

    if solver != "compare":
        results = _run_single_solver(
            solver=solver,
            reynolds_values=reynolds_values,
            base_resolution=base_resolution,
            t_end=t_end,
            compression_rank=compression_rank,
            compression_energy=compression_energy,
            compress_every=compress_every,
            compression_method=compression_method,
            operator_compression_method=operator_compression_method,
            operator_rank=operator_rank,
            operator_energy=operator_energy,
            operator_compress_every=operator_compress_every,
        )
        _write_single_solver_outputs(results, output_dir)
        return results

    combined_results: list[SimulationResult] = []
    for solver_name in COMPARISON_SOLVERS:
        solver_output_dir = output_dir / solver_name
        solver_results = _run_single_solver(
            solver=solver_name,
            reynolds_values=reynolds_values,
            base_resolution=base_resolution,
            t_end=t_end,
            compression_rank=compression_rank,
            compression_energy=compression_energy,
            compress_every=compress_every,
            compression_method=compression_method,
            operator_compression_method=operator_compression_method,
            operator_rank=operator_rank,
            operator_energy=operator_energy,
            operator_compress_every=operator_compress_every,
        )
        _write_single_solver_outputs(solver_results, solver_output_dir)
        combined_results.extend(solver_results)

    _write_csv(combined_results, output_dir / "comparison_summary.csv")
    _plot_comparison_metric(
        combined_results,
        output_dir / "comparison_runtime_vs_reynolds.png",
        "runtime_seconds",
        "Runtime [s]",
    )
    _plot_comparison_metric(
        combined_results,
        output_dir / "comparison_memory_vs_reynolds.png",
        "effective_state_bytes",
        "Persistent state memory [bytes]",
    )
    _plot_comparison_metric(
        combined_results,
        output_dir / "comparison_l2_error_vs_reynolds.png",
        "l2_velocity_error",
        "Relative L2 velocity error",
    )
    _plot_comparison_metric(
        combined_results,
        output_dir / "comparison_energy_error_vs_reynolds.png",
        "relative_energy_error",
        "Relative kinetic energy error",
    )
    _plot_error_memory_tradeoff(combined_results, output_dir / "comparison_error_vs_memory.png")
    _plot_runtime_ratio(combined_results, output_dir / "comparison_runtime_ratio_vs_reynolds.png")
    _plot_compression_runtime_share(combined_results, output_dir / "comparison_compression_runtime_share_vs_reynolds.png")
    _write_scaling_summary(combined_results, output_dir / "scaling_summary.csv")
    return combined_results
