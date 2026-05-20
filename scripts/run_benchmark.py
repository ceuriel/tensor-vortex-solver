from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tensor_vortex_solver import run_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Reynolds scaling experiments for the tensor-vortex solver benchmark."
    )
    parser.add_argument(
        "--solver",
        default="spectral",
        choices=["spectral", "fv", "fv-muscl", "compressed-fv", "compare"],
        help="Solver to benchmark. 'compare' runs spectral, first-order FV, MUSCL FV, and compressed FV side by side.",
    )
    parser.add_argument(
        "--reynolds",
        nargs="+",
        type=float,
        default=[10.0, 100.0, 300.0, 600.0, 900.0],
        help="Reynolds numbers to simulate.",
    )
    parser.add_argument(
        "--base-resolution",
        type=int,
        default=32,
        help="Baseline grid size used before Reynolds scaling is applied.",
    )
    parser.add_argument(
        "--compression-rank",
        type=int,
        default=4,
        help="Maximum retained rank for the compressed finite-volume solver.",
    )
    parser.add_argument(
        "--compression-energy",
        type=float,
        default=0.9999,
        help="Target spectral energy capture for the compressed finite-volume solver.",
    )
    parser.add_argument(
        "--compress-every",
        type=int,
        default=1,
        help="Compress every N time steps in the compressed finite-volume solver.",
    )
    parser.add_argument(
        "--compression-method",
        default="tensor_train",
        choices=["tensor_train", "matrix_svd"],
        help="Compression model used by the compressed finite-volume solver.",
    )
    parser.add_argument(
        "--operator-compression-method",
        default="matrix_power",
        choices=["matrix_svd", "matrix_power", "tensor_train", "none"],
        help="Operator or flux compression model used inside the compressed finite-volume update path.",
    )
    parser.add_argument(
        "--operator-rank",
        type=int,
        default=2,
        help="Maximum retained rank for operator-aware compression inside the compressed finite-volume solver.",
    )
    parser.add_argument(
        "--operator-energy",
        type=float,
        default=0.999,
        help="Target spectral energy capture for operator-aware compression inside the compressed finite-volume solver.",
    )
    parser.add_argument(
        "--operator-compress-every",
        type=int,
        default=3,
        help="Compress operator-side fields every N RHS evaluations in the compressed finite-volume solver.",
    )
    parser.add_argument(
        "--t-end",
        type=float,
        default=1.0,
        help="Final integration time.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory where CSV and plots will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_benchmark(
        reynolds_values=args.reynolds,
        output_dir=args.output_dir,
        base_resolution=args.base_resolution,
        t_end=args.t_end,
        solver=args.solver,
        compression_rank=args.compression_rank,
        compression_energy=args.compression_energy,
        compress_every=args.compress_every,
        compression_method=args.compression_method,
        operator_compression_method=args.operator_compression_method,
        operator_rank=args.operator_rank,
        operator_energy=args.operator_energy,
        operator_compress_every=args.operator_compress_every,
    )

    print("Completed benchmark sweep:")
    for result in results:
        compression_note = ""
        if (result.compression_ratio or 1.0) > 1.0:
            compression_note = (
                f" comp={result.compression_ratio:>5.2f}x"
                f" rank={result.avg_compression_rank:>4.1f}"
            )
            if (result.avg_operator_rank or 0.0) > 0.0:
                compression_note += (
                    f" op_comp={result.avg_operator_compression_ratio:>5.2f}x"
                    f" op_rank={result.avg_operator_rank:>4.1f}"
                )
        print(
            f"  solver={result.solver:<13} "
            f"Re={result.reynolds:>7.1f} "
            f"grid={result.nx}x{result.ny} "
            f"steps={result.steps:>5d} "
            f"runtime={result.runtime_seconds:>8.3f}s "
            f"memory={result.effective_state_bytes or result.state_bytes:>9d}B "
            f"L2={result.l2_velocity_error:>10.3e} "
            f"energy_err={result.relative_energy_error:>10.3e}"
            f"{compression_note}"
        )


if __name__ == "__main__":
    main()
