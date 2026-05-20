from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np

from .compression import (
    CompressedField,
    CompressionConfig,
    MatrixLowRankField,
    TensorTrainField,
    compress_field,
    compression_would_help,
    operator_projection_config,
)
from .problem import (
    TaylorGreenParameters,
    create_grid,
    exact_solution,
    kinetic_energy,
    l2_velocity_error,
    stable_timestep,
)
from .spectral_solver import SimulationResult


FieldState = np.ndarray | MatrixLowRankField | TensorTrainField


@dataclass(frozen=True)
class FvSimulationConfig:
    nx: int
    ny: int | None = None
    t_end: float = 1.0
    dt: float | None = None
    cfl: float | None = None
    diffusive_safety: float = 0.12
    reconstruction: str = "first_order"
    compression: CompressionConfig | None = None


@dataclass(frozen=True)
class OperatorProjectionSummary:
    applied: bool
    dense_bytes: int = 0
    compressed_bytes: int = 0
    rank: float = 0.0
    relative_error: float = 0.0
    runtime_seconds: float = 0.0


def _angular_wavenumbers(n: int, length: float) -> np.ndarray:
    return 2.0 * math.pi * np.fft.fftfreq(n, d=length / n)


def _normalize_reconstruction(reconstruction: str) -> str:
    normalized = reconstruction.strip().lower().replace("-", "_")
    if normalized not in {"first_order", "muscl"}:
        raise ValueError(f"Unsupported FV reconstruction '{reconstruction}'.")
    return normalized


def _default_cfl(reconstruction: str) -> float:
    if reconstruction == "muscl":
        return 0.08
    return 0.1


def _recover_velocity(
    omega: np.ndarray,
    params: TaylorGreenParameters,
    kx: np.ndarray,
    ky: np.ndarray,
    k_squared: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    omega_hat = np.fft.fft2(omega)
    psi_hat = np.zeros_like(omega_hat)
    nonzero = k_squared > 0.0
    psi_hat[nonzero] = omega_hat[nonzero] / k_squared[nonzero]

    u_fluct = np.fft.ifft2(1j * ky[None, :] * psi_hat).real
    v_fluct = np.fft.ifft2(-1j * kx[:, None] * psi_hat).real
    return params.convection_u + u_fluct, params.convection_v + v_fluct


def _upwind_flux(velocity_face: np.ndarray, left_state: np.ndarray, right_state: np.ndarray) -> np.ndarray:
    return np.where(velocity_face >= 0.0, velocity_face * left_state, velocity_face * right_state)


def _minmod_three(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    same_sign = (np.sign(a) == np.sign(b)) & (np.sign(b) == np.sign(c))
    limited = np.minimum(np.abs(a), np.minimum(np.abs(b), np.abs(c)))
    return np.where(same_sign, np.sign(a) * limited, 0.0)


def _limited_slope(field: np.ndarray, axis: int) -> np.ndarray:
    backward = field - np.roll(field, 1, axis=axis)
    forward = np.roll(field, -1, axis=axis) - field
    centered = 0.5 * (forward + backward)
    return _minmod_three(centered, 2.0 * backward, 2.0 * forward)


def _face_states(
    field: np.ndarray,
    axis: int,
    reconstruction: str,
) -> tuple[np.ndarray, np.ndarray]:
    if reconstruction == "first_order":
        return field, np.roll(field, -1, axis=axis)

    slope = _limited_slope(field, axis=axis)
    left_state = field + 0.5 * slope
    right_state = np.roll(field - 0.5 * slope, -1, axis=axis)
    return left_state, right_state


def _materialize_state(state: FieldState) -> np.ndarray:
    if isinstance(state, np.ndarray):
        return state
    return state.to_dense()


def _stored_state_bytes(state: FieldState, auxiliary_bytes: int) -> int:
    if isinstance(state, np.ndarray):
        return int(auxiliary_bytes + state.nbytes)
    return int(auxiliary_bytes + state.compressed_bytes)


def _maybe_compress_state(
    omega: np.ndarray,
    step_index: int,
    compression: CompressionConfig | None,
) -> tuple[FieldState, float | None, float | None, float]:
    if compression is None:
        return omega, None, None, 0.0
    if (step_index + 1) % compression.compress_every != 0:
        return omega, None, None, 0.0
    if not compression_would_help(omega.shape, compression, dense_bytes=omega.nbytes):
        return omega, None, None, 0.0

    started = time.perf_counter()
    compressed = compress_field(omega, compression)
    elapsed = time.perf_counter() - started

    if compressed.compressed_bytes >= omega.nbytes / compression.min_benefit_ratio:
        return omega, None, None, elapsed
    return compressed.compressed, compressed.mean_rank, compressed.relative_error, elapsed


def _maybe_compress_operator_state(
    field: np.ndarray,
    rhs_evaluation_index: int,
    operator_compression: CompressionConfig | None,
) -> tuple[np.ndarray, CompressedField | None, OperatorProjectionSummary]:
    if operator_compression is None:
        return field, None, OperatorProjectionSummary(applied=False)
    if (rhs_evaluation_index + 1) % operator_compression.compress_every != 0:
        return field, None, OperatorProjectionSummary(applied=False)
    if not compression_would_help(field.shape, operator_compression, dense_bytes=field.nbytes):
        return field, None, OperatorProjectionSummary(applied=False)

    started = time.perf_counter()
    compressed = compress_field(field, operator_compression)
    elapsed = time.perf_counter() - started

    if compressed.compressed_bytes >= field.nbytes / operator_compression.min_benefit_ratio:
        return field, None, OperatorProjectionSummary(applied=False, runtime_seconds=elapsed)

    return compressed.approximation, compressed.compressed, OperatorProjectionSummary(
        applied=True,
        dense_bytes=field.nbytes,
        compressed_bytes=compressed.compressed_bytes,
        rank=compressed.mean_rank,
        relative_error=compressed.relative_error,
        runtime_seconds=elapsed,
    )


def _low_rank_laplacian(field: MatrixLowRankField, dx: float, dy: float) -> np.ndarray:
    left_second = (np.roll(field.left, -1, axis=0) - 2.0 * field.left + np.roll(field.left, 1, axis=0)) / (dx**2)
    right_second = (np.roll(field.right_t, -1, axis=1) - 2.0 * field.right_t + np.roll(field.right_t, 1, axis=1)) / (
        dy**2
    )
    diffusion_x = (left_second * field.singular_values) @ field.right_t
    diffusion_y = (field.left * field.singular_values) @ right_second
    return diffusion_x + diffusion_y


def _rhs(
    omega: np.ndarray,
    params: TaylorGreenParameters,
    dx: float,
    dy: float,
    kx: np.ndarray,
    ky: np.ndarray,
    k_squared: np.ndarray,
    reconstruction: str,
    operator_compression: CompressionConfig | None,
    rhs_evaluation_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[OperatorProjectionSummary, ...]]:
    omega_rhs, compressed_operator_state, operator_state_summary = _maybe_compress_operator_state(
        omega,
        rhs_evaluation_index,
        operator_compression,
    )
    u, v = _recover_velocity(omega_rhs, params, kx, ky, k_squared)

    u_face = 0.5 * (u + np.roll(u, -1, axis=0))
    v_face = 0.5 * (v + np.roll(v, -1, axis=1))
    omega_left_x, omega_right_x = _face_states(omega_rhs, axis=0, reconstruction=reconstruction)
    omega_left_y, omega_right_y = _face_states(omega_rhs, axis=1, reconstruction=reconstruction)

    flux_x = _upwind_flux(u_face, omega_left_x, omega_right_x)
    flux_y = _upwind_flux(v_face, omega_left_y, omega_right_y)
    divergence_x = (flux_x - np.roll(flux_x, 1, axis=0)) / dx
    divergence_y = (flux_y - np.roll(flux_y, 1, axis=1)) / dy
    divergence = divergence_x + divergence_y

    if isinstance(compressed_operator_state, MatrixLowRankField):
        laplacian = _low_rank_laplacian(compressed_operator_state, dx, dy)
    else:
        laplacian = (
            (np.roll(omega_rhs, -1, axis=0) - 2.0 * omega_rhs + np.roll(omega_rhs, 1, axis=0)) / (dx**2)
            + (np.roll(omega_rhs, -1, axis=1) - 2.0 * omega_rhs + np.roll(omega_rhs, 1, axis=1)) / (dy**2)
        )
    rhs = -divergence + params.viscosity * laplacian
    return rhs, u, v, (operator_state_summary,)


def run_fv_simulation(
    params: TaylorGreenParameters,
    config: FvSimulationConfig,
) -> SimulationResult:
    reconstruction = _normalize_reconstruction(config.reconstruction)
    nx = config.nx
    ny = nx if config.ny is None else config.ny
    dx = params.domain_length / nx
    dy = params.domain_length / ny
    x, y = create_grid(params.domain_length, nx=nx, ny=ny)
    initial = exact_solution(x, y, t=0.0, params=params)
    state: FieldState = initial.omega.copy()

    kx = _angular_wavenumbers(nx, params.domain_length)
    ky = _angular_wavenumbers(ny, params.domain_length)
    k_squared = kx[:, None] ** 2 + ky[None, :] ** 2

    cfl = config.cfl if config.cfl is not None else _default_cfl(reconstruction)
    dt = config.dt or stable_timestep(
        params=params,
        nx=nx,
        ny=ny,
        cfl=cfl,
        diffusive_safety=config.diffusive_safety,
    )
    steps = max(1, math.ceil(config.t_end / dt))
    dt = config.t_end / steps

    auxiliary_bytes = kx.nbytes + ky.nbytes + k_squared.nbytes
    base_state_bytes = initial.omega.nbytes + auxiliary_bytes
    persistent_state_bytes: list[int] = []
    state_compression_ranks: list[float] = []
    state_compression_errors: list[float] = []
    operator_compression_ranks: list[float] = []
    operator_compression_errors: list[float] = []
    operator_compression_ratios: list[float] = []
    compression_runtime_seconds = 0.0

    operator_compression = None
    if config.compression is not None and config.compression.compress_operators:
        operator_compression = operator_projection_config(config.compression)

    rhs_evaluation_index = 0
    started = time.perf_counter()
    for step in range(steps):
        omega0 = _materialize_state(state)

        rhs1, _u1, _v1, operator_summaries1 = _rhs(
            omega0,
            params,
            dx,
            dy,
            kx,
            ky,
            k_squared,
            reconstruction,
            operator_compression,
            rhs_evaluation_index,
        )
        rhs_evaluation_index += 1
        omega1 = omega0 + dt * rhs1

        rhs2, _u2, _v2, operator_summaries2 = _rhs(
            omega1,
            params,
            dx,
            dy,
            kx,
            ky,
            k_squared,
            reconstruction,
            operator_compression,
            rhs_evaluation_index,
        )
        rhs_evaluation_index += 1
        omega2 = 0.75 * omega0 + 0.25 * (omega1 + dt * rhs2)

        rhs3, _u3, _v3, operator_summaries3 = _rhs(
            omega2,
            params,
            dx,
            dy,
            kx,
            ky,
            k_squared,
            reconstruction,
            operator_compression,
            rhs_evaluation_index,
        )
        rhs_evaluation_index += 1
        omega_next = (omega0 + 2.0 * (omega2 + dt * rhs3)) / 3.0

        state, rank, compression_error, state_compression_seconds = _maybe_compress_state(
            omega_next,
            step,
            config.compression,
        )
        compression_runtime_seconds += state_compression_seconds
        persistent_state_bytes.append(_stored_state_bytes(state, auxiliary_bytes))
        if rank is not None:
            state_compression_ranks.append(rank)
            if compression_error is not None:
                state_compression_errors.append(compression_error)

        for summary in (*operator_summaries1, *operator_summaries2, *operator_summaries3):
            compression_runtime_seconds += summary.runtime_seconds
            if not summary.applied:
                continue
            operator_compression_ranks.append(summary.rank)
            operator_compression_errors.append(summary.relative_error)
            operator_compression_ratios.append(summary.dense_bytes / max(summary.compressed_bytes, 1))

    runtime_seconds = time.perf_counter() - started

    omega = _materialize_state(state)
    u, v = _recover_velocity(omega, params, kx, ky, k_squared)
    exact = exact_solution(x, y, t=config.t_end, params=params)
    energy = kinetic_energy(u, v, params)
    exact_energy = kinetic_energy(exact.u, exact.v, params)
    relative_energy_error = abs(energy - exact_energy) / max(abs(exact_energy), 1e-12)
    effective_state_bytes = int(round(float(np.mean(persistent_state_bytes)))) if persistent_state_bytes else base_state_bytes

    if config.compression is not None:
        solver_name = "compressed_fv"
    elif reconstruction == "muscl":
        solver_name = "fv_muscl"
    else:
        solver_name = "fv"

    return SimulationResult(
        reynolds=params.reynolds,
        nx=nx,
        ny=ny,
        dt=dt,
        steps=steps,
        runtime_seconds=runtime_seconds,
        state_bytes=base_state_bytes,
        l2_velocity_error=l2_velocity_error((u, v), (exact.u, exact.v)),
        kinetic_energy=energy,
        exact_kinetic_energy=exact_energy,
        relative_energy_error=relative_energy_error,
        x=x,
        y=y,
        u=u,
        v=v,
        omega=omega,
        exact_u=exact.u,
        exact_v=exact.v,
        exact_omega=exact.omega,
        solver=solver_name,
        effective_state_bytes=effective_state_bytes,
        compression_ratio=base_state_bytes / max(effective_state_bytes, 1),
        avg_compression_rank=float(np.mean(state_compression_ranks)) if state_compression_ranks else 0.0,
        avg_compression_error=float(np.mean(state_compression_errors)) if state_compression_errors else 0.0,
        avg_operator_compression_ratio=(
            float(np.mean(operator_compression_ratios)) if operator_compression_ratios else 1.0
        ),
        avg_operator_rank=float(np.mean(operator_compression_ranks)) if operator_compression_ranks else 0.0,
        avg_operator_compression_error=(
            float(np.mean(operator_compression_errors)) if operator_compression_errors else 0.0
        ),
        compression_runtime_seconds=compression_runtime_seconds,
    )
