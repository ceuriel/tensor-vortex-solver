from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np

from .problem import (
    TaylorGreenParameters,
    create_grid,
    exact_solution,
    kinetic_energy,
    l2_velocity_error,
    stable_timestep,
)


@dataclass(frozen=True)
class SimulationConfig:
    nx: int
    ny: int | None = None
    t_end: float = 1.0
    dt: float | None = None
    cfl: float = 0.15
    diffusive_safety: float = 0.2


@dataclass
class SimulationResult:
    reynolds: float
    nx: int
    ny: int
    dt: float
    steps: int
    runtime_seconds: float
    state_bytes: int
    l2_velocity_error: float
    kinetic_energy: float
    exact_kinetic_energy: float
    relative_energy_error: float
    x: np.ndarray
    y: np.ndarray
    u: np.ndarray
    v: np.ndarray
    omega: np.ndarray
    exact_u: np.ndarray
    exact_v: np.ndarray
    exact_omega: np.ndarray
    solver: str = "spectral"
    effective_state_bytes: int | None = None
    compression_ratio: float | None = None
    avg_compression_rank: float | None = None
    avg_compression_error: float | None = None
    avg_operator_compression_ratio: float | None = None
    avg_operator_rank: float | None = None
    avg_operator_compression_error: float | None = None
    compression_runtime_seconds: float | None = None

    def to_record(self) -> dict[str, float | int | str]:
        return {
            "solver": self.solver,
            "reynolds": self.reynolds,
            "nx": self.nx,
            "ny": self.ny,
            "dt": self.dt,
            "steps": self.steps,
            "runtime_seconds": self.runtime_seconds,
            "state_bytes": self.state_bytes,
            "effective_state_bytes": self.effective_state_bytes or self.state_bytes,
            "compression_ratio": self.compression_ratio if self.compression_ratio is not None else 1.0,
            "avg_compression_rank": self.avg_compression_rank if self.avg_compression_rank is not None else 0.0,
            "avg_compression_error": self.avg_compression_error if self.avg_compression_error is not None else 0.0,
            "avg_operator_compression_ratio": (
                self.avg_operator_compression_ratio if self.avg_operator_compression_ratio is not None else 1.0
            ),
            "avg_operator_rank": self.avg_operator_rank if self.avg_operator_rank is not None else 0.0,
            "avg_operator_compression_error": (
                self.avg_operator_compression_error if self.avg_operator_compression_error is not None else 0.0
            ),
            "compression_runtime_seconds": (
                self.compression_runtime_seconds if self.compression_runtime_seconds is not None else 0.0
            ),
            "l2_velocity_error": self.l2_velocity_error,
            "kinetic_energy": self.kinetic_energy,
            "exact_kinetic_energy": self.exact_kinetic_energy,
            "relative_energy_error": self.relative_energy_error,
        }


def _angular_wavenumbers(n: int, length: float) -> np.ndarray:
    return 2.0 * math.pi * np.fft.fftfreq(n, d=length / n)


def _dealias_mask(kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    cutoff_x = (2.0 / 3.0) * np.max(np.abs(kx))
    cutoff_y = (2.0 / 3.0) * np.max(np.abs(ky))
    return (np.abs(kx[:, None]) <= cutoff_x) & (np.abs(ky[None, :]) <= cutoff_y)


def _velocity_and_vorticity_derivatives(
    omega_hat: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    k_squared: np.ndarray,
    params: TaylorGreenParameters,
    dealias: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    filtered = omega_hat * dealias
    psi_hat = np.zeros_like(filtered)
    nonzero = k_squared > 0.0
    psi_hat[nonzero] = filtered[nonzero] / k_squared[nonzero]

    u_fluct = np.fft.ifft2(1j * ky[None, :] * psi_hat).real
    v_fluct = np.fft.ifft2(-1j * kx[:, None] * psi_hat).real
    omega_x = np.fft.ifft2(1j * kx[:, None] * filtered).real
    omega_y = np.fft.ifft2(1j * ky[None, :] * filtered).real

    u = params.convection_u + u_fluct
    v = params.convection_v + v_fluct
    omega = np.fft.ifft2(filtered).real
    return u, v, omega, omega_x, omega_y


def _rhs(
    omega_hat: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    k_squared: np.ndarray,
    params: TaylorGreenParameters,
    dealias: np.ndarray,
) -> np.ndarray:
    u, v, _omega, omega_x, omega_y = _velocity_and_vorticity_derivatives(
        omega_hat=omega_hat,
        kx=kx,
        ky=ky,
        k_squared=k_squared,
        params=params,
        dealias=dealias,
    )
    nonlinear = u * omega_x + v * omega_y
    nonlinear_hat = np.fft.fft2(nonlinear) * dealias
    return -nonlinear_hat - params.viscosity * k_squared * omega_hat


def _rk4_step(
    omega_hat: np.ndarray,
    dt: float,
    kx: np.ndarray,
    ky: np.ndarray,
    k_squared: np.ndarray,
    params: TaylorGreenParameters,
    dealias: np.ndarray,
) -> np.ndarray:
    k1 = _rhs(omega_hat, kx, ky, k_squared, params, dealias)
    k2 = _rhs(omega_hat + 0.5 * dt * k1, kx, ky, k_squared, params, dealias)
    k3 = _rhs(omega_hat + 0.5 * dt * k2, kx, ky, k_squared, params, dealias)
    k4 = _rhs(omega_hat + dt * k3, kx, ky, k_squared, params, dealias)
    return omega_hat + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def run_simulation(
    params: TaylorGreenParameters,
    config: SimulationConfig,
) -> SimulationResult:
    nx = config.nx
    ny = nx if config.ny is None else config.ny
    x, y = create_grid(params.domain_length, nx=nx, ny=ny)
    initial = exact_solution(x, y, t=0.0, params=params)
    omega_hat = np.fft.fft2(initial.omega)

    kx = _angular_wavenumbers(nx, params.domain_length)
    ky = _angular_wavenumbers(ny, params.domain_length)
    k_squared = kx[:, None] ** 2 + ky[None, :] ** 2
    dealias = _dealias_mask(kx, ky)

    dt = config.dt or stable_timestep(
        params=params,
        nx=nx,
        ny=ny,
        cfl=config.cfl,
        diffusive_safety=config.diffusive_safety,
    )
    steps = max(1, math.ceil(config.t_end / dt))
    dt = config.t_end / steps

    started = time.perf_counter()
    for _ in range(steps):
        omega_hat = _rk4_step(
            omega_hat=omega_hat,
            dt=dt,
            kx=kx,
            ky=ky,
            k_squared=k_squared,
            params=params,
            dealias=dealias,
        )
    runtime_seconds = time.perf_counter() - started

    u, v, omega, _omega_x, _omega_y = _velocity_and_vorticity_derivatives(
        omega_hat=omega_hat,
        kx=kx,
        ky=ky,
        k_squared=k_squared,
        params=params,
        dealias=dealias,
    )
    exact = exact_solution(x, y, t=config.t_end, params=params)
    energy = kinetic_energy(u, v, params)
    exact_energy = kinetic_energy(exact.u, exact.v, params)
    relative_energy_error = abs(energy - exact_energy) / max(abs(exact_energy), 1e-12)

    state_bytes = omega_hat.nbytes + kx.nbytes + ky.nbytes + k_squared.nbytes + dealias.nbytes

    return SimulationResult(
        reynolds=params.reynolds,
        nx=nx,
        ny=ny,
        dt=dt,
        steps=steps,
        runtime_seconds=runtime_seconds,
        state_bytes=state_bytes,
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
        effective_state_bytes=state_bytes,
        compression_ratio=1.0,
    )
