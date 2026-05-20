from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class TaylorGreenParameters:
    domain_length: float = 2.0 * math.pi
    vortex_velocity: float = 1.0
    convection_u: float = 1.0
    convection_v: float = 0.0
    density: float = 1.0
    reynolds: float = 100.0
    reference_pressure: float = 0.0

    @property
    def viscosity(self) -> float:
        return self.vortex_velocity * self.domain_length / self.reynolds

    @property
    def wavenumber(self) -> float:
        return 2.0 * math.pi / self.domain_length


@dataclass(frozen=True)
class ExactSolution:
    u: np.ndarray
    v: np.ndarray
    p: np.ndarray
    omega: np.ndarray


def create_grid(length: float, nx: int, ny: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    ny = nx if ny is None else ny
    x = np.linspace(0.0, length, nx, endpoint=False)
    y = np.linspace(0.0, length, ny, endpoint=False)
    return np.meshgrid(x, y, indexing="ij")


def exact_solution(
    x: np.ndarray,
    y: np.ndarray,
    t: float,
    params: TaylorGreenParameters,
) -> ExactSolution:
    k = params.wavenumber
    shifted_x = x - params.convection_u * t
    shifted_y = y - params.convection_v * t
    velocity_decay = math.exp(-2.0 * params.viscosity * (k**2) * t)
    pressure_decay = velocity_decay**2

    phase_x = k * shifted_x
    phase_y = k * shifted_y

    u_fluct = -params.vortex_velocity * np.cos(phase_x) * np.sin(phase_y) * velocity_decay
    v_fluct = params.vortex_velocity * np.sin(phase_x) * np.cos(phase_y) * velocity_decay

    u = params.convection_u + u_fluct
    v = params.convection_v + v_fluct
    p = params.reference_pressure - (
        params.density
        * (params.vortex_velocity**2)
        / 4.0
        * (np.cos(2.0 * phase_x) + np.cos(2.0 * phase_y))
        * pressure_decay
    )
    omega = 2.0 * params.vortex_velocity * k * np.cos(phase_x) * np.cos(phase_y) * velocity_decay
    return ExactSolution(u=u, v=v, p=p, omega=omega)


def kinetic_energy(u: np.ndarray, v: np.ndarray, params: TaylorGreenParameters) -> float:
    fluct_u = u - params.convection_u
    fluct_v = v - params.convection_v
    return float(0.5 * np.mean(fluct_u**2 + fluct_v**2))


def l2_velocity_error(
    numerical: tuple[np.ndarray, np.ndarray],
    analytical: tuple[np.ndarray, np.ndarray],
) -> float:
    num_u, num_v = numerical
    exact_u, exact_v = analytical
    diff = np.sqrt((num_u - exact_u) ** 2 + (num_v - exact_v) ** 2)
    ref = np.sqrt(exact_u**2 + exact_v**2)
    denom = np.sqrt(np.mean(ref**2))
    if denom == 0.0:
        return float(np.sqrt(np.mean(diff**2)))
    return float(np.sqrt(np.mean(diff**2)) / denom)


def recommended_resolution(reynolds: float, base_resolution: int = 32) -> int:
    scaled = base_resolution * math.sqrt(max(reynolds, 1.0) / 10.0)
    rounded = int(math.ceil(scaled / 8.0) * 8)
    return max(32, rounded)


def stable_timestep(
    params: TaylorGreenParameters,
    nx: int,
    ny: int | None = None,
    cfl: float = 0.15,
    diffusive_safety: float = 0.2,
) -> float:
    ny = nx if ny is None else ny
    dx = params.domain_length / nx
    dy = params.domain_length / ny
    cell_size = min(dx, dy)
    advective_speed = max(
        abs(params.convection_u) + params.vortex_velocity,
        abs(params.convection_v) + params.vortex_velocity,
        1e-12,
    )
    advective_dt = cfl * cell_size / advective_speed
    diffusive_dt = diffusive_safety * (cell_size**2) / max(params.viscosity, 1e-12)
    return min(advective_dt, diffusive_dt)
