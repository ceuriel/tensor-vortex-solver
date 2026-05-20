from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np


@dataclass(frozen=True)
class CompressionConfig:
    target_rank: int | None = 4
    energy_capture: float | None = 0.9999
    compress_every: int = 1
    min_rank: int = 1
    method: str = "tensor_train"
    min_benefit_ratio: float = 1.05
    compress_operators: bool = True
    operator_method: str = "matrix_power"
    operator_target_rank: int | None = 2
    operator_energy_capture: float | None = 0.999
    operator_compress_every: int = 3
    operator_min_benefit_ratio: float = 1.02
    randomized_oversampling: int = 1
    randomized_power_iterations: int = 0


@dataclass(frozen=True)
class MatrixLowRankField:
    left: np.ndarray
    singular_values: np.ndarray
    right_t: np.ndarray
    dense_shape: tuple[int, int]
    method: str = "matrix_svd"

    @property
    def compressed_bytes(self) -> int:
        return self.left.nbytes + self.singular_values.nbytes + self.right_t.nbytes

    @property
    def max_rank(self) -> int:
        return int(self.singular_values.shape[0])

    @property
    def mean_rank(self) -> float:
        return float(self.max_rank)

    def to_dense(self) -> np.ndarray:
        return (self.left * self.singular_values) @ self.right_t


@dataclass(frozen=True)
class TensorTrainField:
    cores: tuple[np.ndarray, ...]
    dense_shape: tuple[int, int]
    x_factors: tuple[int, int]
    y_factors: tuple[int, int]
    mode_shape: tuple[int, ...]
    internal_ranks: tuple[int, ...]
    method: str = "tensor_train"

    @property
    def compressed_bytes(self) -> int:
        return int(sum(core.nbytes for core in self.cores))

    @property
    def max_rank(self) -> int:
        if not self.internal_ranks:
            return 1
        return int(max(self.internal_ranks))

    @property
    def mean_rank(self) -> float:
        if not self.internal_ranks:
            return 1.0
        return float(np.mean(self.internal_ranks))

    def to_dense(self) -> np.ndarray:
        tensor = self.cores[0]
        for core in self.cores[1:]:
            tensor = np.tensordot(tensor, core, axes=([-1], [0]))

        blocked = np.squeeze(tensor, axis=(0, -1))
        restored = blocked.transpose(0, 2, 1, 3)
        return restored.reshape(self.dense_shape)


CompressedField = MatrixLowRankField | TensorTrainField


@dataclass(frozen=True)
class CompressionResult:
    compressed: CompressedField
    approximation: np.ndarray
    max_rank: int
    mean_rank: float
    compressed_bytes: int
    relative_error: float


def _rank_from_energy(singular_values: np.ndarray, energy_capture: float) -> int:
    total_energy = float(np.sum(singular_values**2))
    if total_energy <= 0.0:
        return 1
    cumulative = np.cumsum(singular_values**2) / total_energy
    return int(np.searchsorted(cumulative, energy_capture, side="left") + 1)


def _truncate_rank(singular_values: np.ndarray, config: CompressionConfig) -> int:
    max_rank = len(singular_values)
    candidates = [max_rank]
    if config.energy_capture is not None:
        candidates.append(_rank_from_energy(singular_values, config.energy_capture))
    if config.target_rank is not None:
        candidates.append(config.target_rank)

    rank = max(config.min_rank, min(candidates))
    return min(rank, max_rank)


@lru_cache(maxsize=None)
def _balanced_factors(n: int) -> tuple[int, int]:
    root = int(math.sqrt(n))
    for factor in range(root, 0, -1):
        if n % factor == 0:
            return factor, n // factor
    return 1, n


@lru_cache(maxsize=None)
def _tensor_train_mode_shape(shape: tuple[int, int]) -> tuple[int, int, int, int]:
    nx, ny = shape
    x_factors = _balanced_factors(nx)
    y_factors = _balanced_factors(ny)
    return (x_factors[0], y_factors[0], x_factors[1], y_factors[1])


def dense_field_bytes(shape: tuple[int, int], dtype: np.dtype = np.dtype(float)) -> int:
    return int(np.prod(shape) * dtype.itemsize)


def estimate_compressed_bytes(shape: tuple[int, int], config: CompressionConfig) -> int | None:
    normalized_method = config.method.strip().lower().replace("-", "_")
    if config.target_rank is None:
        return None

    dense_bytes = dense_field_bytes(shape)
    if normalized_method == "matrix_svd":
        max_rank = min(shape)
        rank = max(config.min_rank, min(config.target_rank, max_rank))
        element_count = shape[0] * rank + rank + rank * shape[1]
        return int(element_count * np.dtype(float).itemsize)

    if normalized_method == "tensor_train":
        mode_shape = _tensor_train_mode_shape(shape)
        rank = max(config.min_rank, config.target_rank)
        rank = min(rank, max(mode_shape))
        tt_elements = mode_shape[0] * rank
        for mode_size in mode_shape[1:-1]:
            tt_elements += rank * mode_size * rank
        tt_elements += rank * mode_shape[-1]
        estimated = int(tt_elements * np.dtype(float).itemsize)
        return min(estimated, dense_bytes)

    return None


def compression_would_help(
    shape: tuple[int, int],
    config: CompressionConfig,
    dense_bytes: int | None = None,
) -> bool:
    dense_bytes = dense_field_bytes(shape) if dense_bytes is None else dense_bytes
    estimated_bytes = estimate_compressed_bytes(shape, config)
    if estimated_bytes is None:
        return True
    return dense_bytes / max(estimated_bytes, 1) >= config.min_benefit_ratio


def operator_projection_config(config: CompressionConfig) -> CompressionConfig:
    return CompressionConfig(
        target_rank=config.operator_target_rank,
        energy_capture=config.operator_energy_capture,
        compress_every=config.operator_compress_every,
        min_rank=config.min_rank,
        method=config.operator_method,
        min_benefit_ratio=config.operator_min_benefit_ratio,
        compress_operators=False,
        operator_method=config.operator_method,
        operator_target_rank=config.operator_target_rank,
        operator_energy_capture=config.operator_energy_capture,
        operator_compress_every=config.operator_compress_every,
        operator_min_benefit_ratio=config.operator_min_benefit_ratio,
    )


def _compress_matrix_field(field: np.ndarray, config: CompressionConfig) -> CompressionResult:
    left, singular_values, right_t = _truncated_svd(field, config)
    rank = _truncate_rank(singular_values, config)

    compressed = MatrixLowRankField(
        left=left[:, :rank],
        singular_values=singular_values[:rank],
        right_t=right_t[:rank, :],
        dense_shape=field.shape,
    )
    approximation = compressed.to_dense()
    denominator = max(np.linalg.norm(field), 1e-12)
    relative_error = float(np.linalg.norm(approximation - field) / denominator)

    return CompressionResult(
        compressed=compressed,
        approximation=approximation,
        max_rank=compressed.max_rank,
        mean_rank=compressed.mean_rank,
        compressed_bytes=compressed.compressed_bytes,
        relative_error=relative_error,
    )


def _compress_matrix_field_power(field: np.ndarray, config: CompressionConfig) -> CompressionResult:
    rank = config.target_rank if config.target_rank is not None else config.min_rank
    rank = max(config.min_rank, min(rank, min(field.shape)))

    residual = field.copy()
    left_vectors: list[np.ndarray] = []
    right_vectors: list[np.ndarray] = []
    singular_values: list[float] = []

    for mode in range(rank):
        right = residual[mode % residual.shape[0], :].copy()
        if np.linalg.norm(right) <= 1e-12:
            right = np.mean(residual, axis=0)
        if np.linalg.norm(right) <= 1e-12:
            break
        right = right / max(np.linalg.norm(right), 1e-12)

        for _ in range(2):
            left = residual @ right
            left_norm = np.linalg.norm(left)
            if left_norm <= 1e-12:
                break
            left = left / left_norm

            right = residual.T @ left
            right_norm = np.linalg.norm(right)
            if right_norm <= 1e-12:
                break
            right = right / right_norm
        else:
            left = residual @ right
            singular_value = float(np.linalg.norm(left))
            if singular_value <= 1e-12:
                continue
            left = left / singular_value

            left_vectors.append(left)
            right_vectors.append(right)
            singular_values.append(singular_value)
            residual = residual - singular_value * np.outer(left, right)
            continue

        break

    if not singular_values:
        return _compress_matrix_field(field, config)

    left = np.column_stack(left_vectors)
    singular_values_array = np.asarray(singular_values)
    right_t = np.vstack(right_vectors)
    compressed = MatrixLowRankField(
        left=left,
        singular_values=singular_values_array,
        right_t=right_t,
        dense_shape=field.shape,
        method="matrix_power",
    )
    approximation = compressed.to_dense()
    denominator = max(np.linalg.norm(field), 1e-12)
    relative_error = float(np.linalg.norm(approximation - field) / denominator)

    return CompressionResult(
        compressed=compressed,
        approximation=approximation,
        max_rank=compressed.max_rank,
        mean_rank=compressed.mean_rank,
        compressed_bytes=compressed.compressed_bytes,
        relative_error=relative_error,
    )


def _reshape_field_for_tensor_train(
    field: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int], tuple[int, ...]]:
    nx, ny = field.shape
    x_factors = _balanced_factors(nx)
    y_factors = _balanced_factors(ny)

    blocked = field.reshape(x_factors[0], x_factors[1], y_factors[0], y_factors[1]).transpose(0, 2, 1, 3)
    return blocked, x_factors, y_factors, blocked.shape


def _should_use_randomized_svd(matrix: np.ndarray, config: CompressionConfig) -> bool:
    if config.target_rank is None:
        return False
    rows, cols = matrix.shape
    max_rank = min(rows, cols)
    if max_rank <= 32:
        return False
    requested_rank = max(config.min_rank, min(config.target_rank, max_rank))
    if requested_rank >= max_rank // 3:
        return False
    return True


def _randomized_range(
    matrix: np.ndarray,
    rank: int,
    oversampling: int,
    power_iterations: int,
) -> np.ndarray:
    rows, cols = matrix.shape
    sample_rank = min(max(rank + oversampling, rank), min(rows, cols))
    seed = (
        rows * 73856093
        + cols * 19349663
        + rank * 83492791
        + oversampling * 2654435761
        + power_iterations * 97531
    ) % (2**32)
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((cols, sample_rank))
    sample = matrix @ omega
    for _ in range(max(power_iterations, 0)):
        sample = matrix @ (matrix.T @ sample)
    q, _r = np.linalg.qr(sample, mode="reduced")
    return q


def _truncated_svd(
    matrix: np.ndarray,
    config: CompressionConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not _should_use_randomized_svd(matrix, config):
        return np.linalg.svd(matrix, full_matrices=False)

    max_rank = min(matrix.shape)
    requested_rank = max(config.min_rank, min(config.target_rank or max_rank, max_rank))
    q = _randomized_range(
        matrix,
        rank=requested_rank,
        oversampling=config.randomized_oversampling,
        power_iterations=config.randomized_power_iterations,
    )
    reduced = q.T @ matrix
    left_small, singular_values, right_t = np.linalg.svd(reduced, full_matrices=False)
    left = q @ left_small
    return left, singular_values, right_t


def _tensor_train_svd(
    tensor: np.ndarray,
    config: CompressionConfig,
) -> tuple[tuple[np.ndarray, ...], tuple[int, ...]]:
    mode_shape = tensor.shape
    residual = tensor.copy()
    left_rank = 1
    cores: list[np.ndarray] = []
    internal_ranks: list[int] = []

    for mode_size in mode_shape[:-1]:
        residual = residual.reshape(left_rank * mode_size, -1)
        left, singular_values, right_t = _truncated_svd(residual, config)
        rank = _truncate_rank(singular_values, config)
        cores.append(left[:, :rank].reshape(left_rank, mode_size, rank))
        residual = singular_values[:rank, None] * right_t[:rank, :]
        internal_ranks.append(rank)
        left_rank = rank

    cores.append(residual.reshape(left_rank, mode_shape[-1], 1))
    return tuple(cores), tuple(internal_ranks)


def _compress_tensor_train_field(field: np.ndarray, config: CompressionConfig) -> CompressionResult:
    tensor, x_factors, y_factors, mode_shape = _reshape_field_for_tensor_train(field)
    cores, internal_ranks = _tensor_train_svd(tensor, config)

    compressed = TensorTrainField(
        cores=cores,
        dense_shape=field.shape,
        x_factors=x_factors,
        y_factors=y_factors,
        mode_shape=mode_shape,
        internal_ranks=internal_ranks,
    )
    approximation = compressed.to_dense()
    denominator = max(np.linalg.norm(field), 1e-12)
    relative_error = float(np.linalg.norm(approximation - field) / denominator)

    return CompressionResult(
        compressed=compressed,
        approximation=approximation,
        max_rank=compressed.max_rank,
        mean_rank=compressed.mean_rank,
        compressed_bytes=compressed.compressed_bytes,
        relative_error=relative_error,
    )


def compress_field(field: np.ndarray, config: CompressionConfig) -> CompressionResult:
    normalized_method = config.method.strip().lower().replace("-", "_")
    if normalized_method == "matrix_svd":
        return _compress_matrix_field(field, config)
    if normalized_method == "matrix_power":
        return _compress_matrix_field_power(field, config)
    if normalized_method == "tensor_train":
        return _compress_tensor_train_field(field, config)
    raise ValueError(f"Unsupported compression method '{config.method}'.")
