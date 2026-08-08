"""Immutable noisy receiver used by the corrected MPUrge VT release run.

This file is a vendored, minimal snapshot of the receiver implementation that
was audited on 2026-08-07.  It intentionally exposes only quantities measured
from one materialized noisy array-CIR acquisition: detected delay/range,
observed peak power, array-derived AoA, transmitter channel, and the noisy CIR.
Latent image-source coordinates, reflection order, and ray identifiers are
never returned to an estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from scipy.signal import find_peaks


C_M_S = 299_792_458.0


@dataclass(frozen=True)
class MultimodalFingerprint:
    ranges_m: np.ndarray
    powers_db: np.ndarray
    aoa_unit: np.ndarray
    tx_ids: np.ndarray
    cir: np.ndarray
    noise_variance: float
    range_bin_m: float

    def __len__(self) -> int:
        return int(len(self.ranges_m))


def cube_array_positions(carrier_hz: float = 60.0e9) -> np.ndarray:
    wavelength = C_M_S / float(carrier_hz)
    half = wavelength / 2.0
    return np.asarray(
        [[ix * half, iy * half, iz * half]
         for ix in (0, 1) for iy in (0, 1) for iz in (0, 1)],
        dtype=np.float64,
    )


def _inside_quad(point: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    v0, v1, v2 = c - a, b - a, point - a
    dot00, dot01, dot02 = np.dot(v0, v0), np.dot(v0, v1), np.dot(v0, v2)
    dot11, dot12 = np.dot(v1, v1), np.dot(v1, v2)
    denominator = dot00 * dot11 - dot01 * dot01
    if abs(float(denominator)) < 1.0e-12:
        return False
    u = (dot11 * dot02 - dot01 * dot12) / denominator
    v = (dot00 * dot12 - dot01 * dot02) / denominator
    return bool(u >= -1.0e-9 and v >= -1.0e-9 and u + v <= 1.0 + 1.0e-9)


def segment_intersects_quad(start: np.ndarray, end: np.ndarray, quad: np.ndarray) -> bool:
    quad = np.asarray(quad, dtype=np.float64)
    normal = np.cross(quad[1] - quad[0], quad[2] - quad[0])
    denominator = float(np.dot(normal, end - start))
    if abs(denominator) < 1.0e-12:
        return False
    t = float(np.dot(normal, quad[0] - start) / denominator)
    if not (1.0e-7 < t < 1.0 - 1.0e-7):
        return False
    point = start + t * (end - start)
    return _inside_quad(point, quad[0], quad[1], quad[2]) or _inside_quad(
        point, quad[0], quad[2], quad[3]
    )


def _source_fields(source) -> tuple[np.ndarray, int, int]:
    return np.asarray(source.xyz_m, dtype=np.float64), int(source.tx_id), int(source.order)


def _noise_variance(clean_cir: np.ndarray, snr_db: float) -> float:
    energy = float(np.sum(np.abs(clean_cir) ** 2))
    return max(energy / (clean_cir.size * 10.0 ** (float(snr_db) / 10.0)), 1.0e-16)


def _estimate_direction(snapshot: np.ndarray) -> np.ndarray:
    cube = np.asarray(snapshot, dtype=np.complex128).reshape(2, 2, 2)
    estimates = []
    for axis in range(3):
        low = np.take(cube, 0, axis=axis).reshape(-1)
        high = np.take(cube, 1, axis=axis).reshape(-1)
        cross = np.sum(high * np.conj(low))
        estimates.append(float(np.clip(-np.angle(cross) / np.pi, -1.0, 1.0)))
    direction = np.asarray(estimates, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm < 1.0e-8:
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    return direction / norm


def _empty_fingerprint(n_bins: int, range_bin_m: float) -> MultimodalFingerprint:
    return MultimodalFingerprint(
        ranges_m=np.empty(0, dtype=np.float64),
        powers_db=np.empty(0, dtype=np.float64),
        aoa_unit=np.empty((0, 3), dtype=np.float64),
        tx_ids=np.empty(0, dtype=np.int64),
        cir=np.empty((0, 8, n_bins), dtype=np.complex64),
        noise_variance=0.0,
        range_bin_m=float(range_bin_m),
    )


def simulate_multimodal_fingerprint(
    position_xyz_m: Sequence[float],
    sources: Sequence[object],
    *,
    rng: np.random.Generator,
    maximum_paths: int = 9,
    snr_db: float = 20.0,
    range_noise_std_m: float = 0.0,
    bandwidth_hz: float = 2.0e9,
    carrier_hz: float = 60.0e9,
    n_bins: int = 256,
    detection_threshold_db: float = 6.0,
    obstructions: Sequence[np.ndarray] = (),
    extra_dropout_probability: float = 0.0,
    separate_transmitters: bool = True,
) -> MultimodalFingerprint:
    """Materialize one jointly noisy array-CIR observation and detect its MPCs."""

    if maximum_paths <= 0 or n_bins <= 2:
        raise ValueError("maximum_paths and n_bins must be positive")
    position = np.asarray(position_xyz_m, dtype=np.float64)
    array_positions = cube_array_positions(carrier_hz)
    wavelength = C_M_S / float(carrier_hz)
    range_bin_m = C_M_S / float(bandwidth_hz)

    by_tx: dict[int, list[tuple[np.ndarray, int]]] = {}
    for source_index, source in enumerate(sources):
        source_xyz, tx_id, order = _source_fields(source)
        if any(segment_intersects_quad(position, source_xyz, quad) for quad in obstructions):
            continue
        vector = source_xyz - position
        distance = float(np.linalg.norm(vector))
        if not np.isfinite(distance) or distance <= 0.0:
            continue
        direction = vector / distance
        ripple = 1.5 * math.sin(
            0.73 * source_index + 0.31 * position[0] - 0.17 * position[1]
        )
        power_db = -20.0 * math.log10(max(distance, 0.05)) - 7.0 * order + ripple
        amplitude = 10.0 ** (power_db / 20.0)
        phase = -2.0 * np.pi * distance / wavelength
        steering = np.exp(-2j * np.pi * (array_positions @ direction) / wavelength)
        snapshot = amplitude * np.exp(1j * phase) * steering
        group = tx_id if separate_transmitters else 0
        by_tx.setdefault(group, []).append((snapshot, int(np.rint(distance / range_bin_m))))

    if not by_tx:
        return _empty_fingerprint(n_bins, range_bin_m)

    cir_blocks: list[np.ndarray] = []
    extracted: list[tuple[float, float, np.ndarray, int]] = []
    total_noise = []
    delay_floor = 0.15 * 10.0 ** (-float(snr_db) / 20.0)
    for group, paths in sorted(by_tx.items()):
        clean = np.zeros((8, n_bins), dtype=np.complex128)
        for snapshot, index in paths:
            if 0 <= index < n_bins:
                clean[:, index] += snapshot
        variance = _noise_variance(clean, snr_db)
        total_noise.append(variance)
        noise = math.sqrt(variance / 2.0) * (
            rng.normal(size=clean.shape) + 1j * rng.normal(size=clean.shape)
        )
        observed = clean + noise
        cir_blocks.append(observed.astype(np.complex64))
        power = np.mean(np.abs(observed) ** 2, axis=0)
        noise_estimate = max(
            float(np.median(power)) / math.log(2.0), variance, 1.0e-16
        )
        threshold = noise_estimate * 10.0 ** (float(detection_threshold_db) / 10.0)
        peaks, _ = find_peaks(power, height=threshold)
        endpoints = []
        if power[0] >= threshold and power[0] > power[1]:
            endpoints.append(0)
        if power[-1] >= threshold and power[-1] > power[-2]:
            endpoints.append(n_bins - 1)
        if endpoints:
            peaks = np.unique(np.r_[peaks, endpoints]).astype(np.int64)
        if not len(peaks):
            continue
        order = peaks[np.argsort(-power[peaks], kind="stable")[:maximum_paths]]
        for index in order:
            if extra_dropout_probability > 0.0 and rng.random() < extra_dropout_probability:
                continue
            measured_range = index * range_bin_m + rng.normal(
                0.0, math.hypot(float(range_noise_std_m), delay_floor)
            )
            if not np.isfinite(measured_range) or measured_range <= 0.0:
                continue
            observed_power = 10.0 * math.log10(max(float(power[index]), 1.0e-20))
            extracted.append(
                (measured_range, observed_power, _estimate_direction(observed[:, index]), group)
            )

    if not extracted:
        return MultimodalFingerprint(
            ranges_m=np.empty(0, dtype=np.float64),
            powers_db=np.empty(0, dtype=np.float64),
            aoa_unit=np.empty((0, 3), dtype=np.float64),
            tx_ids=np.empty(0, dtype=np.int64),
            cir=np.stack(cir_blocks).astype(np.complex64),
            noise_variance=float(np.mean(total_noise)),
            range_bin_m=float(range_bin_m),
        )

    extracted.sort(key=lambda row: -row[1])
    extracted = extracted[:maximum_paths]
    return MultimodalFingerprint(
        ranges_m=np.asarray([row[0] for row in extracted], dtype=np.float64),
        powers_db=np.asarray([row[1] for row in extracted], dtype=np.float64),
        aoa_unit=np.asarray([row[2] for row in extracted], dtype=np.float64),
        tx_ids=np.asarray([row[3] for row in extracted], dtype=np.int64),
        cir=np.stack(cir_blocks).astype(np.complex64),
        noise_variance=float(np.mean(total_noise)),
        range_bin_m=float(range_bin_m),
    )
