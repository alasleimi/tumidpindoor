"""Leakage-safe multimodal receiver for the four Majid replacement protocols.

The geometric image sources are latent propagation state.  Estimators never see
their coordinates, reflection order, or source index.  They receive only peaks
extracted from one noisy array CIR materialized at a frozen acquisition point.
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
    """Observable receiver outputs; no simulator ray identity is retained."""

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
    """Eight half-wavelength-spaced elements on a 2x2x2 cube."""

    wavelength = C_M_S / float(carrier_hz)
    half = wavelength / 2.0
    return np.asarray(
        [[ix * half, iy * half, iz * half] for ix in (0, 1) for iy in (0, 1) for iz in (0, 1)],
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
    """Accept the legacy ``Source`` dataclass without importing its module."""

    return np.asarray(source.xyz_m, dtype=np.float64), int(source.tx_id), int(source.order)


def _noise_variance(clean_cir: np.ndarray, snr_db: float) -> float:
    energy = float(np.sum(np.abs(clean_cir) ** 2))
    return max(energy / (clean_cir.size * 10.0 ** (float(snr_db) / 10.0)), 1.0e-16)


def _estimate_direction(snapshot: np.ndarray) -> np.ndarray:
    """Estimate a 3-D direction from adjacent half-wave array phase ratios."""

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
    """Simulate and extract one noisy multimodal measurement.

    A complex array CIR is formed first.  Complex AWGN is added to every
    antenna/bin sample.  Delay, observed power, and direction are then derived
    from detected CIR peaks.  This intentionally makes the auxiliary channels
    noisy and coupled instead of exposing deterministic simulator attributes.
    """

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
        ripple = 1.5 * math.sin(0.73 * source_index + 0.31 * position[0] - 0.17 * position[1])
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
        noise_estimate = max(float(np.median(power)) / math.log(2.0), variance, 1.0e-16)
        threshold = noise_estimate * 10.0 ** (float(detection_threshold_db) / 10.0)
        peaks, properties = find_peaks(power, height=threshold)
        # scipy excludes the endpoints; add them if they are genuine local maxima.
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
            extracted.append((measured_range, observed_power, _estimate_direction(observed[:, index]), group))

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

    # The receiver exposes strongest observed peaks, not latent path order.
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


def corrupt_stored_fingerprint(
    fingerprint: MultimodalFingerprint,
    *,
    rng: np.random.Generator,
    extra_range_std_m: float,
    extra_power_std_db: float,
    extra_angle_std_deg: float,
    dropout_probability: float,
) -> MultimodalFingerprint:
    """Measurement-only augmentation; never calls the propagation simulator."""

    count = len(fingerprint)
    if count == 0:
        return fingerprint
    keep = rng.random(count) >= float(dropout_probability)
    ranges = fingerprint.ranges_m + rng.normal(0.0, float(extra_range_std_m), count)
    powers = fingerprint.powers_db + rng.normal(0.0, float(extra_power_std_db), count)
    angles = np.deg2rad(float(extra_angle_std_deg))
    directions = []
    for direction in fingerprint.aoa_unit:
        trial = direction + rng.normal(0.0, angles, 3)
        norm = max(float(np.linalg.norm(trial)), 1.0e-9)
        directions.append(trial / norm)
    valid = keep & np.isfinite(ranges) & (ranges > 0.0)
    selected = np.flatnonzero(valid)
    selected = selected[np.argsort(-powers[selected], kind="stable")]
    return MultimodalFingerprint(
        ranges_m=ranges[selected],
        powers_db=powers[selected],
        aoa_unit=np.asarray(directions, dtype=np.float64)[selected],
        tx_ids=fingerprint.tx_ids[selected],
        cir=fingerprint.cir.copy(),
        noise_variance=float(fingerprint.noise_variance),
        range_bin_m=float(fingerprint.range_bin_m),
    )


def pack_paths(
    fingerprints: Sequence[MultimodalFingerprint],
    *,
    maximum_paths: int,
    range_scale_m: float,
    power_centre_db: float = -40.0,
    power_scale_db: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack noisy [range, power, AoA xyz] tokens for set networks."""

    values = np.zeros((len(fingerprints), maximum_paths, 5), dtype=np.float32)
    mask = np.zeros((len(fingerprints), maximum_paths), dtype=bool)
    for row, fingerprint in enumerate(fingerprints):
        count = min(maximum_paths, len(fingerprint))
        if count == 0:
            continue
        values[row, :count, 0] = fingerprint.ranges_m[:count] / max(float(range_scale_m), 1.0e-6)
        values[row, :count, 1] = np.clip(
            (fingerprint.powers_db[:count] - float(power_centre_db)) / max(float(power_scale_db), 1.0e-6),
            -4.0,
            4.0,
        )
        values[row, :count, 2:5] = fingerprint.aoa_unit[:count]
        mask[row, :count] = True
    return values, mask


def normalized_cir_magnitude(fingerprints: Sequence[MultimodalFingerprint]) -> np.ndarray:
    """Flatten unit-Frobenius noisy CIR magnitude for CAEZ-style models."""

    rows = []
    for fingerprint in fingerprints:
        value = np.abs(fingerprint.cir).astype(np.float32).reshape(-1)
        norm = max(float(np.linalg.norm(value)), 1.0e-8)
        rows.append(value / norm)
    return np.asarray(rows, dtype=np.float32)


def coherent_adp_distance(query: np.ndarray, reference: np.ndarray) -> float:
    """Coherent normalized array-delay-profile dissimilarity."""

    q = np.asarray(query, dtype=np.complex128).reshape(-1, query.shape[-1])
    r = np.asarray(reference, dtype=np.complex128).reshape(-1, reference.shape[-1])
    taps = min(q.shape[1], r.shape[1])
    q, r = q[:, :taps], r[:, :taps]
    numerator = np.abs(np.sum(np.conj(q) * r, axis=0)) ** 2
    denominator = np.sum(np.abs(q) ** 2, axis=0) * np.sum(np.abs(r) ** 2, axis=0)
    similarity = np.sum(numerator / np.maximum(denominator, 1.0e-12))
    active = np.sum((np.sum(np.abs(q) ** 2, axis=0) + np.sum(np.abs(r) ** 2, axis=0)) > 1.0e-12)
    return float(1.0 - similarity / max(int(active), 1))
