"""Leakage-safe multimodal observations for the four reconstructed protocols.

The rectangular image-source simulator owns latent ray/source information.  This
module converts those latent paths into one noisy receiver acquisition and
exports only sensor-like quantities: range, received power/complex gain, a
noisy global arrival direction, transmitter channel, and noisy array CIR.  It
never exports image-source coordinates, reflection order, or ray identity.

The model is deliberately explicit rather than presented as the unavailable
NIST Q-D receiver.  Its purpose is a fair, reproducible feature-rich extension:
all methods use the same sparse acquisitions, and no physical feature remains a
clean side channel when the delay is noisy.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from experiments.paper_protocol_replications.common import Source, segment_intersects_quad


SPEED_OF_LIGHT_M_S = 299_792_458.0


@dataclass(frozen=True)
class MultimodalObservation:
    """One materialized receiver acquisition.

    Rows of the path arrays describe jointly extracted MPC tokens.  Their order
    is observed-power order, not latent ray order.  `cir` is antenna by delay
    bin and contains receiver noise in bins without an extracted path as well.
    """

    ranges_m: np.ndarray
    powers_db: np.ndarray
    aoa_unit: np.ndarray
    tx_ids: np.ndarray
    complex_gains: np.ndarray
    path_snr_db: np.ndarray
    cir: np.ndarray
    range_bin_m: float
    sensor_snr_db: float

    def __len__(self) -> int:
        return int(self.ranges_m.shape[0])

    def validate(self) -> None:
        count = len(self)
        if self.powers_db.shape != (count,):
            raise ValueError("power/path length mismatch")
        if self.aoa_unit.shape != (count, 3):
            raise ValueError("AoA must have shape [paths,3]")
        if self.tx_ids.shape != (count,):
            raise ValueError("transmitter/path length mismatch")
        if self.complex_gains.shape != (count,):
            raise ValueError("gain/path length mismatch")
        if self.path_snr_db.shape != (count,):
            raise ValueError("SNR/path length mismatch")
        if self.cir.ndim != 2 or not np.iscomplexobj(self.cir):
            raise ValueError("CIR must be a complex [antenna,tap] matrix")
        if count:
            norms = np.linalg.norm(self.aoa_unit, axis=1)
            if not np.allclose(norms, 1.0, atol=2.0e-5):
                raise ValueError("AoA rows must be unit vectors")
            if not np.all(np.isfinite(self.ranges_m)) or not np.all(self.ranges_m > 0.0):
                raise ValueError("ranges must be finite and positive")
            if np.any(np.diff(self.powers_db) > 1.0e-8):
                raise ValueError("paths must be in descending observed-power order")


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-12:
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    return np.asarray(vector, dtype=np.float64) / norm


def _perturb_direction(
    direction: np.ndarray,
    standard_deviation_rad: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Perturb a unit direction in its tangent plane and renormalize."""

    direction = _unit(direction)
    helper = np.asarray([0.0, 0.0, 1.0])
    if abs(float(np.dot(direction, helper))) > 0.9:
        helper = np.asarray([0.0, 1.0, 0.0])
    tangent_1 = _unit(np.cross(direction, helper))
    tangent_2 = _unit(np.cross(direction, tangent_1))
    offset = rng.normal(0.0, standard_deviation_rad, 2)
    return _unit(direction + offset[0] * tangent_1 + offset[1] * tangent_2)


def _random_direction(rng: np.random.Generator) -> np.ndarray:
    value = rng.normal(size=3)
    return _unit(value)


def _latent_paths(
    position_xyz_m: Sequence[float],
    sources: Sequence[Source],
    obstructions: Sequence[np.ndarray],
    carrier_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return private latent ranges, powers, Tx IDs, directions, and gains."""

    position = np.asarray(position_xyz_m, dtype=np.float64)
    rows: list[tuple[float, float, int, np.ndarray, complex]] = []
    wavelength = SPEED_OF_LIGHT_M_S / float(carrier_hz)
    for source_index, source in enumerate(sources):
        if any(segment_intersects_quad(position, source.xyz_m, quad) for quad in obstructions):
            continue
        difference = np.asarray(source.xyz_m, dtype=np.float64) - position
        distance = float(np.linalg.norm(difference))
        ripple_db = 1.5 * math.sin(
            0.73 * source_index + 0.31 * position[0] - 0.17 * position[1]
        )
        power_db = -20.0 * math.log10(max(distance, 0.05)) - 7.0 * source.order + ripple_db
        amplitude = 10.0 ** (power_db / 20.0)
        phase = -2.0 * math.pi * distance / wavelength + 0.19 * source_index
        gain = amplitude * np.exp(1j * phase)
        rows.append((distance, power_db, int(source.tx_id), _unit(difference), complex(gain)))
    if not rows:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int64),
            np.empty((0, 3), dtype=np.float64),
            np.empty(0, dtype=np.complex128),
        )
    return (
        np.asarray([row[0] for row in rows], dtype=np.float64),
        np.asarray([row[1] for row in rows], dtype=np.float64),
        np.asarray([row[2] for row in rows], dtype=np.int64),
        np.asarray([row[3] for row in rows], dtype=np.float64),
        np.asarray([row[4] for row in rows], dtype=np.complex128),
    )


def simulate_multimodal_observation(
    position_xyz_m: Sequence[float],
    sources: Sequence[Source],
    *,
    rng: np.random.Generator,
    maximum_paths: int = 9,
    carrier_hz: float = 60.0e9,
    bandwidth_hz: float = 2.0e9,
    sensor_snr_db: float = 20.0,
    range_awgn_std_m: float = 0.0,
    base_aoa_std_deg: float = 4.0,
    detection_threshold_db: float = 1.5,
    false_alarm_rate: float = 0.05,
    maximum_range_m: float = 80.0,
    antennas: int = 8,
    cir_bins: int = 512,
    obstructions: Sequence[np.ndarray] = (),
) -> MultimodalObservation:
    """Create one jointly noisy path/CIR acquisition.

    `sensor_snr_db` is total mean-path SNR.  Weak paths consequently have lower
    individual SNR and larger delay/direction uncertainty.  The range CRLB-like
    term is combined in quadrature with the paper's requested range AWGN.
    """

    ranges, _, tx_ids, directions, clean_gains = _latent_paths(
        position_xyz_m, sources, obstructions, carrier_hz
    )
    snr_linear = 10.0 ** (float(sensor_snr_db) / 10.0)
    if len(clean_gains):
        mean_path_energy = float(np.mean(np.abs(clean_gains) ** 2))
        noise_variance = max(mean_path_energy / snr_linear, 1.0e-18)
    else:
        noise_variance = 1.0e-12
    path_noise = np.sqrt(noise_variance / 2.0) * (
        rng.normal(size=len(clean_gains)) + 1j * rng.normal(size=len(clean_gains))
    )
    observed_gains = clean_gains + path_noise
    individual_snr = np.abs(clean_gains) ** 2 / noise_variance
    observed_snr = np.abs(observed_gains) ** 2 / noise_variance
    keep = 10.0 * np.log10(np.maximum(observed_snr, 1.0e-12)) >= detection_threshold_db

    kept_ranges: list[float] = []
    kept_gains: list[complex] = []
    kept_directions: list[np.ndarray] = []
    kept_tx: list[int] = []
    kept_snr: list[float] = []
    nominal_snr = snr_linear
    for index in np.flatnonzero(keep):
        path_snr = max(float(individual_snr[index]), 1.0e-6)
        receiver_sigma = SPEED_OF_LIGHT_M_S / (
            2.0 * float(bandwidth_hz) * math.sqrt(2.0 * path_snr)
        )
        range_sigma = math.sqrt(receiver_sigma**2 + float(range_awgn_std_m) ** 2)
        observed_range = float(ranges[index] + rng.normal(0.0, range_sigma))
        if not np.isfinite(observed_range) or observed_range <= 0.0:
            continue
        angle_sigma_deg = float(base_aoa_std_deg) * math.sqrt(nominal_snr / path_snr)
        angle_sigma_deg = float(np.clip(angle_sigma_deg, 0.25, 35.0))
        kept_ranges.append(observed_range)
        kept_gains.append(complex(observed_gains[index]))
        kept_directions.append(
            _perturb_direction(directions[index], math.radians(angle_sigma_deg), rng)
        )
        kept_tx.append(int(tx_ids[index]))
        kept_snr.append(10.0 * math.log10(max(float(observed_snr[index]), 1.0e-12)))

    # False alarms are complete noisy MPC tokens, not range-only ghosts.
    false_count = int(rng.poisson(max(float(false_alarm_rate), 0.0)))
    transmitter_count = max((source.tx_id for source in sources), default=0) + 1
    for _ in range(false_count):
        kept_ranges.append(float(rng.uniform(0.1, maximum_range_m)))
        false_gain = np.sqrt(noise_variance / 2.0) * (
            rng.normal() + 1j * rng.normal()
        )
        kept_gains.append(complex(false_gain))
        kept_directions.append(_random_direction(rng))
        kept_tx.append(int(rng.integers(0, max(transmitter_count, 1))))
        kept_snr.append(10.0 * math.log10(max(abs(false_gain) ** 2 / noise_variance, 1.0e-12)))

    if kept_ranges:
        ranges_out = np.asarray(kept_ranges, dtype=np.float64)
        gains_out = np.asarray(kept_gains, dtype=np.complex128)
        directions_out = np.asarray(kept_directions, dtype=np.float64)
        tx_out = np.asarray(kept_tx, dtype=np.int64)
        snr_out = np.asarray(kept_snr, dtype=np.float64)
        powers_out = 10.0 * np.log10(np.maximum(np.abs(gains_out) ** 2, 1.0e-20))
        order = np.argsort(-powers_out, kind="stable")[:maximum_paths]
        ranges_out, gains_out = ranges_out[order], gains_out[order]
        directions_out, tx_out = directions_out[order], tx_out[order]
        snr_out, powers_out = snr_out[order], powers_out[order]
    else:
        ranges_out = np.empty(0, dtype=np.float64)
        gains_out = np.empty(0, dtype=np.complex128)
        directions_out = np.empty((0, 3), dtype=np.float64)
        tx_out = np.empty(0, dtype=np.int64)
        snr_out = np.empty(0, dtype=np.float64)
        powers_out = np.empty(0, dtype=np.float64)

    # Deposit the same observed gains/directions into a fixed global ULA CIR,
    # then add background receiver noise.  This is what coherent methods see.
    range_bin_m = SPEED_OF_LIGHT_M_S / float(bandwidth_hz)
    cir = np.sqrt(noise_variance / 2.0) * (
        rng.normal(size=(antennas, cir_bins))
        + 1j * rng.normal(size=(antennas, cir_bins))
    )
    antenna_index = np.arange(antennas, dtype=np.float64)
    for measured_range, gain, direction in zip(
        ranges_out, gains_out, directions_out, strict=True
    ):
        delay_bin = int(np.rint(measured_range / range_bin_m))
        if not 0 <= delay_bin < cir_bins:
            continue
        # Half-wavelength ULA aligned to global x.
        steering = np.exp(-1j * math.pi * antenna_index * float(direction[0]))
        cir[:, delay_bin] += gain * steering
    cir = cir.astype(np.complex64)

    observation = MultimodalObservation(
        ranges_m=ranges_out,
        powers_db=powers_out,
        aoa_unit=directions_out,
        tx_ids=tx_out,
        complex_gains=gains_out.astype(np.complex64),
        path_snr_db=snr_out,
        cir=cir,
        range_bin_m=float(range_bin_m),
        sensor_snr_db=float(sensor_snr_db),
    )
    observation.validate()
    return observation


def path_feature_tensor(
    observations: Sequence[MultimodalObservation],
    *,
    maximum_paths: int,
    range_scale_m: float,
    maximum_tx: int,
    fields: tuple[str, ...] = ("range", "power", "aoa", "complex", "snr", "tx"),
) -> tuple[np.ndarray, np.ndarray]:
    """Pack selected observable fields without exposing latent identifiers."""

    widths = {
        "range": 1,
        "power": 1,
        "aoa": 3,
        "complex": 2,
        "snr": 1,
        "tx": max(int(maximum_tx), 1),
    }
    unknown = set(fields) - set(widths)
    if unknown:
        raise ValueError(f"unknown observable fields: {sorted(unknown)}")
    feature_width = sum(widths[field] for field in fields)
    values = np.zeros((len(observations), maximum_paths, feature_width), dtype=np.float32)
    mask = np.zeros((len(observations), maximum_paths), dtype=bool)
    for row, observation in enumerate(observations):
        count = min(maximum_paths, len(observation))
        cursor = 0
        for field in fields:
            width = widths[field]
            if count:
                if field == "range":
                    block = observation.ranges_m[:count, None] / max(float(range_scale_m), 1.0e-6)
                elif field == "power":
                    # Per-acquisition centring removes arbitrary absolute gain
                    # while retaining physically observed relative power.
                    power = observation.powers_db[:count]
                    block = ((power - np.max(power)) / 20.0)[:, None]
                elif field == "aoa":
                    block = observation.aoa_unit[:count]
                elif field == "complex":
                    gain = observation.complex_gains[:count]
                    scale = max(float(np.sqrt(np.sum(np.abs(gain) ** 2))), 1.0e-9)
                    block = np.column_stack((gain.real / scale, gain.imag / scale))
                elif field == "snr":
                    block = np.clip(observation.path_snr_db[:count, None] / 30.0, -1.0, 2.0)
                elif field == "tx":
                    block = np.zeros((count, width), dtype=np.float64)
                    ids = np.clip(observation.tx_ids[:count], 0, width - 1)
                    block[np.arange(count), ids] = 1.0
                else:  # pragma: no cover - guarded above
                    raise AssertionError(field)
                values[row, :count, cursor : cursor + width] = block
            cursor += width
        mask[row, :count] = True
    return values, mask


def normalized_cir_tensor(observations: Sequence[MultimodalObservation]) -> np.ndarray:
    """Return real/imag CIR channels with per-acquisition Frobenius scaling."""

    output = []
    for observation in observations:
        cir = np.asarray(observation.cir, dtype=np.complex64)
        scale = max(float(np.linalg.norm(cir)), 1.0e-9)
        output.append(np.stack((cir.real / scale, cir.imag / scale), axis=0))
    return np.asarray(output, dtype=np.float32)
