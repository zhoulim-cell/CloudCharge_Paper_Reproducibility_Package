from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
import hashlib

import numpy as np

from .constants import BOLTZMANN, ELEMENTARY_CHARGE, EPSILON_0, PI
from .frontier_numerics import bernoulli, terminal_velocity_stokes


@dataclass
class ScavengingDiagnostics:
    aerosol_removed_m3: np.ndarray
    transferred_charge_e_m3: np.ndarray
    mean_electrostatic_multiplier: np.ndarray


def aerosol_diffusivity(radius_m: np.ndarray, temperature_k: float, viscosity_pa_s: float = 1.7e-5) -> np.ndarray:
    return BOLTZMANN * temperature_k / (6.0 * PI * viscosity_pa_s * np.maximum(radius_m, 1.0e-12))


# Kernels are independent of z and time for the current 1-D model.  Caching
# avoids rebuilding several-MB four-dimensional arrays at every model step.
_KERNEL_CACHE: "OrderedDict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]" = OrderedDict()
_KERNEL_CACHE_MAX = 4


def _array_digest(x: np.ndarray) -> bytes:
    a = np.ascontiguousarray(x, dtype=np.float64)
    h = hashlib.blake2b(digest_size=12)
    h.update(np.asarray(a.shape, dtype=np.int64).tobytes())
    h.update(a.tobytes())
    return h.digest()


def _kernel_key(
    aerosol_r_m: np.ndarray,
    aerosol_q_e: np.ndarray,
    droplet_r_m: np.ndarray,
    droplet_q_e: np.ndarray,
    temperature_k: float,
    include_brownian: bool,
    include_interception: bool,
    include_electrostatic: bool,
) -> str:
    h = hashlib.blake2b(digest_size=20)
    for x in (aerosol_r_m, aerosol_q_e, droplet_r_m, droplet_q_e):
        h.update(_array_digest(x))
    h.update(np.float64(temperature_k).tobytes())
    h.update(bytes((include_brownian, include_interception, include_electrostatic)))
    return h.hexdigest()


def pair_kernel_components(
    aerosol_r_m: np.ndarray,
    aerosol_q_e: np.ndarray,
    droplet_r_m: np.ndarray,
    droplet_q_e: np.ndarray,
    temperature_k: float,
    include_brownian: bool = True,
    include_interception: bool = True,
    include_electrostatic: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return charged kernel, neutral kernel, and electrostatic multiplier.

    The result is cached because radii, charge grids and temperature are fixed
    during a simulation.  Arrays have dimensions (aerosol radius, aerosol
    charge, droplet radius, droplet charge).
    """
    key = _kernel_key(
        aerosol_r_m, aerosol_q_e, droplet_r_m, droplet_q_e, temperature_k,
        include_brownian, include_interception, include_electrostatic,
    )
    cached = _KERNEL_CACHE.get(key)
    if cached is not None:
        _KERNEL_CACHE.move_to_end(key)
        return cached

    ar = aerosol_r_m[:, None, None, None]
    aq = aerosol_q_e[None, :, None, None]
    dr = droplet_r_m[None, None, :, None]
    dq = droplet_q_e[None, None, None, :]
    contact = ar + dr
    neutral = np.zeros(
        (aerosol_r_m.size, aerosol_q_e.size, droplet_r_m.size, droplet_q_e.size),
        dtype=np.float64,
    )
    if include_brownian:
        da = aerosol_diffusivity(aerosol_r_m, temperature_k)[:, None, None, None]
        neutral += 4.0 * PI * da * contact
    if include_interception:
        vt = terminal_velocity_stokes(droplet_r_m)[None, None, :, None]
        geom = PI * contact**2 * vt
        interception_efficiency = np.minimum(1.0, 2.0 * ar / np.maximum(dr, 1.0e-30))
        neutral += geom * interception_efficiency
    if include_electrostatic:
        phi = (
            aq * dq * ELEMENTARY_CHARGE**2
            / (4.0 * PI * EPSILON_0 * contact * BOLTZMANN * temperature_k)
        )
        multiplier = np.clip(bernoulli(phi), 0.0, 100.0)
    else:
        multiplier = np.ones_like(neutral)
    charged = neutral * multiplier
    value = (charged, neutral, multiplier)
    _KERNEL_CACHE[key] = value
    _KERNEL_CACHE.move_to_end(key)
    while len(_KERNEL_CACHE) > _KERNEL_CACHE_MAX:
        _KERNEL_CACHE.popitem(last=False)
    return value


def pair_kernel(
    aerosol_r_m: np.ndarray,
    aerosol_q_e: np.ndarray,
    droplet_r_m: np.ndarray,
    droplet_q_e: np.ndarray,
    temperature_k: float,
    include_brownian: bool = True,
    include_interception: bool = True,
    include_electrostatic: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible wrapper returning charged kernel and multiplier."""
    charged, _neutral, multiplier = pair_kernel_components(
        aerosol_r_m, aerosol_q_e, droplet_r_m, droplet_q_e, temperature_k,
        include_brownian, include_interception, include_electrostatic,
    )
    return charged, multiplier


def _integer_charge_shifts(charge_states_e: np.ndarray) -> np.ndarray:
    rounded = np.rint(charge_states_e).astype(np.int64)
    if not np.allclose(charge_states_e, rounded, atol=1.0e-10, rtol=0.0):
        raise ValueError("Active charge transfer requires integer aerosol charge states.")
    return rounded


def _deposit_shifted_charge_batch(
    destination: np.ndarray,
    events_by_aerosol_charge: np.ndarray,
    aerosol_charge_states_e: np.ndarray,
) -> None:
    """Deposit [z,qa,r,qd] events into clipped qd+qa bins."""
    nq = destination.shape[-1]
    for iqa, shift in enumerate(_integer_charge_shifts(aerosol_charge_states_e)):
        ev = events_by_aerosol_charge[:, iqa]
        if not np.any(ev > 0.0):
            continue
        shift = int(shift)
        if shift == 0:
            destination += ev
        elif shift > 0:
            if shift >= nq:
                destination[:, :, -1] += np.sum(ev, axis=-1)
            else:
                destination[:, :, shift:] += ev[:, :, : nq - shift]
                destination[:, :, -1] += np.sum(ev[:, :, nq - shift :], axis=-1)
        else:
            width = -shift
            if width >= nq:
                destination[:, :, 0] += np.sum(ev, axis=-1)
            else:
                destination[:, :, : nq - width] += ev[:, :, width:]
                destination[:, :, 0] += np.sum(ev[:, :, :width], axis=-1)


def apply_electroscavenging(
    aerosol: np.ndarray,
    droplet: np.ndarray,
    aerosol_r_m: np.ndarray,
    aerosol_q_e: np.ndarray,
    droplet_r_m: np.ndarray,
    droplet_q_e: np.ndarray,
    temperature_k: float,
    dt: float,
    kernel_scale: float,
    max_fraction_per_step: float,
    active_transfer: bool,
    include_brownian: bool,
    include_interception: bool,
    include_electrostatic: bool,
    max_droplet_transfer_fraction_per_update: float = 0.20,
    vertical_block_size: int = 4,
) -> tuple[np.ndarray, np.ndarray, ScavengingDiagnostics]:
    """Apply one conservative, vectorized electroscavenging update.

    Compared with v5.0 this implementation:
      * builds/caches the charged and neutral kernels only once;
      * removes the nested Python loops over all aerosol/droplet states;
      * limits total outgoing droplet transitions collectively rather than once
        for every aerosol state;
      * subtracts only aerosol events that are actually transferred, ensuring
        number and charge consistency for the operator-split update.
    """
    if dt <= 0.0:
        raise ValueError("Electroscavenging dt must be positive.")
    charged, neutral, _multiplier = pair_kernel_components(
        aerosol_r_m,
        aerosol_q_e,
        droplet_r_m,
        droplet_q_e,
        temperature_k,
        include_brownian,
        include_interception,
        include_electrostatic,
    )
    charged = charged * kernel_scale
    neutral = neutral * kernel_scale
    a_new = aerosol.copy()
    d_new = droplet.copy()
    nz = aerosol.shape[0]
    removed = np.zeros(nz, dtype=float)
    qtransfer = np.zeros(nz, dtype=float)
    mean_mult = np.ones(nz, dtype=float)

    # Contractions over all vertical levels use optimized NumPy kernels.
    encounter_e_all = np.einsum("akrq,zrq->zak", charged, droplet, optimize=True)
    encounter_n_all = np.einsum("akrq,zrq->zak", neutral, droplet, optimize=True)
    frac_all = np.minimum(
        1.0 - np.exp(-np.maximum(encounter_e_all, 0.0) * dt),
        max_fraction_per_step,
    )
    requested_loss_all = aerosol * frac_all
    numer = np.sum(aerosol * encounter_e_all, axis=(1, 2))
    denom_mult = np.sum(aerosol * encounter_n_all, axis=(1, 2))
    np.divide(numer, denom_mult, out=mean_mult, where=denom_mult > 0.0)

    valid = (np.sum(droplet, axis=(1, 2)) > 0.0) & (np.sum(aerosol, axis=(1, 2)) > 0.0)
    if not active_transfer:
        removed[valid] = np.sum(requested_loss_all[valid], axis=(1, 2))
        qtransfer[valid] = np.sum(
            requested_loss_all[valid] * aerosol_q_e[None, None, :], axis=(1, 2)
        )
    else:
        block = max(1, int(vertical_block_size))
        for z0 in range(0, nz, block):
            z1 = min(nz, z0 + block)
            vb = valid[z0:z1]
            if not np.any(vb):
                continue
            # Keep only valid levels in this block to avoid unnecessary 5-D allocations.
            local_indices = np.nonzero(vb)[0]
            global_indices = z0 + local_indices
            d = droplet[global_indices]
            requested = requested_loss_all[global_indices]
            encounter = encounter_e_all[global_indices]

            denom = np.maximum(encounter, 1.0e-300)
            events = (
                requested[:, :, :, None, None]
                * charged[None, :, :, :, :]
                * d[:, None, None, :, :]
                / denom[:, :, :, None, None]
            )
            events *= (encounter > 0.0)[:, :, :, None, None]

            # Reject transitions that would leave the represented droplet
            # charge grid.  Corresponding aerosol particles remain in the
            # aerosol spectrum, preserving charge exactly rather than silently
            # clipping q_d+q_a at a boundary.
            qmin = float(droplet_q_e[0])
            qmax = float(droplet_q_e[-1])
            for iqa, qa_value in enumerate(aerosol_q_e):
                valid_q = (droplet_q_e + qa_value >= qmin) & (droplet_q_e + qa_value <= qmax)
                events[:, :, iqa, :, ~valid_q] = 0.0

            outgoing_raw = np.sum(events, axis=(1, 2))
            cap = max_droplet_transfer_fraction_per_update * np.maximum(d, 0.0)
            scale = np.ones_like(outgoing_raw)
            mask = outgoing_raw > cap
            scale[mask] = cap[mask] / np.maximum(outgoing_raw[mask], 1.0e-300)
            events *= scale[:, None, None, :, :]

            actual_loss = np.sum(events, axis=(3, 4))
            events_by_qa = np.sum(events, axis=1)  # [z,qa,dr,dq]
            outgoing = np.sum(events_by_qa, axis=1)

            a_block = a_new[global_indices].copy()
            d_block = d_new[global_indices].copy()
            a_block -= actual_loss
            d_block -= outgoing
            _deposit_shifted_charge_batch(d_block, events_by_qa, aerosol_q_e)
            a_new[global_indices] = a_block
            d_new[global_indices] = d_block

            removed[global_indices] = np.sum(actual_loss, axis=(1, 2))
            qtransfer[global_indices] = np.sum(
                actual_loss * aerosol_q_e[None, None, :], axis=(1, 2)
            )

    return (
        np.maximum(a_new, 0.0),
        np.maximum(d_new, 0.0),
        ScavengingDiagnostics(
            aerosol_removed_m3=removed,
            transferred_charge_e_m3=qtransfer,
            mean_electrostatic_multiplier=mean_mult,
        ),
    )
