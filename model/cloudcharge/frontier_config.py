from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

from .config import SimulationConfig


@dataclass
class AerosolSpectrumConfig:
    radius_bins: int = 6
    radius_min_m: float = 5.0e-9
    radius_max_m: float = 5.0e-7
    geometric_mean_radius_m: float = 4.0e-8
    geometric_std: float = 1.65
    charge_min_e: int = -5
    charge_max_e: int = 5
    initial_charge_std_e: float = 0.8
    measured_attachment_table: str = ""  # optional CSV: radius_m,charge_e,polarity,rate_m3_s


@dataclass
class TransportConfig:
    enabled: bool = True
    air_velocity_down_m_s: float = 0.0
    eddy_diffusivity_m2_s: float = 0.03
    ion_eddy_diffusivity_m2_s: float = 0.03
    cloud_edge_pulsation_amplitude_m_s: float = 0.0
    cloud_edge_pulsation_period_s: float = 900.0
    include_droplet_sedimentation: bool = True
    include_condensational_growth: bool = False
    supersaturation_peak: float = 0.002
    condensational_growth_coefficient_m2_s: float = 1.0e-10


@dataclass
class ElectrodynamicsConfig:
    include_displacement_current: bool = True
    gauss_projection_relaxation: float = 0.35
    electric_field_floor_v_m: float = -5000.0
    electric_field_ceiling_v_m: float = 5000.0
    include_ion_diffusion_current: bool = True
    enforce_global_charge_neutrality: bool = False
    projection_interval_steps: int = 1
    # open_inflow reproduces the original local-domain treatment.
    # clear_air_reservoir embeds the finite domain between two undisturbed
    # clear-air reservoirs and is recommended for extended-domain profiles.
    boundary_condition_mode: str = "open_inflow"
    boundary_reservoir_cells: int = 2
    boundary_field_anchor_weight: float = 10.0


@dataclass
class ScavengingConfig:
    enabled: bool = False
    active_transfer: bool = False
    kernel_scale: float = 1.0
    max_fraction_per_step: float = 0.05
    include_brownian: bool = True
    include_interception: bool = True
    include_electrostatic: bool = True
    aerosol_density_kg_m3: float = 1800.0
    update_interval_s: float = 10.0
    max_droplet_transfer_fraction_per_update: float = 0.20
    vertical_block_size: int = 4


@dataclass
class ForcingConfig:
    current_mode: str = "step"  # constant | step | sinusoid | forbush
    ionization_mode: str = "step"  # constant | step | sinusoid | forbush
    step_time_s: float = 300.0
    current_density_initial_a_m2: float = 1.5e-12
    current_density_final_a_m2: float = 3.0e-12
    ionization_initial_m3_s: float = 8.0e6
    ionization_final_m3_s: float = 8.0e6
    sinusoid_period_s: float = 3600.0
    sinusoid_fractional_amplitude: float = 0.25
    forbush_drop_fraction: float = 0.30
    forbush_recovery_time_s: float = 1800.0


@dataclass
class FrontierNumericsConfig:
    duration_s: float = 600.0
    max_dt_s: float = 0.5
    min_dt_s: float = 1.0e-4
    cfl_safety: float = 0.45
    diagnostics_interval_s: float = 10.0
    snapshot_times_s: list[float] = field(default_factory=lambda: [0.0, 60.0, 300.0, 600.0])
    max_spectral_cells: int = 3_000_000
    positivity_floor_m3: float = 0.0


@dataclass
class AssimilationConfig:
    ensemble_size: int = 16
    iterations: int = 3
    inflation: float = 1.02
    random_seed: int = 42
    parameter_names: list[str] = field(
        default_factory=lambda: [
            "current_scale",
            "ionization_scale",
            "aerosol_scale",
            "eddy_diffusivity_scale",
        ]
    )
    prior_std_fraction: float = 0.25


@dataclass
class FrontierConfig:
    base: SimulationConfig = field(default_factory=SimulationConfig)
    aerosol_spectrum: AerosolSpectrumConfig = field(default_factory=AerosolSpectrumConfig)
    transport: TransportConfig = field(default_factory=TransportConfig)
    electrodynamics: ElectrodynamicsConfig = field(default_factory=ElectrodynamicsConfig)
    scavenging: ScavengingConfig = field(default_factory=ScavengingConfig)
    forcing: ForcingConfig = field(default_factory=ForcingConfig)
    numerics: FrontierNumericsConfig = field(default_factory=FrontierNumericsConfig)
    assimilation: AssimilationConfig = field(default_factory=AssimilationConfig)

    def validate(self) -> None:
        self.base.validate()
        a = self.aerosol_spectrum
        if a.radius_bins < 2 or a.radius_min_m <= 0 or a.radius_max_m <= a.radius_min_m:
            raise ValueError("Invalid aerosol radius grid.")
        if a.geometric_std <= 1:
            raise ValueError("Aerosol geometric_std must exceed one.")
        if a.charge_min_e >= 0 or a.charge_max_e <= 0:
            raise ValueError("Aerosol charge grid must straddle zero.")
        t = self.transport
        if t.eddy_diffusivity_m2_s < 0 or t.ion_eddy_diffusivity_m2_s < 0:
            raise ValueError("Eddy diffusivities must be non-negative.")
        e = self.electrodynamics
        if not (0 <= e.gauss_projection_relaxation <= 1):
            raise ValueError("Gauss projection relaxation must be in [0,1].")
        if e.projection_interval_steps < 1:
            raise ValueError("projection_interval_steps must be >=1.")
        if e.boundary_condition_mode not in {"open_inflow", "clear_air_reservoir"}:
            raise ValueError("Unknown electrodynamic boundary_condition_mode.")
        if e.boundary_reservoir_cells < 1:
            raise ValueError("boundary_reservoir_cells must be >=1.")
        if e.boundary_field_anchor_weight < 0:
            raise ValueError("boundary_field_anchor_weight must be non-negative.")
        s = self.scavenging
        if s.update_interval_s <= 0:
            raise ValueError("scavenging.update_interval_s must be positive.")
        if not (0 < s.max_droplet_transfer_fraction_per_update <= 1):
            raise ValueError("scavenging.max_droplet_transfer_fraction_per_update must be in (0,1].")
        if s.vertical_block_size < 1:
            raise ValueError("scavenging.vertical_block_size must be >= 1.")
        f = self.forcing
        if f.current_mode not in {"constant", "step", "sinusoid", "forbush"}:
            raise ValueError("Unknown current forcing mode.")
        if f.ionization_mode not in {"constant", "step", "sinusoid", "forbush"}:
            raise ValueError("Unknown ionization forcing mode.")
        n = self.numerics
        if n.duration_s <= 0 or n.max_dt_s <= 0 or n.min_dt_s <= 0:
            raise ValueError("Invalid time settings.")
        if n.min_dt_s > n.max_dt_s:
            raise ValueError("min_dt_s cannot exceed max_dt_s.")
        if not (0 < n.cfl_safety < 1):
            raise ValueError("cfl_safety must be in (0,1).")
        nz = self.base.domain.grid_points
        nd = self.base.spectrum.radius_bins * (
            self.base.spectrum.charge_max_e - self.base.spectrum.charge_min_e + 1
        )
        na = a.radius_bins * (a.charge_max_e - a.charge_min_e + 1)
        total = nz * (nd + na)
        if total > n.max_spectral_cells:
            raise ValueError(
                f"Frontier spectrum has {total:,} cells, exceeding max_spectral_cells={n.max_spectral_cells:,}."
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FrontierConfig":
        cfg = cls(
            base=SimulationConfig.from_dict(data.get("base", {})),
            aerosol_spectrum=AerosolSpectrumConfig(**data.get("aerosol_spectrum", {})),
            transport=TransportConfig(**data.get("transport", {})),
            electrodynamics=ElectrodynamicsConfig(**data.get("electrodynamics", {})),
            scavenging=ScavengingConfig(**data.get("scavenging", {})),
            forcing=ForcingConfig(**data.get("forcing", {})),
            numerics=FrontierNumericsConfig(**data.get("numerics", {})),
            assimilation=AssimilationConfig(**data.get("assimilation", {})),
        )
        cfg.validate()
        return cfg

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "FrontierConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
