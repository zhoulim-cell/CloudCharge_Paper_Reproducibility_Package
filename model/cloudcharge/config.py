from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass
class AtmosphereConfig:
    altitude_km: float = 4.0
    ion_pair_production_m3_s: float = 8.0e6
    current_density_a_m2: float = 3.0e-12
    recombination_m3_s: float = 1.65e-12
    positive_mobility_ref_m2_v_s: float = 1.4e-4
    negative_mobility_ref_m2_v_s: float = 1.9e-4


@dataclass
class AerosolConfig:
    total_number_m3: float = 5.0e9
    mean_radius_m: float = 4.0e-8


@dataclass
class DropletConfig:
    spectrum_mode: str = "fixed_peak"  # original_gamma | fixed_peak
    fixed_peak_number_m3: float = 2.0e8
    initial_radius_m: float = 1.0e-5
    target_radius_m: float = 1.0e-5
    growth_time_s: float = 0.0
    max_abs_charge_e: float = 500.0


@dataclass
class SpectrumConfig:
    """Discretisation and physics controls for N(r, k, z, t).

    The stored spectrum is bin-integrated number concentration [m-3] in
    radius bin i and integer charge state k.  The symbol k is used here to
    avoid confusing droplet charge with the ion-pair production rate q.
    """

    radius_bins: int = 18
    radius_min_m: float = 2.0e-6
    radius_max_m: float = 25.0e-6
    radius_spacing: str = "log"  # log | linear
    initial_size_distribution: str = "lognormal"  # lognormal | monodisperse
    geometric_std: float = 1.35
    charge_min_e: int = -180
    charge_max_e: int = 180
    initial_charge_mode: str = "neutral"  # neutral | gaussian
    initial_charge_std_e: float = 1.0
    charge_boundary: str = "reflecting"  # reflecting
    growth_law: str = "scaled_radius"  # none | scaled_radius | linear_radius | linear_radius_squared
    include_spectrum_feedback: bool = True
    carrier_coupling: str = "paper_constrained"  # paper_constrained | fully_direct
    store_full_spectrum: bool = True
    max_spectral_cells: int = 1_500_000
    normalize_number_each_step: bool = True
    charge_subcycle_safety: float = 0.80


@dataclass
class DomainConfig:
    length_m: float = 35.0
    grid_points: int = 351
    cloud_start_m: float = 2.0
    edge_ramp_m: float = 10.0
    core_width_m: float = 11.0


@dataclass
class TimeDependentConfig:
    """Schedule used by the Zhou-Tinsley (2012) time-dependent model."""

    initial_current_density_a_m2: float = 1.5e-12
    second_current_density_a_m2: float = 3.0e-12
    initial_ion_pair_production_m3_s: float = 8.0e6
    second_ion_pair_production_m3_s: float = 8.0e6
    step_time_s: float = 5000.0
    startup_end_s: float = 5000.0
    total_duration_s: float = 10000.0
    fit_time_constants: bool = True


@dataclass
class PhysicsConfig:
    # paper2007_bvp: equilibrium model in JGR 2007
    # paper2012_time: reduced/calibrated time-dependent model in ASR 2012
    # paper2012_direct: direct mean-charge time integration
    # paper2012_spectrum: self-consistent N(r,k,z,t) charging model
    # transient_charge_ode: retained modern/legacy exploratory solver
    solution_mode: str = "paper2007_bvp"
    attachment_mode: str = "paper2007_table"  # paper2007_table | corrected | legacy_shifted_boole
    charging_mode: str = "revised"  # revised | legacy
    field_mode: str = "poisson_projection"  # poisson_projection | poisson_only | conductivity_only
    field_anchor: str = "bottom"  # bottom | top | least_squares
    include_drift_charging: bool = False
    include_ion_diffusion: bool = False
    use_legacy_initialization: bool = False


@dataclass
class NumericsConfig:
    duration_s: float = 600.0
    max_dt_s: float = 1.0
    min_dt_s: float = 1.0e-4
    positivity_safety: float = 0.80
    field_iterations: int = 6
    field_relaxation: float = 0.50
    field_tolerance: float = 1.0e-6
    conservative_smoothing: float = 0.0
    bvp_tolerance: float = 1.0e-5
    bvp_max_nodes: int = 30000
    snapshot_times_s: list[float] = field(default_factory=lambda: [0.0, 600.0])
    diagnostics_interval_s: float = 10.0


@dataclass
class SimulationConfig:
    atmosphere: AtmosphereConfig = field(default_factory=AtmosphereConfig)
    aerosol: AerosolConfig = field(default_factory=AerosolConfig)
    droplet: DropletConfig = field(default_factory=DropletConfig)
    spectrum: SpectrumConfig = field(default_factory=SpectrumConfig)
    domain: DomainConfig = field(default_factory=DomainConfig)
    time_dependent: TimeDependentConfig = field(default_factory=TimeDependentConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    numerics: NumericsConfig = field(default_factory=NumericsConfig)

    def validate(self) -> None:
        if self.atmosphere.altitude_km < 0:
            raise ValueError("Altitude must be non-negative.")
        if self.atmosphere.ion_pair_production_m3_s <= 0:
            raise ValueError("Ion-pair production rate must be positive.")
        if self.atmosphere.recombination_m3_s <= 0:
            raise ValueError("Recombination coefficient must be positive.")
        if self.aerosol.total_number_m3 <= 0:
            raise ValueError("Total aerosol number must be positive.")
        if self.aerosol.mean_radius_m <= 0:
            raise ValueError("Aerosol radius must be positive.")
        if self.droplet.initial_radius_m <= 0 or self.droplet.target_radius_m <= 0:
            raise ValueError("Droplet radii must be positive.")
        if self.domain.grid_points < 21:
            raise ValueError("At least 21 grid points are required.")
        if self.domain.length_m <= 0:
            raise ValueError("Domain length must be positive.")
        cloud_end = self.domain.cloud_start_m + 2.0 * self.domain.edge_ramp_m + self.domain.core_width_m
        if self.domain.cloud_start_m < 0 or cloud_end > self.domain.length_m:
            raise ValueError("Cloud profile must fit inside the domain: start + 2*ramp + core <= length.")
        if self.numerics.duration_s <= 0 or self.numerics.max_dt_s <= 0:
            raise ValueError("Duration and time step must be positive.")
        if self.numerics.min_dt_s <= 0 or self.numerics.min_dt_s > self.numerics.max_dt_s:
            raise ValueError("min_dt must be positive and no larger than max_dt.")
        if not (0 < self.numerics.positivity_safety <= 1):
            raise ValueError("Positivity safety must lie in (0, 1].")
        if not (0 < self.numerics.field_relaxation <= 1):
            raise ValueError("Field relaxation must lie in (0, 1].")
        modes = {"paper2007_bvp", "paper2012_time", "paper2012_direct", "paper2012_spectrum", "transient_charge_ode"}
        if self.physics.solution_mode not in modes:
            raise ValueError("Unknown solution mode.")
        if self.physics.attachment_mode not in {"paper2007_table", "corrected", "legacy_shifted_boole"}:
            raise ValueError("Unknown attachment mode.")
        if self.physics.charging_mode not in {"revised", "legacy"}:
            raise ValueError("Unknown charging mode.")
        if self.physics.field_mode not in {"poisson_projection", "poisson_only", "conductivity_only"}:
            raise ValueError("Unknown field mode.")
        if self.physics.field_anchor not in {"bottom", "top", "least_squares"}:
            raise ValueError("Unknown field anchor.")
        if self.numerics.bvp_tolerance <= 0:
            raise ValueError("BVP tolerance must be positive.")
        if self.numerics.bvp_max_nodes < self.domain.grid_points:
            raise ValueError("BVP max nodes must be at least the number of output grid points.")
        if self.physics.solution_mode == "paper2007_bvp" and self.atmosphere.current_density_a_m2 <= 1.0e-18:
            raise ValueError("paper2007_bvp requires downward-positive Jz > 0.")

        td = self.time_dependent
        if td.initial_current_density_a_m2 <= 0 or td.second_current_density_a_m2 <= 0:
            raise ValueError("2012 current densities must be positive downward.")
        if td.initial_ion_pair_production_m3_s <= 0 or td.second_ion_pair_production_m3_s <= 0:
            raise ValueError("2012 ion-pair production rates must be positive.")
        if td.step_time_s <= 0 or td.total_duration_s <= td.step_time_s:
            raise ValueError("2012 total duration must exceed the positive step time.")
        if td.startup_end_s < self.droplet.growth_time_s:
            raise ValueError("Startup end must not precede completion of droplet growth.")

        sp = self.spectrum
        if sp.radius_bins < 4:
            raise ValueError("At least four radius bins are required for the spectrum model.")
        if sp.radius_min_m <= 0 or sp.radius_max_m <= sp.radius_min_m:
            raise ValueError("Spectrum radius bounds are invalid.")
        if sp.radius_spacing not in {"log", "linear"}:
            raise ValueError("Unknown radius spacing.")
        if sp.initial_size_distribution not in {"lognormal", "monodisperse"}:
            raise ValueError("Unknown initial size distribution.")
        if sp.geometric_std <= 1.0 and sp.initial_size_distribution == "lognormal":
            raise ValueError("Lognormal geometric standard deviation must exceed one.")
        if sp.charge_min_e >= 0 or sp.charge_max_e <= 0 or sp.charge_max_e <= sp.charge_min_e:
            raise ValueError("Charge grid must straddle zero.")
        if sp.initial_charge_mode not in {"neutral", "gaussian"}:
            raise ValueError("Unknown initial charge mode.")
        if sp.initial_charge_std_e <= 0:
            raise ValueError("Initial charge standard deviation must be positive.")
        if sp.charge_boundary != "reflecting":
            raise ValueError("Only reflecting charge boundaries are currently supported.")
        if sp.growth_law not in {"none", "scaled_radius", "linear_radius", "linear_radius_squared"}:
            raise ValueError("Unknown spectrum growth law.")
        if sp.carrier_coupling not in {"paper_constrained", "fully_direct"}:
            raise ValueError("Unknown spectral carrier coupling mode.")
        if not (0 < sp.charge_subcycle_safety <= 1):
            raise ValueError("Charge subcycle safety must lie in (0, 1].")
        spectral_cells = self.domain.grid_points * sp.radius_bins * (sp.charge_max_e - sp.charge_min_e + 1)
        if self.physics.solution_mode == "paper2012_spectrum" and spectral_cells > sp.max_spectral_cells:
            raise ValueError(
                f"Spectrum grid has {spectral_cells:,} cells, exceeding max_spectral_cells={sp.max_spectral_cells:,}. "
                "Reduce z, radius or charge resolution, or deliberately raise the safety limit."
            )

        if self.physics.solution_mode in {"paper2012_time", "paper2012_direct", "paper2012_spectrum"}:
            self.numerics.duration_s = td.total_duration_s

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimulationConfig":
        return cls(
            atmosphere=AtmosphereConfig(**data.get("atmosphere", {})),
            aerosol=AerosolConfig(**data.get("aerosol", {})),
            droplet=DropletConfig(**data.get("droplet", {})),
            spectrum=SpectrumConfig(**data.get("spectrum", {})),
            domain=DomainConfig(**data.get("domain", {})),
            time_dependent=TimeDependentConfig(**data.get("time_dependent", {})),
            physics=PhysicsConfig(**data.get("physics", {})),
            numerics=NumericsConfig(**data.get("numerics", {})),
        )

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "SimulationConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        cfg = cls.from_dict(data)
        cfg.validate()
        return cfg
