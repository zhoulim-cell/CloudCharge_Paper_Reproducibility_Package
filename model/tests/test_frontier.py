from __future__ import annotations

from copy import deepcopy
import unittest

import numpy as np

from cloudcharge.frontier_config import FrontierConfig
from cloudcharge.frontier_numerics import charge_moments
from cloudcharge.frontier_observations import make_synthetic_observations, observation_vector
from cloudcharge.frontier_scavenging import apply_electroscavenging, pair_kernel
from cloudcharge.frontier_solver import run_frontier_model


def fast_config(duration: float = 60.0) -> FrontierConfig:
    cfg = FrontierConfig()
    cfg.base.domain.grid_points = 25
    cfg.base.domain.length_m = 35.0
    cfg.base.domain.cloud_start_m = 2.0
    cfg.base.domain.edge_ramp_m = 10.0
    cfg.base.domain.core_width_m = 11.0
    cfg.base.droplet.initial_radius_m = 8.0e-6
    cfg.base.droplet.target_radius_m = 10.0e-6
    cfg.base.droplet.growth_time_s = 0.0
    cfg.base.spectrum.radius_bins = 4
    cfg.base.spectrum.radius_min_m = 4.0e-6
    cfg.base.spectrum.radius_max_m = 16.0e-6
    cfg.base.spectrum.charge_min_e = -30
    cfg.base.spectrum.charge_max_e = 30
    cfg.aerosol_spectrum.radius_bins = 3
    cfg.aerosol_spectrum.charge_min_e = -2
    cfg.aerosol_spectrum.charge_max_e = 2
    cfg.transport.enabled = True
    cfg.transport.eddy_diffusivity_m2_s = 0.01
    cfg.transport.ion_eddy_diffusivity_m2_s = 0.01
    cfg.transport.include_droplet_sedimentation = False
    cfg.electrodynamics.gauss_projection_relaxation = 0.5
    cfg.forcing.current_mode = "constant"
    cfg.forcing.ionization_mode = "constant"
    cfg.forcing.current_density_initial_a_m2 = 3.0e-12
    cfg.forcing.current_density_final_a_m2 = 3.0e-12
    cfg.forcing.ionization_initial_m3_s = 8.0e6
    cfg.forcing.ionization_final_m3_s = 8.0e6
    cfg.forcing.step_time_s = duration + 1.0
    cfg.numerics.duration_s = duration
    cfg.numerics.max_dt_s = 0.5
    cfg.numerics.diagnostics_interval_s = max(10.0, duration / 3.0)
    cfg.numerics.snapshot_times_s = [0.0, duration]
    cfg.assimilation.ensemble_size = 4
    cfg.assimilation.iterations = 1
    cfg.validate()
    return cfg


class FrontierModelTests(unittest.TestCase):
    def test_frontier_forward_is_finite_and_charge_resolved(self) -> None:
        cfg = fast_config(30.0)
        result = run_frontier_model(cfg)
        final = result.snapshots[30.0]
        self.assertEqual(
            np.asarray(final["droplet_spectrum_m3"]).shape,
            (25, 4, 61),
        )
        self.assertEqual(
            np.asarray(final["aerosol_spectrum_m3"]).shape,
            (25, 3, 5),
        )
        for name in [
            "droplet_spectrum_m3",
            "aerosol_spectrum_m3",
            "n_positive_m3",
            "n_negative_m3",
            "electric_field_v_m",
            "charge_density_c_m3",
        ]:
            self.assertTrue(np.all(np.isfinite(np.asarray(final[name]))), name)
        self.assertTrue(np.all(np.asarray(final["n_positive_m3"]) > 0.0))
        self.assertTrue(np.all(np.asarray(final["n_negative_m3"]) > 0.0))
        self.assertLess(float(result.diagnostics["final_max_relative_total_current_residual"]), 0.2)

    def test_boundary_polarity_emerges(self) -> None:
        cfg = fast_config(120.0)
        result = run_frontier_model(cfg)
        self.assertTrue(bool(result.diagnostics["opposite_droplet_sign_test"]))
        self.assertTrue(bool(result.diagnostics["opposite_space_charge_sign_test"]))
        self.assertGreater(float(result.diagnostics["upper_peak_droplet_charge_e"]), 0.0)
        self.assertLess(float(result.diagnostics["lower_peak_droplet_charge_e"]), 0.0)

    def test_charge_birth_death_preserves_particle_number_without_transport(self) -> None:
        cfg = fast_config(20.0)
        cfg.transport.enabled = False
        result = run_frontier_model(cfg)
        first = result.snapshots[0.0]
        final = result.snapshots[20.0]
        n0 = np.sum(np.asarray(first["droplet_spectrum_m3"], dtype=float), axis=(1, 2))
        n1 = np.sum(np.asarray(final["droplet_spectrum_m3"], dtype=float), axis=(1, 2))
        mask = n0 > 0
        self.assertLess(float(np.max(np.abs(n1[mask] / n0[mask] - 1.0))), 1.0e-6)

    def test_observation_operator_roundtrip(self) -> None:
        cfg = fast_config(10.0)
        result = run_frontier_model(cfg)
        obs = make_synthetic_observations(
            result,
            10.0,
            [5.0, 15.0, 28.0],
            ["electric_field_v_m", "droplet_charge_e", "conductivity_s_m"],
            relative_noise=0.0,
            absolute_noise={"electric_field_v_m": 1.0e-30, "droplet_charge_e": 1.0e-30},
            seed=3,
        )
        predicted = observation_vector(result, obs)
        self.assertEqual(predicted.shape, obs.value.shape)
        self.assertTrue(np.all(np.isfinite(predicted)))
        self.assertLess(float(np.max(np.abs(predicted - obs.value))), 1.0e-15 * max(1.0, float(np.max(np.abs(predicted)))))

    def test_electrostatic_kernel_has_expected_polarity_order(self) -> None:
        ar = np.array([50.0e-9])
        aq = np.array([-1.0, 0.0, 1.0])
        dr = np.array([10.0e-6])
        dq = np.array([-20.0, 0.0, 20.0])
        _, mult = pair_kernel(ar, aq, dr, dq, 273.0, True, True, True)
        # aerosol +1 with droplet -20 is attractive; +1/+20 is repulsive.
        attractive = float(mult[0, 2, 0, 0])
        neutral = float(mult[0, 1, 0, 1])
        repulsive = float(mult[0, 2, 0, 2])
        self.assertGreater(attractive, neutral)
        self.assertLess(repulsive, neutral)

    def test_active_scavenging_removes_aerosol_and_preserves_drop_number(self) -> None:
        aerosol = np.zeros((1, 1, 3), dtype=float)
        aerosol[0, 0, 2] = 1.0e8  # +1 e
        droplet = np.zeros((1, 1, 7), dtype=float)
        droplet[0, 0, 3] = 2.0e8  # 0 e
        a2, d2, diag = apply_electroscavenging(
            aerosol,
            droplet,
            np.array([50.0e-9]),
            np.array([-1.0, 0.0, 1.0]),
            np.array([10.0e-6]),
            np.arange(-3.0, 4.0),
            273.0,
            dt=0.1,
            kernel_scale=1.0,
            max_fraction_per_step=0.01,
            active_transfer=True,
            include_brownian=True,
            include_interception=True,
            include_electrostatic=True,
        )
        self.assertLess(float(np.sum(a2)), float(np.sum(aerosol)))
        self.assertAlmostEqual(float(np.sum(d2)), float(np.sum(droplet)), delta=1.0e-7 * float(np.sum(droplet)))
        self.assertGreaterEqual(float(diag.aerosol_removed_m3[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
