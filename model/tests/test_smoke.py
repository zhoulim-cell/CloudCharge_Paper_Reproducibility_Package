from __future__ import annotations

import unittest
import numpy as np

from cloudcharge.config import SimulationConfig
from cloudcharge.solver import CloudChargeModel


class CloudChargeSmokeTests(unittest.TestCase):
    def test_paper2007_baseline_has_opposite_boundary_polarity(self) -> None:
        cfg = SimulationConfig()
        model = CloudChargeModel(cfg)
        result = model.run()
        self.assertTrue(bool(result.diagnostics["opposite_droplet_sign_test"]))
        self.assertTrue(bool(result.diagnostics["opposite_space_charge_sign_test"]))
        self.assertGreater(float(result.diagnostics["upper_peak_droplet_charge_e"]), 50.0)
        self.assertLess(float(result.diagnostics["lower_peak_droplet_charge_e"]), -50.0)
        self.assertLess(float(result.diagnostics["final_max_relative_current_residual"]), 1.0e-10)

    def test_paper2007_table_matches_table1_baseline(self) -> None:
        cfg = SimulationConfig()
        model = CloudChargeModel(cfg)
        self.assertAlmostEqual(model.beta_pos / 1.0e-12, 1.61, places=10)
        self.assertAlmostEqual(model.beta_neg / 1.0e-12, 2.27, places=10)
        self.assertAlmostEqual(model.gamma_pos / 1.0e-12, 3.64, places=10)
        self.assertAlmostEqual(model.gamma_neg / 1.0e-12, 4.98, places=10)

    def test_transient_short_run_still_available(self) -> None:
        cfg = SimulationConfig()
        cfg.physics.solution_mode = "transient_charge_ode"
        cfg.physics.attachment_mode = "corrected"
        cfg.numerics.duration_s = 2.0
        cfg.numerics.snapshot_times_s = [0.0, 2.0]
        cfg.domain.grid_points = 181
        model = CloudChargeModel(cfg)
        result = model.run()
        final = result.snapshots[2.0]
        self.assertTrue(np.all(np.asarray(final["n_positive_m3"]) > 0.0))
        self.assertTrue(np.all(np.asarray(final["n_negative_m3"]) > 0.0))
        self.assertTrue(np.all(np.isfinite(final["electric_field_v_m"])))


    def test_paper2012_baseline_matches_reported_response(self) -> None:
        cfg = SimulationConfig()
        cfg.physics.solution_mode = "paper2012_time"
        cfg.aerosol.total_number_m3 = 1.0e9
        cfg.droplet.initial_radius_m = 0.01e-6
        cfg.droplet.target_radius_m = 10.0e-6
        cfg.droplet.growth_time_s = 600.0
        cfg.domain.grid_points = 81
        cfg.time_dependent.total_duration_s = 10000.0
        cfg.numerics.duration_s = 10000.0
        cfg.numerics.max_dt_s = 10.0
        cfg.numerics.snapshot_times_s = [0.0, 600.0, 5000.0, 10000.0]
        result = CloudChargeModel(cfg).run()
        self.assertAlmostEqual(float(result.diagnostics["response_tau_used_s"]), 800.0, delta=1.0)
        self.assertAlmostEqual(float(result.diagnostics["upper_peak_droplet_charge_e"]), 107.0, delta=1.0)
        self.assertAlmostEqual(float(result.diagnostics["lower_peak_droplet_charge_e"]), -117.0, delta=1.0)
        self.assertTrue(bool(result.diagnostics["opposite_droplet_sign_test"]))


    def test_size_charge_spectrum_conserves_number_and_polarity(self) -> None:
        cfg = SimulationConfig()
        cfg.physics.solution_mode = "paper2012_spectrum"
        cfg.physics.include_drift_charging = True
        cfg.domain.grid_points = 31
        cfg.aerosol.total_number_m3 = 1.0e9
        cfg.droplet.initial_radius_m = 10.0e-6
        cfg.droplet.target_radius_m = 10.0e-6
        cfg.droplet.growth_time_s = 0.0
        cfg.time_dependent.step_time_s = 1000.0
        cfg.time_dependent.startup_end_s = 1000.0
        cfg.time_dependent.total_duration_s = 2000.0
        cfg.numerics.max_dt_s = 10.0
        cfg.numerics.snapshot_times_s = [0.0, 1000.0, 2000.0]
        cfg.numerics.diagnostics_interval_s = 100.0
        cfg.spectrum.radius_bins = 6
        cfg.spectrum.radius_min_m = 5.0e-6
        cfg.spectrum.radius_max_m = 15.0e-6
        cfg.spectrum.charge_min_e = -120
        cfg.spectrum.charge_max_e = 120
        cfg.spectrum.max_spectral_cells = 300000
        result = CloudChargeModel(cfg).run()
        final = result.snapshots[2000.0]
        self.assertEqual(np.asarray(final["droplet_spectrum_m3"]).shape, (31, 6, 241))
        self.assertTrue(bool(result.diagnostics["opposite_droplet_sign_test"]))
        self.assertLess(float(result.diagnostics["final_max_number_conservation_error"]), 1.0e-12)
        self.assertLess(float(result.diagnostics["final_charge_grid_edge_fraction"]), 1.0e-3)

    def test_legacy_attachment_available(self) -> None:
        cfg = SimulationConfig()
        cfg.physics.solution_mode = "transient_charge_ode"
        cfg.physics.attachment_mode = "legacy_shifted_boole"
        model = CloudChargeModel(cfg)
        self.assertGreater(model.beta_pos, 0.0)
        self.assertGreater(model.gamma_neg, model.beta_neg)


if __name__ == "__main__":
    unittest.main()
