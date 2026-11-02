from __future__ import annotations

import numpy as np

from cloudcharge.frontier_scavenging import apply_electroscavenging


def test_active_transfer_conserves_droplet_number_and_charge() -> None:
    nz = 3
    aerosol_r = np.geomspace(5.0e-9, 2.0e-7, 3)
    aerosol_q = np.arange(-2, 3, dtype=float)
    droplet_r = np.geomspace(2.0e-6, 15.0e-6, 4)
    droplet_q = np.arange(-20, 21, dtype=float)
    aerosol = np.ones((nz, aerosol_r.size, aerosol_q.size)) * 2.0e6
    aerosol[:, :, aerosol_q > 0] *= 1.7
    droplet = np.ones((nz, droplet_r.size, droplet_q.size)) * 3.0e4

    a_new, d_new, _ = apply_electroscavenging(
        aerosol,
        droplet,
        aerosol_r,
        aerosol_q,
        droplet_r,
        droplet_q,
        261.0,
        5.0,
        1.0,
        0.05,
        True,
        True,
        True,
        True,
        0.20,
        2,
    )
    aerosol_loss = aerosol - a_new
    droplet_delta = d_new - droplet
    assert np.min(a_new) >= 0.0
    assert np.min(d_new) >= 0.0
    assert abs(float(np.sum(droplet_delta))) < 1.0e-5
    aerosol_charge_removed = float(np.sum(aerosol_loss * aerosol_q[None, None, :]))
    droplet_charge_added = float(np.sum(droplet_delta * droplet_q[None, None, :]))
    assert np.isclose(droplet_charge_added, aerosol_charge_removed, rtol=1.0e-10, atol=1.0e-5)
