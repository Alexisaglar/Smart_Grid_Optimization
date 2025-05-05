"""
Optimisation core for a one-bus-per-EV Virtual Power Plant.

Implements exactly the mathematical model given in the LaTeX draft
(equations 1–12), including:

‒ fixed native demand Pᴰ  
‒ PV-allocation hierarchy (PV→Load →BESS →EV →Curtail)  
‒ signed power variables for BESS, EV and Grid  
‒ linear-programme formulation solved with CVXPY + HiGHS/ECOS

Author : Alexis Aguilar Celis   – May 2025
"""

from __future__ import annotations
import time
import numpy as np
import pandas as pd
import cvxpy as cp
import utils.config as config   # adapt if your repo uses a different path


def solve_VPP_optimisation(
    ev_data          : pd.DataFrame,
    ev_present_matrix: np.ndarray,
    pv_power_forecasts: np.ndarray,
    p_demand         : np.ndarray,
):
    """
    Parameters
    ----------
    ev_data : DataFrame       – one row per EV / bus
      Required columns:
        ['capacity_mwh', 'max_power_mw', 'min_power_mw',
         'soc_at_arrival', 'target_soc', 'departure_time']
    ev_present_matrix : ndarray shape (N, T)   – 0 / 1 (vehicle plugged in)
    pv_power_forecasts: ndarray shape (N, T)   – PV generation [MW]
    p_demand          : ndarray shape (N, T)   – native load  [MW]

    Returns
    -------
    tuple (
        ev_power       [N,T],
        bess_power     [N,T],
        grid_power     [N,T],
        bess_soc       [N,T+1],
        ev_soc         [N,T+1],
        objective_value,
        solver_status
    )
    All power arrays are signed (+ = import / charging, − = export).
    """

    # ---------------------------------------------------------------- data
    tic          = time.time()
    N, T         = ev_present_matrix.shape
    dt           = config.TIME_STEP_DURATION_H

    prices       = np.asarray(config.ELECTRICITY_PRICE_PER_MWH)  # length T
    pv_power     = np.asarray(pv_power_forecasts)
    demand       = np.asarray(p_demand)

    # ---- broadcast scalar BESS parameters to length N
    cap_bess     = np.broadcast_to(config.BESS_CAPACITY_MWH,  (N,))
    p_bess_max   = np.broadcast_to(config.BESS_MAX_POWER_MW,  (N,))
    soc0_bess    = np.broadcast_to(config.BESS_INITIAL_SOC,   (N,))
    soc_bess_min = np.broadcast_to(config.BESS_MIN_SOC,       (N,))
    soc_bess_max = np.broadcast_to(config.BESS_MAX_SOC,       (N,))
    eta_bc, eta_bd = config.BESS_CHARGING_EFFICIENCY, config.BESS_DISCHARGING_EFFICIENCY

    # ---------------------------------------------------------------- vars
    # BESS (split into non-neg parts for LP)
    bess_chg = cp.Variable((N, T), name="bess_chg")      # ≥0
    bess_dis = cp.Variable((N, T), name="bess_dis")      # ≥0
    bess_p   = bess_chg - bess_dis                       # signed
    bess_soc = cp.Variable((N, T + 1), name="bess_soc")

    # EV (also split)
    ev_chg   = cp.Variable((N, T), name="ev_chg")        # ≥0
    ev_dis   = cp.Variable((N, T), name="ev_dis")        # ≥0
    ev_p     = ev_chg - ev_dis                           # signed
    ev_soc   = cp.Variable((N, T + 1), name="ev_soc")

    # PV allocation flows (all ≥0)
    pv_to_load = cp.Variable((N, T), name="pv_load")
    pv_to_bess = cp.Variable((N, T), name="pv_bess")
    pv_to_ev   = cp.Variable((N, T), name="pv_ev")
    pv_curt    = cp.Variable((N, T), name="pv_curt")

    # Grid power (signed: + import, − export)
    grid_p = cp.Variable((N, T), name="grid")

    # ---------------------------------------------------------------- constraints
    c = []

    # 1) Bus balance (eq. 1)
    c += [grid_p + bess_p + ev_p + demand - pv_power == 0]

    # 2)-5) PV hierarchy (eqs. 2-5)
    c += [pv_to_load <= pv_power,
          pv_to_load <= demand]

    c += [pv_to_bess <= pv_power - pv_to_load,
          pv_to_bess <= bess_chg]           # cannot exceed requested charge

    c += [pv_to_ev   <= pv_power - pv_to_load - pv_to_bess,
          pv_to_ev   <= ev_chg]

    c += [pv_curt == pv_power - pv_to_load - pv_to_bess - pv_to_ev,
          pv_curt >= 0]

    # Non-negativity already implicit except for ev/bess dis
    c += [var >= 0 for var in (bess_chg, bess_dis, ev_chg, ev_dis)]

    # 6)-7) Device power limits (signed form via split variables)
    c += [bess_chg  <= p_bess_max[:, None],
          bess_dis  <= p_bess_max[:, None]]

    for i in range(N):
        mask   = ev_present_matrix[i]
        p_max  = ev_data.at[i, 'max_power_mw']
        p_min  = ev_data.at[i, 'min_power_mw']   # negative
        c += [
            ev_chg[i] <=  p_max * mask,
            ev_dis[i] <= -p_min * mask
        ]

    # 8)-9) Energy dynamics
    c += [bess_soc[:, 0] == soc0_bess]
    c += [bess_soc[:, 1:] == bess_soc[:, :-1] +
          (eta_bc * bess_chg - bess_dis / eta_bd) * dt / cap_bess[:, None]]

    c += [bess_soc >= soc_bess_min[:, None],
          bess_soc <= soc_bess_max[:, None]]

    c += [ev_soc[:, 0] == ev_data['soc_at_arrival'].values]
    for i in range(N):
        cap_ev   = ev_data.at[i, 'capacity_mwh']
        dep_idx  = int(ev_data.at[i, 'departure_time'])
        target   = ev_data.at[i, 'target_soc']
        mask     = ev_present_matrix[i]
        c += [
            ev_soc[i, 1:] ==
                ev_soc[i, :-1] +
                (config.EV_CHARGING_EFFICIENCY * ev_chg[i] -
                 ev_dis[i] / config.EV_DISCHARGING_EFFICIENCY) * dt / cap_ev,

            ev_soc[i, 1:] >= config.EV_MIN_SOC * mask,
            ev_soc[i, 1:] <= config.EV_MAX_SOC * mask + 1 - mask,
            ev_soc[i, dep_idx] >= target
        ]

    # ---------------------------------------------------------------- objective (eq. 12)
    total_grid = cp.sum(grid_p, axis=0)          # MW per time step
    cost       = cp.sum(cp.multiply(prices, total_grid) * dt)

    prob = cp.Problem(cp.Minimize(cost), c)

    # ---------------------------------------------------------------- solve
    try:
        prob.solve(solver="HiGHS", verbose=False)
    except cp.error.SolverError:
        prob.solve(verbose=False)   # fall back to any installed solver

    toc = time.time()
    print(f"Optimisation finished in {toc - tic:.2f} s – status: {prob.status}")

    return (
        ev_p.value,
        bess_p.value,
        grid_p.value,
        bess_soc.value,
        ev_soc.value,
        prob.value,
        prob.status
    )
