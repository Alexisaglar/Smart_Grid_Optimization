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

    # ---------------------------------------------------------------- parameters
    N, T = ev_present_matrix.shape
    dt = config.TIME_STEP_DURATION_H
    prices = config.ELECTRICITY_PRICE_PER_MWH  # length T
    pv_power = pv_power_forecasts
    demand_power = np.full((33, T), p_demand)
    cap_bess = config.BESS_CAPACITY_MWH
    p_bess_max = config.BESS_MAX_POWER_MW
    soc0_bess = config.BESS_INITIAL_SOC
    soc_bess_min = config.BESS_MIN_SOC
    soc_bess_max = config.BESS_MAX_SOC
    eta_bc, eta_bd = config.BESS_CHARGING_EFFICIENCY, config.BESS_DISCHARGING_EFFICIENCY

    # ---------------------------------------------------------------- vars
    # BESS (split into non-neg parts for LP)
    bess_charge = cp.Variable((N, T), name="bess_charge")
    bess_discharge = cp.Variable((N, T), name="bess_discharge")
    bess_power   = bess_charge - bess_discharge
    bess_soc = cp.Variable((N, T + 1), name="bess_soc")

    # EV (also split)
    ev_charge = cp.Variable((N, T), name="ev_charge")
    ev_discharge = cp.Variable((N, T), name="ev_discharge")
    ev_power = ev_charge - ev_discharge
    ev_soc = cp.Variable((N, T + 1), name="ev_soc")

    # PV allocation flows (all ≥0)
    pv_to_load = cp.Variable((N, T), name="pv_load")
    pv_to_bess = cp.Variable((N, T), name="pv_bess")
    pv_to_ev   = cp.Variable((N, T), name="pv_ev")
    pv_curt    = cp.Variable((N, T), name="pv_curt")

    # Grid power (signed: + import, − export)
    grid_power = cp.Variable((N, T), name="grid_power")

    # ---------------------------------------------------------------- constraints
    c = []

    # 1) Bus balance (eq. 1)
    c += [grid_power + bess_power + ev_power + demand_power - pv_power == 0]

    # 2)-5) PV hierarchy (eqs. 2-5)
    c += [pv_to_load <= pv_power,
          pv_to_load <= demand_power]

    c += [pv_to_bess <= pv_power - pv_to_load,
          pv_to_bess <= bess_charge]

    c += [pv_to_ev   <= pv_power - pv_to_load - pv_to_bess,
          pv_to_ev   <= ev_charge]

    c += [pv_curt == pv_power - pv_to_load - pv_to_bess - pv_to_ev,
          pv_curt >= 0]

    # Non-negativity already implicit except for ev/bess dis
    c += [var >= 0 for var in (bess_charge, bess_discharge, ev_charge, ev_discharge)]

    # 6)-7) Device power limits (signed form via split variables)
    c += [bess_charge  <= p_bess_max[:, None],
          bess_discharge  <= p_bess_max[:, None]]

    for i in range(N):
        mask   = ev_present_matrix[i]
        p_max  = ev_data.at[i, 'max_power_mw']
        p_min  = ev_data.at[i, 'min_power_mw']   # negative
        c += [
            ev_charge[i] <=  p_max * mask,
            ev_discharge[i] <= -p_min * mask
        ]

    # 8)-9) Energy dynamics
    c += [bess_soc[:, 0] == soc0_bess]
    c += [bess_soc[:, 1:] == bess_soc[:, :-1] +
          (eta_bc * bess_charge - bess_discharge / eta_bd) * dt / cap_bess[:, None]]

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
                (config.EV_CHARGING_EFFICIENCY * ev_charge[i] -
                 ev_discharge[i] / config.EV_DISCHARGING_EFFICIENCY) * dt / cap_ev,

            ev_soc[i, 1:] >= config.EV_MIN_SOC * mask,
            ev_soc[i, 1:] <= config.EV_MAX_SOC * mask + 1 - mask,
            ev_soc[i, dep_idx] >= target
        ]

    # ---------------------------------------------------------------- objective (eq. 12)
    total_grid = cp.sum(grid_power, axis=0)          # MW per time step
    cost       = cp.sum(cp.multiply(prices, total_grid) * dt)

    prob = cp.Problem(cp.Minimize(cost), c)

    # ---------------------------------------------------------------- solve
    try:
        prob.solve(solver="HiGHS", verbose=True)
    except cp.error.SolverError:
        prob.solve(verbose=True)   # fall back to any installed solver

    # toc = time.time()
    # print(f"Optimisation finished in {toc - tic:.2f} s – status: {prob.status}")

    return (
        ev_power.value,
        bess_power.value,
        grid_power.value,
        bess_soc.value,
        ev_soc.value,
        prob.value,
        prob.status
    )
