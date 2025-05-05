# ---------------------------------------------------------------------------
# VPP scheduling – version: ONE EV  ↔  ONE BUS
# ---------------------------------------------------------------------------
import cvxpy as cp
import numpy as np
import pandas as pd
import utils.config as config            # or utils.config  – adapt to your repo
import time


def solve_VPP_optimisation(ev_data: pd.DataFrame,
                             ev_present_matrix: np.ndarray,
                             pv_power_forecasts: np.ndarray):
    """
    Schedules EVs+BESS, assuming *each EV sits on its own bus*
    (there is no parking-lot aggregation).
    Shapes:
        ev_data               : len = num_evs
        ev_present_matrix     : [num_evs, T]   0/1
        pv_power_forecasts    : [num_evs, T]   MW available on each bus
    Returns (tuple):
        ev_power, bess_power, bess_soc, bus_power, obj_value, status
        – every array has shape [num_evs, T] except SOC which is [num_evs,T+1]
    """

    # ------------------------------------------------------------------ data
    t0          = time.time()
    num_evs     = len(ev_data)                  # also #buses, #BESS
    T           = config.NUM_TIME_STEPS
    dt          = config.TIME_STEP_DURATION_H   # h

    prices      = config.ELECTRICITY_PRICE_PER_MWH      # [T]
    pv_power    = pv_power_forecasts                     # [num_evs, T]

    # BESS parameters – make sure they are arrays length num_evs
    cap         = np.broadcast_to(config.BESS_CAPACITY_MWH,  (num_evs,))
    p_max       = np.broadcast_to(config.BESS_MAX_POWER_MW,  (num_evs,))
    p_min       = np.broadcast_to(config.BESS_MIN_POWER_MW,  (num_evs,))  # positive!
    soc0        = np.broadcast_to(config.BESS_INITIAL_SOC,   (num_evs,))
    soc_min     = np.broadcast_to(config.BESS_MIN_SOC,       (num_evs,))
    soc_max     = np.broadcast_to(config.BESS_MAX_SOC,       (num_evs,))
    eta_c, eta_d = config.BESS_CHARGING_EFFICIENCY, config.BESS_DISCHARGING_EFFICIENCY

    # ---------------------------------------------------------------- vars
    # BESS
    bess_charge   = cp.Variable((num_evs, T), name="bess_chg")   # ≥0
    bess_discharge= cp.Variable((num_evs, T), name="bess_dis")   # ≥0
    bess_power    = bess_charge - bess_discharge                 # signed
    bess_soc      = cp.Variable((num_evs, T+1), name="bess_soc")

    # EV
    ev_charge     = cp.Variable((num_evs, T), name="ev_chg")     # ≥0
    ev_discharge  = cp.Variable((num_evs, T), name="ev_dis")     # ≥0
    ev_power      = ev_charge - ev_discharge                     # signed
    ev_soc        = cp.Variable((num_evs, T+1), name="ev_soc")

    # Bus net exchange with the grid
    bus_grid_power = cp.Variable((num_evs, T), name="bus_grid_p")  # +ve → grid buys

    # ----------------------------------------------------------- constraints
    c = []

    # --- non-negativity
    c += [var >= 0 for var in (bess_charge, bess_discharge,
                               ev_charge,   ev_discharge)]

    # --- BESS dynamics & limits
    c += [bess_soc[:, 0] == soc0]
    c += [bess_soc[:, 1:] == bess_soc[:, :-1] +
          (bess_charge * eta_c - bess_discharge / eta_d) * dt / cap[:, None]]
    c += [bess_soc >= soc_min[:, None],
          bess_soc <= soc_max[:, None],
          bess_charge   <= p_max[:, None],
          bess_discharge<= p_max[:, None]]           # discharge limit (same magnitude)

    # --- EV dynamics & limits  (one EV per row)
    c += [ev_soc[:, 0] == ev_data['soc_at_arrival'].values]
    for i in range(num_evs):
        cap_ev   = ev_data.at[i, 'capacity_mwh']
        p_max_ev = ev_data.at[i, 'max_power_mw']
        p_min_ev = ev_data.at[i, 'min_power_mw']     # negative for V2G
        target   = ev_data.at[i, 'target_soc']
        dep_idx  = int(ev_data.at[i, 'departure_time'])

        mask     = ev_present_matrix[i]              # 0/1 length T

        c += [
            ev_soc[i, 1:] == ev_soc[i, :-1] +
                (ev_charge[i] * config.EV_CHARGING_EFFICIENCY -
                 ev_discharge[i] / config.EV_DISCHARGING_EFFICIENCY) * dt / cap_ev,

            ev_charge[i]     <=  p_max_ev * mask,
            ev_discharge[i]  <= -p_min_ev * mask,    # p_min_ev should be −ve
            ev_soc[i, 1:]    >= config.EV_MIN_SOC    * mask,
            ev_soc[i, 1:]    <= config.EV_TARGET_SOC * mask + 1.0*(1-mask),
            ev_soc[i, dep_idx] >= target,
        ]

    # --- Power balance on every bus, every step
    c += [bus_grid_power == ev_power + bess_power - pv_power]

    # ------------------------------------------------------------- objective
    grid_total = cp.sum(bus_grid_power, axis=0)           # MW per time step
    cost       = cp.sum(cp.multiply(prices, grid_total))  # $
    prob       = cp.Problem(cp.Minimize(cost), c)

    # -------------------------------------------------------------- solve
    try:
        prob.solve(verbose=True)                          # let cvxpy pick solver
    except cp.error.SolverError as err:
        print("Solver failed:", err)
        return None

    # -------------------------------------------------------------- return
    return (ev_power.value,
            bess_power.value,
            bess_soc.value,
            bus_grid_power.value,
            prob.value,
            prob.status)
