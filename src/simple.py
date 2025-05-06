from time import time
import numpy as np
from pandapower import PV
import utils.config as config
import pandas as pd
import cvxpy as cp
import utils.results as results
def solve_VPP_optimisation(
    # ev_data          : pd.DataFrame,
    # ev_present_matrix: np.ndarray,
    pv_power_forecasts: np.ndarray,
    p_demand         : np.ndarray,
):
    # ---------------------------------------------------------------- parameters
    BUSES, TIME = 33, 24
    BESS_DEGRAD_COST_EUR_PER_MWH = np.full(BUSES, 15.0)   # or a 1-D vector per BESS
    time_step = config.TIME_STEP_DURATION_H
    electricity_prices = config.ELECTRICITY_PRICE_PER_MWH  # length T
    demand_power = np.full((BUSES, TIME), p_demand)
    cap_bess = config.BESS_CAPACITY_MWH
    p_bess_max = config.BESS_MAX_POWER_MW
    soc0_bess = config.BESS_INITIAL_SOC
    soc_bess_min = config.BESS_MIN_SOC
    soc_bess_max = config.BESS_MAX_SOC
    bess_charging_eff, bess_discharging_eff = config.BESS_CHARGING_EFFICIENCY, config.BESS_DISCHARGING_EFFICIENCY

    # ---------------------------------------------------------------- vars
    # BESS (split into non-neg parts for LP)
    bess_charge = cp.Variable((BUSES, TIME), name="bess_charge", nonneg=True)
    bess_discharge = cp.Variable((BUSES, TIME), name="bess_discharge", nonneg=True)
    bess_power   = bess_charge - bess_discharge
    bess_soc = cp.Variable((BUSES, TIME), name="bess_soc", nonneg=True)

    # PV allocation flows (all ≥0)
    pv_to_load = cp.Variable((BUSES, TIME), name="pv_load", nonneg=True)
    pv_to_bess = cp.Variable((BUSES, TIME), name="pv_bess", nonneg=True)
    pv_to_grid   = cp.Variable((BUSES, TIME), nonneg=True)   # export
    pv_to_ev   = cp.Variable((BUSES, TIME), name="pv_ev", nonneg=True)
    pv_curt    = cp.Variable((BUSES, TIME), name="pv_curt", nonneg=True)

    # Grid power (signed: + import, − export)
    grid_import  = cp.Variable((BUSES, TIME), nonneg=True)
    grid_export  = pv_to_grid                               # couple those if you like
    # total MWh cycled (charge + discharge) at each bus and time step
    bess_throughput = bess_charge + bess_discharge        # shape (BUSES, TIME)
    # scalar cost [€]
    deg_cost = cp.sum(cp.multiply(BESS_DEGRAD_COST_EUR_PER_MWH[:, None],
                              bess_throughput) * time_step)
    # ----------------------------------------------------------------
    # 2. Power-balance constraints (hierarchy enforced)
    # ----------------------------------------------------------------
    constraints = []

    # (2-A) Demand at each bus and time step must be covered:
    constraints += [
        pv_to_load + bess_discharge + grid_import == demand_power        # shape (BUSES, TIME)
    ]

    # (2-B) PV routed into the BESS can only come from on-site PV:
    constraints += [
        pv_to_bess == bess_charge                                        # pure PV→BESS path
    ]

    # (2-C) PV splitting equation (all PV must go somewhere):
    constraints += [
        pv_to_load + pv_to_bess + pv_to_grid + pv_curt == pv_power_forecasts
    ]

    # ----------------------------------------------------------------
    # 3. Upper bounds reflecting device and demand limits
    # ----------------------------------------------------------------
    constraints += [
        pv_to_load    <= pv_power_forecasts,
        pv_to_load    <= demand_power,

        pv_to_bess    <= pv_power_forecasts - pv_to_load,
        pv_to_bess    <= bess_charge,                     # charger rating already in BESS block

        pv_curt       >= 0,                              # already non-negative but explicit is fine
    ]

    # ----------------------------------------------------------------
    # 4. BESS power and energy limits
    # ----------------------------------------------------------------
    constraints += [
        bess_charge[config.PV_BUSES, :]    <= p_bess_max[:, None],   # charge limit
        bess_discharge[config.PV_BUSES, :] <= p_bess_max[:, None],   # discharge limit

        # State-of-charge dynamics
        bess_soc[config.PV_BUSES, 0] == soc0_bess,

        bess_soc[config.PV_BUSES, 1:] == bess_soc[config.PV_BUSES, :-1] +
            (bess_charging_eff * bess_charge[config.PV_BUSES, 1:]
             - bess_discharge[config.PV_BUSES, 1:] / bess_discharging_eff) * time_step / cap_bess[:, None],

        bess_soc[config.PV_BUSES, :] >= soc_bess_min[:, None],
        bess_soc[config.PV_BUSES, :] <= soc_bess_max[:, None],
    ]
    # Buses that have NO battery
    inactive_buses = np.setdiff1d(np.arange(BUSES), config.PV_BUSES)

    #  No charging, no discharging, no energy stored on those buses
    constraints += [
        bess_charge[inactive_buses, :]    == 0,
        bess_discharge[inactive_buses, :] == 0,
        bess_soc[inactive_buses, :]       == 0,
    ]
    # ----------------------------------------------------------------
    # 5. Grid import / export coupling (optional extras)
    # ----------------------------------------------------------------
    # If your market model requires a *net* power vector, do:
    # net_grid = grid_import - pv_to_grid       # shape (BUSES, TIME) or summed later
    # otherwise leave as is
    # positive numbers (€/MWh)
    import_tariff  = config.ELECTRICITY_PRICE_PER_MWH     # length TIME
    export_tariff  = config.ELECTRICITY_PRICE_PER_MWH        # length TIME or scalar

    total_import = cp.sum(grid_import, axis=0)            # length TIME
    total_export = cp.sum(pv_to_grid,  axis=0)            # length TIME

    energy_cost  = cp.sum(cp.multiply(import_tariff, total_import) * time_step)
    export_rev   = cp.sum(cp.multiply(export_tariff, total_export) * time_step)

    objective = cp.Minimize(energy_cost - export_rev + deg_cost)
    prob      = cp.Problem(objective, constraints)

    # # ----------------------------------------------------------------
    # # … then build the objective and solve as usual
    # prob = cp.Problem(cp.Minimize(import_cost - export_revenue), constraints)

    # ---------------------------------------------------------------- solve
    try:
        prob.solve(solver="HiGHS", verbose=True)
    except cp.error.SolverError:
        prob.solve(verbose=True)   # fall back to any installed solver

    # toc = time.time()
    # print(f"Optimisation finished in {toc - tic:.2f} s – status: {prob.status}")

    return (
        bess_power.value,
        grid_import.value,
        grid_export.value,
        bess_soc.value,
        prob.value,
        prob.status,
        pv_to_load.value,
        pv_to_bess.value,
        pv_to_ev.value,
        pv_curt.value,
    )
