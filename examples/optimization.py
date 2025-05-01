import cvxpy as cp
import numpy as np
import pandas as pd
import config
import time

def solve_VPP_optimization(ev_data, ev_present_matrix, pv_power_forecasts):
    """
    Solves the VPP optimization problem for EV and BESS scheduling.
    Args:
        ev_data (pd.DataFrame): DataFrame with EV parameters and schedules.
        ev_present_matrix (np.array): Boolean matrix [num_evs, num_time_steps] indicating EV presence.
        pv_power_forecasts (np.array): Array [num_lots, num_time_steps] of available PV power (MW).
    Returns:
        tuple: Solved EV power (MW), BESS power (MW), BESS SOC, total VPP power (MW), objective value, status.
               Returns None for variables if optimization fails.
    """
    start_time = time.time()
    num_lots = config.NUM_PARKING_LOTS
    num_evs = len(ev_data)
    T = config.NUM_TIME_STEPS
    time_steps = range(T)
    dt = config.TIME_STEP_DURATION_H # Time step duration in hours

    # --- Decision Variables ---
    # BESS power (positive for charging, negative for discharging)
    bess_power = cp.Variable((num_lots, T), name="bess_power")
    # Introduce separate charge/discharge variables for BESS
    bess_charge_power = cp.Variable((num_lots, T), name="bess_charge_power")
    bess_discharge_power = cp.Variable((num_lots, T), name="bess_discharge_power")
    bess_soc = cp.Variable((num_lots, T + 1), name="bess_soc")

    # EV power (positive for charging, negative for discharging)
    ev_power = cp.Variable((num_evs, T), name="ev_power")
    # Introduce separate charge/discharge variables for EV
    ev_charge_power = cp.Variable((num_evs, T), name="ev_charge_power")
    ev_discharge_power = cp.Variable((num_evs, T), name="ev_discharge_power")
    ev_soc = cp.Variable((num_evs, T + 1), name="ev_soc")

    # Total power exchanged by each VPP with the grid
    total_vpp_power_per_lot = cp.Variable((num_lots, T), name="total_vpp_power")

    # --- Parameters ---
    prices = config.ELECTRICITY_PRICE_PER_MWH # $/MW per time step
    pv_power = pv_power_forecasts # MW

    # BESS parameters
    bess_capacity_mwh = config.BESS_CAPACITY_MWH
    bess_max_power_mw = config.BESS_MAX_POWER_MW
    bess_min_power_mw = config.BESS_MIN_POWER_MW
    bess_initial_soc = config.BESS_INITIAL_SOC
    bess_min_soc = config.BESS_MIN_SOC
    bess_max_soc = config.BESS_MAX_SOC
    bess_eff_c = config.BESS_CHARGING_EFFICIENCY
    bess_eff_d = config.BESS_DISCHARGING_EFFICIENCY

    # --- Constraints ---
    constraints = []

    # Link net power to charge/discharge power
    constraints += [bess_power == bess_charge_power - bess_discharge_power]
    constraints += [ev_power == ev_charge_power - ev_discharge_power]

    # Non-negativity for charge/discharge power
    constraints += [bess_charge_power >= 0]
    constraints += [bess_discharge_power >= 0]
    constraints += [ev_charge_power >= 0]
    constraints += [ev_discharge_power >= 0]

    # BESS Constraints
    constraints += [bess_soc[:, 0] == bess_initial_soc] # Initial SOC
    constraints += [bess_soc[:, 1:] == bess_soc[:, :-1] + (bess_charge_power * bess_eff_c - bess_discharge_power / bess_eff_d) * dt / bess_capacity_mwh[:, np.newaxis]] # SOC update
    constraints += [bess_soc >= bess_min_soc]
    constraints += [bess_soc <= bess_max_soc]
    constraints += [bess_charge_power <= bess_max_power_mw[:, np.newaxis]] # Charge power limits
    constraints += [bess_discharge_power <= -bess_min_power_mw[:, np.newaxis]] # Discharge power limits

    # EV Constraints
    constraints += [ev_soc[:, 0] == ev_data['initial_soc'].values] # Initial SOC
    for ev_idx in range(num_evs):
        ev = ev_data.iloc[ev_idx]
        lot_id = ev['lot_id']
        capacity = ev['capacity_mwh']
        max_p = ev['max_power_mw']
        min_p = ev['min_power_mw'] # Negative for V2G
        target_soc = ev['target_soc']
        dep_time_idx = int(ev['departure_time']) # Ensure dep_time_idx is integer

        constraints += [
            ev_soc[ev_idx, 1:] == ev_soc[ev_idx, :-1] + (
                ev_charge_power[ev_idx, :] * config.EV_CHARGING_EFFICIENCY - 
                ev_discharge_power[ev_idx, :] / config.EV_DISCHARGING_EFFICIENCY
            ) * dt / capacity,
            ev_charge_power[ev_idx, :] <= max_p * ev_present_matrix[ev_idx, :],
            ev_discharge_power[ev_idx, :] <= -min_p * ev_present_matrix[ev_idx, :],
            ev_charge_power[ev_idx, :] >= 0,
            ev_discharge_power[ev_idx, :] >= 0,
            ev_soc[ev_idx, 1:] >= config.EV_MIN_SOC * ev_present_matrix[ev_idx, :],
            ev_soc[ev_idx, 1:] <= config.EV_TARGET_SOC * ev_present_matrix[ev_idx, :] + 1.0 * (1 - ev_present_matrix[ev_idx, :]),
            ev_soc[ev_idx, dep_time_idx] >= target_soc,
        ]

    # --- Objective Function ---
    for lot in range(num_lots):
        evs_in_lot = ev_data[ev_data['lot_id'] == lot].index
        if len(evs_in_lot) > 0:
             constraints += [total_vpp_power_per_lot[lot,:] == cp.sum(ev_power[evs_in_lot, :], axis=0) + bess_power[lot,:] - pv_power[lot,:]]
        else:
             constraints += [total_vpp_power_per_lot[lot,:] == bess_power[lot,:] - pv_power[lot,:]]

    total_grid_exchange_power = cp.sum(total_vpp_power_per_lot, axis=0)
    objective = cp.Minimize(cp.sum(cp.multiply(prices, total_grid_exchange_power)))

    # --- Solve Problem ---
    problem = cp.Problem(objective, constraints)
    print("Solving optimization problem...")
    objective_value = None # Initialize objective value
    solver_options = {}
    problem_status = None # Initialize problem status

    try:
        # Enable verbose output and pass solver-specific options
        objective_value = problem.solve(solver='ECOS', verbose=True, **solver_options)
        problem_status = problem.status # Store the status
    except cp.error.SolverError as e:
        print(f"Solver Error encountered: {e}")
        problem_status = "Solver Error" # Assign custom status
    except Exception as e: # Catch other potential errors
        print(f"An unexpected error occurred during optimization: {e}")
        problem_status = "Unexpected Error" # Assign custom status

    end_time = time.time()
    print(f"Optimization finished in {end_time - start_time:.2f} seconds.")
    # Check status more robustly
    if objective_value is None or problem_status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        print(f"--- Optimization FAILED ---")
        print(f"Solver Status: {problem_status}")
        print(f"Objective Value: {objective_value}")
        print("Potential causes: Infeasible problem (conflicting constraints) or Unbounded problem.")
        print("Check solver output above for detailed messages (e.g., 'primal infeasible', 'dual infeasible').")
        print("Verify constraints (SOC limits, power limits, EV arrival/departure/target SOC) and input data.")
        # Return None values indicating failure
        return None, None, None, None, objective_value, problem_status
    else:
        print(f"--- Optimization SUCCEEDED ---")
        print(f"Solver Status: {problem_status}")
        print(f"Optimal objective value: {objective_value}")
        solved_ev_power = ev_power.value
        solved_bess_power = bess_power.value
        solved_bess_soc = bess_soc.value
        solved_total_vpp_power = total_vpp_power_per_lot.value
        return solved_ev_power, solved_bess_power, solved_bess_soc, solved_total_vpp_power, objective_value, problem_status

