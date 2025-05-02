import pandas as pd
import numpy as np
import pandapower as pp
import cvxpy as cp

import src.grid_setup as grid_setup
import utils.config as config
import src.component_models as component_models
import src.optimisation as optimisation

def run_simulation():
    """
    Runs the Smart Grid Optimisation simulation.
    """
    net = grid_setup.setup_pandapower_network()
    time_steps = range(config.NUM_TIME_STEPS)

    # Generate EV Data
    ev_data, ev_present_matrix = component_models.generate_ev_data(
        config.EV_INDICES,
        config.NUM_EVS_PER_LOT,
        config.NUM_TIME_STEPS
    )
    pv_power_forecasts = np.zeros((len(config.PV_BUSES), config.NUM_TIME_STEPS))
    for bus_idx in range(len(config.PV_BUSES)):
        pv_power_forecasts[bus_idx, :] = component_models.calculate_pv_power_mw(time_steps, bus_idx)

    # --- Optimization ---
    (solved_ev_power, solved_bess_power, solved_bess_soc,
     solved_total_vpp_power, objective_value, status) = optimisation.solve_VPP_optimisation(
         ev_data, ev_present_matrix, pv_power_forecasts
     )

    if status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        print("Exiting due to optimization failure.")
        return

    # --- Post-processing & Power Flow (Optional Verification) ---
    print("\n--- Simulation Results ---")
    print(f"Total Optimized Cost: {objective_value:.2f}")

    # Store results (example)
    results_summary = {
        'ev_power': solved_ev_power,
        'bess_power': solved_bess_power,
        'bess_soc': solved_bess_soc,
        'vpp_power': solved_total_vpp_power,
        'pv_power': pv_power_forecasts,
        'prices': config.ELECTRICITY_PRICE_PER_MWH,
        'ev_data': ev_data,
        'ev_presence': ev_present_matrix
    }



    print(pv_power_forecasts)

if __name__ == '__main__':
    run_simulation()
