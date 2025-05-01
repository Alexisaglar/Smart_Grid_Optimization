import numpy as np
import pandas as pd
import pandapower as pp
import cvxpy as cp # Add this import

import config
import grid_setup
import component_models
import optimization
# import results # Assuming a results.py for plotting/saving

def run_simulation():
    """
    Runs the VPP simulation and optimization.
    """
    # --- Setup ---
    net = grid_setup.setup_pandapower_network()
    time_steps = range(config.NUM_TIME_STEPS)

    # Generate EV data
    ev_data, ev_present_matrix = component_models.generate_ev_data(
        config.NUM_PARKING_LOTS,
        config.NUM_EVS_PER_LOT,
        config.NUM_TIME_STEPS
    )

    # Calculate PV power forecasts for all lots
    pv_power_forecasts = np.zeros((config.NUM_PARKING_LOTS, config.NUM_TIME_STEPS))
    for lot_id in range(config.NUM_PARKING_LOTS):
        pv_power_forecasts[lot_id, :] = component_models.calculate_pv_power_mw(time_steps, lot_id)

    # --- Optimization ---
    (solved_ev_power, solved_bess_power, solved_bess_soc,
     solved_total_vpp_power, objective_value, status) = optimization.solve_VPP_optimization(
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

    # Optional: Run pandapower power flow for each time step for verification
    # This checks if the grid constraints are met given the optimized VPP schedules
    print("\nRunning pandapower power flow for verification (optional)...")
    hourly_results_pf = []
    for t in time_steps:
        # Update loads based on profile
        net.load.p_mw = net.load.original_p_mw * config.LOAD_PROFILE[t]
        net.load.q_mvar = net.load.original_q_mvar * config.LOAD_PROFILE[t] # Assuming same profile for Q

        # Update VPP injections/loads at parking lot buses
        # Remove previous VPP elements if they exist
        if 'vpp_load_gen' in net:
            pp.drop_elements(net, net.vpp_load_gen.index, et='sgen')
            pp.drop_elements(net, net.vpp_load_gen.index, et='load')
            del net.vpp_load_gen # Clear tracking table

        vpp_elements = []
        for lot_id in range(config.NUM_PARKING_LOTS):
            bus_idx = config.PARKING_LOT_BUS_INDICES[lot_id]
            vpp_power_mw = solved_total_vpp_power[lot_id, t]

            # Represent net VPP power: positive = injection (sgen), negative = consumption (load)
            if vpp_power_mw >= 0:
                 idx = pp.create_sgen(net, bus=bus_idx, p_mw=vpp_power_mw, q_mvar=0, name=f"VPP_Lot_{lot_id}_T_{t}", type='vpp')
                 vpp_elements.append({'et': 'sgen', 'index': idx})
            else:
                 idx = pp.create_load(net, bus=bus_idx, p_mw=-vpp_power_mw, q_mvar=0, name=f"VPP_Lot_{lot_id}_T_{t}", type='vpp')
                 vpp_elements.append({'et': 'load', 'index': idx})

        # Store created elements for easy removal later
        if vpp_elements:
             net.vpp_load_gen = pd.DataFrame(vpp_elements)


        try:
            pp.runpp(net, algorithm='nr', calculate_voltage_angles=True)
            # Store relevant results (e.g., voltages, line loadings)
            hourly_results_pf.append({
                'time': t,
                'vm_pu': net.res_bus.vm_pu.copy(),
                'loading_percent': net.res_line.loading_percent.copy(),
                'ploss_mw': net.res_line.pl_mw.sum(),
                'qloss_mvar': net.res_line.ql_mvar.sum(),
                'grid_power_mw': net.res_ext_grid.p_mw.iloc[0] # Power from external grid
            })
            # Basic check
            max_loading = net.res_line.loading_percent.max()
            min_vm = net.res_bus.vm_pu.min()
            max_vm = net.res_bus.vm_pu.max()
            print(f"T={t}: PF Success. Grid P: {net.res_ext_grid.p_mw.iloc[0]:.2f} MW, Max Load: {max_loading:.1f}%, Vmin: {min_vm:.3f}, Vmax: {max_vm:.3f}")

        except pp.LoadflowNotConverged:
            print(f"Error: Power flow did not converge at time step {t}")
            hourly_results_pf.append({'time': t, 'status': 'PF Failed'})
            # break # Optional: stop simulation if PF fails

    # --- Display/Save Results ---
    # Add calls to results.py functions here
    # Example: results.plot_schedules(results_summary)
    # Example: results.save_results(results_summary, hourly_results_pf)
    print("\nSimulation complete.")


if __name__ == '__main__':
    run_simulation()

