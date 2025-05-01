import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import config

def plot_schedules(results):
    """Plots key results like BESS SOC, power schedules, and prices."""
    num_lots = config.NUM_PARKING_LOTS
    T = config.NUM_TIME_STEPS
    time_axis = np.arange(T)
    time_axis_soc = np.arange(T + 1)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Plot Prices and Total VPP Power
    axes[0].plot(time_axis, results['prices'], 'r-o', label='Price ($/MW/step)')
    total_vpp_power = np.sum(results['vpp_power'], axis=0)
    axes[0].bar(time_axis, total_vpp_power, alpha=0.6, label='Total VPP Power (MW)')
    axes[0].set_ylabel('Price / Power')
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_title('Electricity Price and Net VPP Power Exchange')

    # Plot BESS SOC for each lot
    for lot_id in range(num_lots):
        axes[1].plot(time_axis_soc, results['bess_soc'][lot_id, :], '-o', markersize=3, label=f'BESS Lot {lot_id} SOC')
    axes[1].set_ylabel('BESS SOC')
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True)

    # Plot Aggregated Power Components for the first lot (example)
    lot_id_to_plot = 0
    evs_in_lot = results['ev_data'][results['ev_data']['lot_id'] == lot_id_to_plot].index
    agg_ev_power = np.sum(results['ev_power'][evs_in_lot, :], axis=0) if len(evs_in_lot) > 0 else np.zeros(T)

    axes[2].plot(time_axis, results['pv_power'][lot_id_to_plot, :], 'g-^', label=f'PV Lot {lot_id_to_plot} (MW)')
    axes[2].plot(time_axis, results['bess_power'][lot_id_to_plot, :], 'b-s', label=f'BESS Lot {lot_id_to_plot} (MW)')
    axes[2].plot(time_axis, agg_ev_power, 'm-x', label=f'Agg. EV Lot {lot_id_to_plot} (MW)')
    axes[2].plot(time_axis, results['vpp_power'][lot_id_to_plot, :], 'k--', label=f'Net VPP Lot {lot_id_to_plot} (MW)')
    axes[2].set_ylabel('Power (MW)')
    axes[2].set_xlabel('Time Step (hour)')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

def save_results(results_summary, pf_results=None, filename="simulation_results.xlsx"):
    """Saves key results to an Excel file."""
    with pd.ExcelWriter(filename) as writer:
        pd.DataFrame(results_summary['prices'], columns=['ElectricityPrice']).to_excel(writer, sheet_name='Prices')
        pd.DataFrame(results_summary['pv_power'].T, columns=[f'PV_Lot_{i}' for i in range(config.NUM_PARKING_LOTS)]).to_excel(writer, sheet_name='PV_Power_MW')
        pd.DataFrame(results_summary['bess_power'].T, columns=[f'BESS_Power_Lot_{i}' for i in range(config.NUM_PARKING_LOTS)]).to_excel(writer, sheet_name='BESS_Power_MW')
        pd.DataFrame(results_summary['bess_soc'].T, columns=[f'BESS_SOC_Lot_{i}' for i in range(config.NUM_PARKING_LOTS)]).to_excel(writer, sheet_name='BESS_SOC')
        pd.DataFrame(results_summary['vpp_power'].T, columns=[f'VPP_Power_Lot_{i}' for i in range(config.NUM_PARKING_LOTS)]).to_excel(writer, sheet_name='VPP_Power_MW')
        results_summary['ev_data'].to_excel(writer, sheet_name='EV_Data')
        # Add EV power saving if needed (can be large)
        # pd.DataFrame(results_summary['ev_power']).to_excel(writer, sheet_name='EV_Power_MW')

        if pf_results:
             pf_df = pd.DataFrame(pf_results)
             pf_df.to_excel(writer, sheet_name='PowerFlow_Summary')

    print(f"Results saved to {filename}")

# Add plotting/saving calls in simulation.py after results are generated
# Example in simulation.py:
# import results
# ... (after optimization)
# results.plot_schedules(results_summary)
# results.save_results(results_summary, hourly_results_pf)

