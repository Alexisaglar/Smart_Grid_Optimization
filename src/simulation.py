import pandas as pd
import numpy as np
import pandapower as pp
import cvxpy as cp
import matplotlib.pyplot as plt
import seaborn as sns

from examples.config import LOAD_PROFILE
import src.grid_setup as grid_setup
import utils.config as config
import src.component_models as component_models
import src.optimisation as optimisation
import src.simple as simple
import utils.results as results

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
    pv_power_forecasts = np.zeros((33, 24))
    for bus_idx in config.PV_BUSES:
        pv_power_forecasts[bus_idx, :] = component_models.calculate_pv_power_mw(time_steps, bus_idx)

    bess_power, grid_import, grid_export, bess_soc, opt_val, status, pv_to_load, pv_to_bess, pv_to_ev, pv_curt = simple.solve_VPP_optimisation(
        pv_power_forecasts, config.LOAD_PROFILE 
    )

    # time_index = pd.date_range("2025-01-01",
    #                            periods=grid_export.shape[0],
    #                            freq=f"{config.TIME_STEP_DURATION_H}H")
    # plt.plot(grid_power, label='grid_energy')
    plt.plot(pv_curt[8,:], label='pv_curt')
    plt.plot(pv_to_load[8,:], label='pv_to_load')
    plt.plot(pv_to_bess[8,:], label='pv_to_bess')
    plt.plot(bess_soc[8,:], label='bess_soc')
    plt.plot(pv_power_forecasts[8,:], label='pv')
    plt.plot(np.full(24, 0))
    plt.legend()
    plt.show()


    print(f"Net system cost for the 24 h horizon:  €{opt_val:,.0f}")
    plot_vpp_results(
        time_steps=range(config.NUM_TIME_STEPS),
        grid_import=grid_import,          # (33, 24)
        grid_export=grid_export,          # (33, 24)
        bess_soc=bess_soc,                # (33, 24)
        bess_charge=np.maximum(0, bess_power),      # split ⇢ charge
        bess_discharge=np.maximum(0, -bess_power),  # split ⇢ discharge
        pv_to_load=pv_to_load,
        pv_to_bess=pv_to_bess,
        pv_to_grid=grid_export,           # by definition
        pv_curt=pv_curt
    )



def plot_vpp_results(time_steps,         # iterable of length T
                     grid_import,        # (B, T)  non-neg
                     grid_export,        # (B, T)  non-neg
                     bess_soc,           # (B, T)
                     bess_charge,        # (B, T)  non-neg
                     bess_discharge,     # (B, T)  non-neg
                     pv_to_load,         # (B, T)
                     pv_to_bess,         # (B, T)
                     pv_to_grid,         # (B, T)
                     pv_curt):           # (B, T)
    """
    Creates four figures that jointly describe the simulation outcome.
    """

    # ---- 1. system-wide import vs export ------------------------------------
    total_imp = grid_import.sum(axis=0)
    total_exp = grid_export.sum(axis=0)

    plt.figure(figsize=(10, 3))
    plt.plot(time_steps, total_imp,  label="Import  (+)", lw=1.8)
    plt.plot(time_steps, -total_exp, label="Export  (–)", lw=1.8)   # negative to show opposite dir
    plt.axhline(0, color='k', lw=.7)
    plt.ylabel("MW")
    plt.title("Grid exchange (system total)")
    plt.legend()
    plt.tight_layout()

    # ---- 2. battery SOC heat-map -------------------------------------------
    plt.figure(figsize=(11, 4))
    sns.heatmap(bess_soc,
                cmap="YlGnBu",
                cbar_kws=dict(label="SOC (p.u.)"),
                linewidths=.2,
                linecolor='grey')
    plt.ylabel("Bus")
    plt.xlabel("Hour")
    plt.title("Battery State of Charge by bus & hour")
    plt.tight_layout()

    # ---- 3. PV allocation stacked area --------------------------------------
    pv_load_tot  = pv_to_load.sum(axis=0)
    pv_bess_tot  = pv_to_bess.sum(axis=0)
    pv_grid_tot  = pv_to_grid.sum(axis=0)
    pv_curt_tot  = pv_curt.sum(axis=0)

    plt.figure(figsize=(10, 3.3))
    plt.stackplot(time_steps,
                  pv_load_tot,
                  pv_bess_tot,
                  pv_grid_tot,
                  pv_curt_tot,
                  labels=["PV → Load",
                          "PV → BESS charge",
                          "PV → Grid export",
                          "PV curtailed"])
    plt.legend(loc="upper left", ncol=2)
    plt.ylabel("MW")
    plt.title("Where does the PV go?")
    plt.tight_layout()

    # ---- 4. per-bus energy balance summary ----------------------------------
    # total energy over the horizon
    e_load       = pv_to_load.sum(axis=1)               # MWh equiv.
    e_bess_in    = bess_charge.sum(axis=1)
    e_bess_out   = bess_discharge.sum(axis=1)
    e_export     = pv_to_grid.sum(axis=1) + grid_export.sum(axis=1)
    e_import     = grid_import.sum(axis=1)

    ind = np.arange(bess_soc.shape[0])  # buses 0…32

    plt.figure(figsize=(11, 4))
    plt.bar(ind,  e_import,                       label="Import")
    plt.bar(ind, -e_export,   bottom=e_import*0,  label="Export")  # plotted downward
    plt.bar(ind,  e_bess_in,  bottom=e_import,    label="BESS chg")
    plt.bar(ind, -e_bess_out, bottom=-e_export,   label="BESS dis")
    plt.xlabel("Bus")
    plt.ylabel("MWh over horizon")
    plt.title("Energy balance by bus (positive = into bus)")
    plt.legend(ncol=4)
    plt.tight_layout()

    plt.show()

if __name__ == '__main__':
    run_simulation()
