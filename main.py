import pyomo.environ as pyo
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as nw
import matplotlib.pyplot as plt
import seaborn as sns # Add this import to the top of your file

# --- Placeholder classes (unchanged) ---
class PvSystem:
    def __init__(self, network, bus_idx, pv_parameters, forecast, name):
        self.network = network
        self.bus_idx = bus_idx
        self.pv_parameters = pv_parameters
        self.forecast = forecast
        self.name = name
        self.pv_idx = bus_idx

class BatterySystem:
    def __init__(self, network, bus_idx, capacity_mwh, charge_efficiency, discharge_efficiency, max_p_mw, initial_soc_percent, name, replacement_cost, lifetime_throughput):
            self.network = network
            self.bus_idx = bus_idx
            self.capacity_mwh = capacity_mwh
            self.charge_efficiency = charge_efficiency
            self.discharge_efficiency = discharge_efficiency
            self.max_p_mw = max_p_mw
            self.initial_soc = initial_soc_percent
            self.name = name
            self.replacement_cost = replacement_cost
            self.lifetime_throughput = lifetime_throughput

def calculate_heuristic_schedule(
    pv_objects: list,
    bess_objects: list,
    load_forecasts: np.array,
) -> pd.DataFrame:
    """
    Simulates a greedy self-consumption heuristic for BESS dispatch.
    """
    # --- Setup ---
    pv_forecasts = {pv.name: pv.forecast for pv in pv_objects}
    bess_map = {b.name: b for b in bess_objects}
    bess_names = list(bess_map.keys())
    
    total_pv_forecast = sum(pv_forecasts.values())
    net_load = load_forecasts - total_pv_forecast.copy() # Use a copy to avoid modifying the original

    # --- Initialize results arrays ---
    p_bess_net = {name: np.zeros(24) for name in bess_names}
    soc_bess = {name: np.zeros(24) for name in bess_names}

    for name, obj in bess_map.items():
        soc_bess[name][0] = obj.initial_soc

    # --- Main Loop: Iterate through each hour ---
    for t in range(24):
        if t > 0:
            for name, obj in bess_map.items():
                power_prev, soc_prev = p_bess_net[name][t-1], soc_bess[name][t-1]
                if power_prev < 0: # Was charging
                    soc_change = (-power_prev * obj.charge_efficiency) / obj.capacity_mwh * 100
                    soc_bess[name][t] = soc_prev + soc_change
                else: # Was discharging
                    soc_change = (power_prev / obj.discharge_efficiency) / obj.capacity_mwh * 100
                    soc_bess[name][t] = soc_prev - soc_change
        
        for name, obj in bess_map.items():
            current_soc = soc_bess[name][t]
            if net_load[t] > 0: # Power deficit
                power_needed = net_load[t]
                available_energy_mwh = ((current_soc - 20) / 100) * obj.capacity_mwh
                power_from_energy = (available_energy_mwh / 1) * obj.discharge_efficiency
                discharge_power = min(power_needed, obj.max_p_mw, power_from_energy)
                p_bess_net[name][t] = discharge_power
                net_load[t] -= discharge_power
            elif net_load[t] < 0: # Power surplus
                power_surplus = -net_load[t]
                available_space_mwh = ((90 - current_soc) / 100) * obj.capacity_mwh
                power_to_space = (available_space_mwh / 1) / obj.charge_efficiency
                charge_power = min(power_surplus, obj.max_p_mw, power_to_space)
                p_bess_net[name][t] = -charge_power
                net_load[t] += charge_power

    # --- CORRECTED RESULTS FORMATTING ---
    res = {'load': load_forecasts}
    # Add individual PV forecasts to the results, matching the optimizer's format
    for pv_name, forecast in pv_forecasts.items():
        res[f'pv_{pv_name}'] = forecast
        
    for name in bess_names:
        res[f'p_{name}'] = p_bess_net[name]
        res[f'soc_{name}'] = soc_bess[name]
    
    heuristic_df = pd.DataFrame(res)
    
    # Calculate grid power as the final balancing residual
    pv_total = heuristic_df[[col for col in heuristic_df.columns if 'pv_' in col]].sum(axis=1)
    bess_total = heuristic_df[[col for col in heuristic_df.columns if 'p_bess' in col]].sum(axis=1)
    final_net_power = heuristic_df['load'] - pv_total - bess_total
    heuristic_df['grid_import'] = final_net_power.clip(lower=0)
    heuristic_df['grid_export'] = -final_net_power.clip(upper=0)
    
    return heuristic_df

def calculate_full_schedule(
    net: pp.pandapowerNet,
    pv_objects: list,
    bess_objects: list,
    grid_price: list,
    forecast_range: int,
    load_forecasts: np.array,
    enable_dr: bool = True,
) -> pd.DataFrame:
    # Data Mapping---
    pv_map = {pv.name: pv for pv in pv_objects}
    pv_forecasts = {pv.name: pv.forecast for pv in pv_objects}
    pv_bus_map = {pv.name: pv.bus_idx for pv in pv_objects}
    bess_map = {b.name: b for b in bess_objects}
    bess_bus_map = {b.name: b.bus_idx for b in bess_objects}
    lines, buses, slack_bus_id = net.line, net.bus, net.ext_grid.bus.iloc[0]
    base_loads_p = net.load.set_index('bus')['p_mw']
    base_loads_q = net.load.set_index('bus')['q_mvar']

    # Model
    model = pyo.ConcreteModel("Full Coordinated Dispatch")

    # Sets
    model.T = pyo.RangeSet(0, forecast_range - 1)
    model.PV_IDs = pyo.Set(initialize=pv_map.keys())
    model.BESS_IDs = pyo.Set(initialize=bess_map.keys())
    model.BUS_IDs = pyo.Set(initialize=buses.index)
    model.LINE_IDs = pyo.Set(initialize=lines.index)


    # Control Variables
    model.p_curt = pyo.Var(model.BUS_IDs, model.T, domain=pyo.NonNegativeReals)
    model.c_inc = pyo.Var(model.T, domain=pyo.NonNegativeReals) # Incentive rate in £/MWh
    model.p_grid = pyo.Var(model.T, domain=pyo.Reals)
    model.q_grid = pyo.Var(model.T, domain=pyo.Reals)
    model.p_flow = pyo.Var(model.T, model.LINE_IDs, domain=pyo.Reals)
    model.q_flow = pyo.Var(model.T, model.LINE_IDs, domain=pyo.Reals)
    model.v_sq = pyo.Var(model.T, model.BUS_IDs, domain=pyo.NonNegativeReals)
    model.p_charge = pyo.Var(model.BESS_IDs, model.T, domain=pyo.NonNegativeReals)
    model.p_discharge = pyo.Var(model.BESS_IDs, model.T, domain=pyo.NonNegativeReals)
    model.soc_bess = pyo.Var(model.BESS_IDs, model.T, domain=pyo.NonNegativeReals, bounds=(20, 90))



    # if we have demand response:
    if enable_dr:
        model.p_curt = pyo.Var(model.BUS_IDs, model.T, domain=pyo.NonNegativeReals)
        model.c_inc = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    else:
        # If DR is disabled, create an empty variable to avoid errors
        model.p_curt = pyo.Param(model.BUS_IDs, model.T, default=0)

    # Objective Function 
    def full_objective_rule(model):
        grid_cost = sum(model.p_grid[t] * grid_price[t] for t in model.T)
        degradation_cost = sum(
            (bess_map[b_name].replacement_cost / bess_map[b_name].lifetime_throughput) * model.p_discharge[b_name, t]
            for b_name in model.BESS_IDs for t in model.T
        )
        
        incentive_cost = 0
        if enable_dr:
            incentive_cost = sum(
                model.c_inc[t] * model.p_curt[i, t]
                for i in model.BUS_IDs if i in base_loads_p.index for t in model.T
            )
        return grid_cost + degradation_cost + incentive_cost
    model.objective = pyo.Objective(rule=full_objective_rule, sense=pyo.minimize)


    # Constraints
    def active_power_balance_rule(model, t, i):
        flow_out = sum(model.p_flow[t, l] for l in model.LINE_IDs if lines.from_bus[l] == i)
        flow_in = sum(model.p_flow[t, l] for l in model.LINE_IDs if lines.to_bus[l] == i)
        pv_gen = sum(pv_forecasts[pv_id][t] for pv_id in model.PV_IDs if pv_bus_map[pv_id] == i)
        bess_dispatch = sum(model.p_discharge[b, t] - model.p_charge[b, t] for b in model.BESS_IDs if bess_bus_map[b] == i)
        grid_inj = model.p_grid[t] if i == slack_bus_id else 0
        load_p = base_loads_p.get(i, 0) * (load_forecasts / load_forecasts.max())[t]
        base_load_p = base_loads_p.get(i, 0) * (load_forecasts / load_forecasts.max())[t]
        curtailed_load_p = model.p_curt[i, t]
        actual_load_p = base_load_p - curtailed_load_p
        return (pv_gen + bess_dispatch + grid_inj + flow_in - flow_out - actual_load_p) == 0
    model.active_power_balance = pyo.Constraint(model.T, model.BUS_IDs, rule=active_power_balance_rule)

    if enable_dr:
        # Max curtailment per hour
        def max_curtailment_rule(model, i, t):
            if i not in base_loads_p.index:
                return model.p_curt[i, t] == 0 # No load to curtail at this bus
            base_load_p = base_loads_p.get(i, 0) * (load_forecasts / load_forecasts.max())[t]
            return model.p_curt[i, t] <= 0.60 * base_load_p
        model.max_curtailment_constraint = pyo.Constraint(model.BUS_IDs, model.T, rule=max_curtailment_rule)

        # Total energy curtailment limit over the horizon
        def total_energy_curtailment_rule(model, i):
            if i not in base_loads_p.index:
                return pyo.Constraint.Skip # Skip buses with no load
            total_base_energy = sum(base_loads_p.get(i, 0) * (load_forecasts / load_forecasts.max())[t] for t in model.T)
            total_curtailed_energy = sum(model.p_curt[i, t] for t in model.T)
            return total_curtailed_energy <= 0.40 * total_base_energy
        model.total_energy_curtailment_constraint = pyo.Constraint(model.BUS_IDs, rule=total_energy_curtailment_rule)

        # Consumer benefit must be positive
        def consumer_benefit_rule(model, i):
            if i not in base_loads_p.index:
                return pyo.Constraint.Skip
            # Assuming beta and xi are uniform for all consumers for this example
            beta = 0.1 
            xi = 0.5 
            
            incentive_earned = sum(xi * model.c_inc[t] * model.p_curt[i, t] for t in model.T)
            discomfort_cost = sum((1 - xi) * beta * (model.p_curt[i, t]**2) for t in model.T)
            
            return incentive_earned >= discomfort_cost
        model.consumer_benefit_constraint = pyo.Constraint(model.BUS_IDs, rule=consumer_benefit_rule)

        # Incentive rate bounds
        min_price = grid_price.min()
        max_price = grid_price.max()
        def incentive_bounds_rule(model, t):
            return pyo.inequality(0.5 * min_price, model.c_inc[t], max_price)
        model.incentive_bounds = pyo.Constraint(model.T, rule=incentive_bounds_rule)

    def soc_evolution_rule(model, b_name, t):
        bess_obj = bess_map[b_name]
        if t == 0: return model.soc_bess[b_name, t] == bess_obj.initial_soc
        soc_prev = model.soc_bess[b_name, t-1]
        charge_power = model.p_charge[b_name, t-1]; discharge_power = model.p_discharge[b_name, t-1]
        soc_change_charge = (charge_power * bess_obj.charge_efficiency) / bess_obj.capacity_mwh * 100
        soc_change_discharge = (discharge_power / bess_obj.discharge_efficiency) / bess_obj.capacity_mwh * 100
        return model.soc_bess[b_name, t] == soc_prev + soc_change_charge - soc_change_discharge
    model.soc_evolution = pyo.Constraint(model.BESS_IDs, model.T, rule=soc_evolution_rule)
    def charge_limit_rule(model, b_name, t): return model.p_charge[b_name, t] <= bess_map[b_name].max_p_mw
    model.charge_limit = pyo.Constraint(model.BESS_IDs, model.T, rule=charge_limit_rule)
    def discharge_limit_rule(model, b_name, t): return model.p_discharge[b_name, t] <= bess_map[b_name].max_p_mw
    model.discharge_limit = pyo.Constraint(model.BESS_IDs, model.T, rule=discharge_limit_rule)

    # --- NEW CYCLICAL SOC CONSTRAINT ---
    def soc_cyclical_rule(model, b_name):
        # The SoC at the last hour must be >= the SoC at the first hour.
        return model.soc_bess[b_name, forecast_range - 1] >= model.soc_bess[b_name, 0]
    model.soc_cyclical_constraint = pyo.Constraint(model.BESS_IDs, rule=soc_cyclical_rule)
    # ------------------------------------

    # (Network constraints remain the same)
    def reactive_power_balance_rule(model, t, i): # ... unchanged
        flow_out_q = sum(model.q_flow[t, l] for l in model.LINE_IDs if lines.from_bus[l] == i); flow_in_q = sum(model.q_flow[t, l] for l in model.LINE_IDs if lines.to_bus[l] == i)
        grid_inj_q = model.q_grid[t] if i == slack_bus_id else 0; load_q = base_loads_q.get(i, 0) * (load_forecasts / load_forecasts.max())[t]
        return (grid_inj_q + flow_in_q - flow_out_q - load_q) == 0
    model.reactive_power_balance = pyo.Constraint(model.T, model.BUS_IDs, rule=reactive_power_balance_rule)
    def voltage_drop_rule(model, t, l):
        from_bus = lines.at[l, 'from_bus']; to_bus = lines.at[l, 'to_bus']; r = lines.at[l, 'r_ohm_per_km'] * lines.at[l, 'length_km']
        x = lines.at[l, 'x_ohm_per_km'] * lines.at[l, 'length_km']; v_base_kv = buses.at[from_bus, 'vn_kv']
        return model.v_sq[t, to_bus] == model.v_sq[t, from_bus] - (2 / (v_base_kv**2)) * (r * model.p_flow[t, l] + x * model.q_flow[t, l])
    model.voltage_drop = pyo.Constraint(model.T, model.LINE_IDs, rule=voltage_drop_rule)
    def voltage_limit_rule(model, t, i): 
        return pyo.inequality(0.95**2, model.v_sq[t, i], 1.05**2)
    model.voltage_limits = pyo.Constraint(model.T, model.BUS_IDs, rule=voltage_limit_rule)
    def slack_voltage_rule(model, t): 
        return model.v_sq[t, slack_bus_id] == 1.0**2
    model.slack_voltage = pyo.Constraint(model.T, rule=slack_voltage_rule)

    # --- Solve and Process Results ---
    solver = pyo.SolverFactory('ipopt')
    results = solver.solve(model)

    # --- 6. Process Results ---
    if (results.solver.status == pyo.SolverStatus.ok) and (results.solver.termination_condition == pyo.TerminationCondition.optimal):
        print("\nOptimal coordinated schedule found!")
        
        # (The code to process `res` for the schedule_df is the same)
        grid_power_vals = [pyo.value(model.p_grid[t]) for t in model.T]
        res = {'load': load_forecasts, 'grid_price': grid_price, 'grid_import': [max(0, p) for p in grid_power_vals], 'grid_export': [-min(0, p) for p in grid_power_vals]}
        for pv_id, forecast in pv_forecasts.items(): res[f'pv_{pv_id}'] = forecast
        for b_name in model.BESS_IDs:
            p_charge_vals = [pyo.value(model.p_charge[b_name, t]) for t in model.T]; p_discharge_vals = [pyo.value(model.p_discharge[b_name, t]) for t in model.T]
            res[f'p_{b_name}'] = np.array(p_discharge_vals) - np.array(p_charge_vals)
            res[f'soc_{b_name}'] = [pyo.value(model.soc_bess[b_name, t]) for t in model.T]
        schedule_df = pd.DataFrame(res)

        # --- ADD THIS BLOCK TO EXTRACT VOLTAGE DATA ---
        voltage_data = {}
        for i in model.BUS_IDs:
            voltage_data[f'bus_{i}'] = [np.sqrt(pyo.value(model.v_sq[t, i])) for t in model.T]
        voltage_df = pd.DataFrame(voltage_data, index=model.T)
        # -----------------------------------------------

        return schedule_df, voltage_df # <-- RETURN BOTH DATAFRAMES

    else:
        print("\nSolver failed to find an optimal solution for the full model.")
        return None, None # <-- RETURN NONE FOR BOTH

# The plotting function and main execution block are unchanged
def plot_full_dispatch(df: pd.DataFrame): # ... unchanged
    pv_cols = [col for col in df.columns if 'pv_' in col]; bess_cols = [col for col in df.columns if col.startswith('p_bess')]
    df['total_pv'] = df[pv_cols].sum(axis=1); df['total_bess_p'] = df[bess_cols].sum(axis=1)
    df['bess_discharge'] = df['total_bess_p'].clip(lower=0); df['bess_charge'] = -df['total_bess_p'].clip(upper=0)
    df['curtailed_load'] = df['load'] - (df['total_pv'] + df['bess_discharge'] + df['grid_import'] - df['bess_charge'])
    hours = df.index
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax1.set_xlabel('Hour of the Day'); ax1.set_ylabel('Power (MW)')
     # Stacked area for met load
    ax1.stackplot(hours, df['total_pv'], df['bess_discharge'], df['grid_import'],
                  labels=['PV Generation', 'BESS Discharge', 'Grid Import'],
                  colors=['#FFD700', '#90EE90', '#ADD8E6'], alpha=0.7)
    # Stacked area for curtailed load on top
    ax1.stackplot(hours, df['load'] - df['curtailed_load'], df['curtailed_load'],
                  labels=['', 'Load Curtailed'], # Empty label to avoid duplicating
                  colors=['none', '#D3D3D3'], alpha=0.8)
    ax1.plot(hours, df['load'], label='Total Load', color='black', linewidth=2.5)
    ax1.tick_params(axis='y'); ax1.set_xlim(0, 23); ax1.grid(True, linestyle='--', alpha=0.6)
    ax2 = ax1.twinx()
    ax2.plot(hours, df['grid_price'], label='Grid Price', color='#8B0000', linestyle='--', marker='o', markersize=4)
    ax2.set_ylabel('Grid Price (£/MWh)', color='#8B0000'); ax2.tick_params(axis='y', labelcolor='#8B0000')
    soc_col = [col for col in df.columns if 'soc_' in col][0]
    ax2.plot(hours, df[soc_col], label='BESS SoC', color='purple', linestyle=':', linewidth=2.5)
    lines, labels = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left', ncol=2)
    plt.title('24-Hour Coordinated Dispatch (PV, BESS, & Grid)')
    plt.xticks(np.arange(0, 24, 2)); fig.tight_layout()
    plt.savefig("full_dispatch_schedule_cyclical.png")
    plt.show()
    print("\nSaved cyclical dispatch plot to 'full_dispatch_schedule_cyclical.png'")

def plot_voltage_timeseries(voltage_df: pd.DataFrame):
    """
    Creates a time-series line plot of voltage for selected buses.
    """
    # Select a few key buses to plot for clarity
    buses_to_plot = ['bus_0', 'bus_17', 'bus_32']
    
    plt.figure(figsize=(12, 6))
    
    for bus_col in buses_to_plot:
        if bus_col in voltage_df.columns:
            plt.plot(voltage_df.index, voltage_df[bus_col], label=bus_col, marker='.', linestyle='-')

    # Add horizontal lines for voltage limits
    plt.axhline(y=1.05, color='r', linestyle='--', label='Upper Limit (1.05 p.u.)')
    plt.axhline(y=0.95, color='r', linestyle='--', label='Lower Limit (0.95 p.u.)')
    
    # Formatting
    plt.title('Voltage Profiles for Key Buses')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Voltage (p.u.)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(np.arange(0, 24, 2))
    plt.xlim(0, 23)
    plt.ylim(0.94, 1.06) # Zoom in on the normal operating range
    plt.tight_layout()
    
    # Save the figure
    plt.savefig("voltage_timeseries.png")
    plt.show()
    print("Saved voltage time-series plot to 'voltage_timeseries.png'")

def plot_voltage_heatmap(voltage_df: pd.DataFrame):
    """
    Creates a heatmap of the network's bus voltages over time.
    """
    plt.figure(figsize=(12, 8))
    
    # Transpose the DataFrame so buses are on the y-axis and time is on the x-axis
    heatmap = sns.heatmap(voltage_df.T, cmap='viridis', center=1.0,
                          cbar_kws={'label': 'Voltage (p.u.)'})
    
    heatmap.set_title('Bus Voltage Profile Over 24 Hours')
    heatmap.set_xlabel('Hour of the Day')
    heatmap.set_ylabel('Bus ID')
    
    plt.tight_layout()
    plt.savefig("voltage_heatmap.png")
    plt.show()
    print("Saved voltage heatmap to 'voltage_heatmap.png'") 

def plot_comparison(optimal_df, heuristic_df, grid_price):
    """
    Creates a side-by-side plot comparing the optimal and heuristic dispatch schedules.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True, sharey=True)
    
    # --- Plot 1: Optimal Dispatch ---
    plot_single_dispatch(ax1, optimal_df, grid_price, "Optimal Coordinated Dispatch")
    
    # --- Plot 2: Heuristic Dispatch ---
    plot_single_dispatch(ax2, heuristic_df, grid_price, "Heuristic (Greedy Self-Consumption) Dispatch")
    
    ax2.set_xlabel('Hour of the Day')
    plt.tight_layout()
    plt.savefig("comparison_dispatch.png")
    plt.show()
    print("\nSaved comparison plot to 'comparison_dispatch.png'")

def plot_single_dispatch(ax, df, grid_price, title):
    """Helper function to plot a single dispatch schedule onto a given subplot axis."""
    # (This function contains the plotting logic from your previous `plot_full_dispatch`)
    pv_cols = [col for col in df.columns if 'pv_' in col]; bess_cols = [col for col in df.columns if col.startswith('p_bess')]
    df['total_pv'] = df[pv_cols].sum(axis=1); df['total_bess_p'] = df[bess_cols].sum(axis=1)
    df['bess_discharge'] = df['total_bess_p'].clip(lower=0); df['bess_charge'] = -df['total_bess_p'].clip(upper=0)
    
    # For heuristic, 'curtailed_load' doesn't exist, so we assume it's zero
    if 'curtailed_load' not in df.columns: df['curtailed_load'] = 0
        
    met_load = df['load'] - df['curtailed_load']
    hours = df.index
    
    ax.set_ylabel('Power (MW)')
    ax.set_title(title)
    ax.stackplot(hours, df['total_pv'], df['bess_discharge'], df['grid_import'],
                  labels=['PV Generation', 'BESS Discharge', 'Grid Import'],
                  colors=['#FFD700', '#90EE90', '#ADD8E6'], alpha=0.7)
    ax.stackplot(hours, met_load, df['curtailed_load'],
                  labels=['', 'Load Curtailed'], colors=['none', '#D3D3D3'], alpha=0.8)
    ax.plot(hours, df['load'], label='Original Load', color='black', linewidth=2.5)
    ax.stackplot(hours, -df['bess_charge'], labels=['BESS Charge'], colors=['#FFB6C1'], alpha=0.7)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper left')

def calculate_total_cost(schedule_df, grid_price, bess_objects):
    """Calculates the total operational cost from a schedule DataFrame."""
    bess_map = {b.name: b for b in bess_objects}
    
    # Calculate total grid import cost
    grid_cost = (schedule_df['grid_import'] * grid_price).sum()
    
    # Calculate total degradation cost
    total_degradation_cost = 0
    bess_p_cols = [col for col in schedule_df.columns if col.startswith('p_bess')]
    for col in bess_p_cols:
        bess_name = col.replace('p_', '')
        bess_obj = bess_map[bess_name]
        degradation_cost_per_mwh = bess_obj.replacement_cost / bess_obj.lifetime_throughput
        
        # Ensure only discharge (positive power) contributes to cost
        bess_discharge = schedule_df[col].clip(lower=0)
        total_degradation_cost += (bess_discharge * degradation_cost_per_mwh).sum()
        
    return grid_cost + total_degradation_cost

if __name__ == "__main__":
    # This section is identical to the previous script
    network = nw.case33bw()
    pv_silicon = PvSystem(network=network, bus_idx=17, pv_parameters={},
                          forecast=np.concatenate([np.zeros(6),
                                                   np.sin(np.linspace(0, np.pi, 12)) * 0.9,
                                                   np.zeros(6)]),
                          name="silicon_pv")
    pv_emerging = PvSystem(network=network, bus_idx=16, pv_parameters={},
                           forecast=np.concatenate([np.zeros(6),
                                                    np.sin(np.linspace(0, np.pi, 12)) * 0.9,
                                                    np.zeros(6)]),
                           name="emerging_pv")
    bess1 = BatterySystem(
        network=network, bus_idx=16, capacity_mwh=0.5, 
        charge_efficiency=0.95, discharge_efficiency=0.95, 
        max_p_mw=0.2, initial_soc_percent=50, name="bess_1",
        replacement_cost=1500, # Example: £150k
        lifetime_throughput=2000 # Example: 2000 MWh
    )
    bess2 = BatterySystem(
        network=network, bus_idx=17, capacity_mwh=0.5, 
        charge_efficiency=0.95, discharge_efficiency=0.95, 
        max_p_mw=0.2, initial_soc_percent=50, name="bess_2",
        replacement_cost=1500, # Example: £150k
        lifetime_throughput=2000 # Example: 2000 MWh
    )
    total_load_forecast = network.load.p_mw.sum() * (np.sin(np.linspace(0, 2*np.pi, 24)) * 0.4 + 0.8)
    grid_price_data = np.array([10, 10, 10, 10, 15, 20, 50, 60, 50, 40, 30, 20] * 2)

    print("Running full network optimization with PV and BESS (cyclical constraint)...")
    optimal_df, voltage_df = calculate_full_schedule(
        net=network,
        pv_objects=[pv_silicon, pv_emerging],
        bess_objects=[bess1, bess2],
        grid_price=grid_price_data,
        forecast_range=24,
        load_forecasts=total_load_forecast,
        enable_dr=False
    )
    heuristic_df = calculate_heuristic_schedule(
        pv_objects=[pv_silicon, pv_emerging],
        bess_objects=[bess1],
        load_forecasts=total_load_forecast
    )
    heuristic_df['grid_price'] = grid_price_data # Add for cost calculation

    # --- Compare Results ---
    if optimal_df is not None:
        total_optimal_cost = calculate_total_cost(optimal_df, grid_price_data, [bess1, bess2])
        total_heuristic_cost = calculate_total_cost(heuristic_df, grid_price_data, [bess1, bess2])

        print("\n--- Cost Comparison ---")
        print(f"Optimal Model Total Cost: £{total_optimal_cost:.2f}")
        print(f"Heuristic Model Total Cost: £{total_heuristic_cost:.2f}")
        savings = total_heuristic_cost - total_optimal_cost
        savings_pct = (savings / total_heuristic_cost) * 100
        print(f"Value of Optimization: £{savings:.2f} ({savings_pct:.1f}% savings)")

        # Generate the comparison plot
        plot_comparison(optimal_df, heuristic_df, grid_price_data)
