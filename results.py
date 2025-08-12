import pyomo.environ as pyo
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as nw
import matplotlib.pyplot as plt
import seaborn as sns

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
    def __init__(self, network, bus_idx, capacity_mwh, charge_efficiency, discharge_efficiency, max_p_mw, initial_soc_percent, name, replacement_cost, lifetime_throughput, b_d_param, a_b_param):
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
        self.b_d = b_d_param
        self.a_b = a_b_param

# --- Heuristic Model for Baseline Comparison ---
def calculate_heuristic_schedule(
    pv_objects: list,
    bess_objects: list,
    load_forecasts: np.array,
) -> pd.DataFrame:
    pv_forecasts = {pv.name: pv.forecast for pv in pv_objects}
    bess_map = {b.name: b for b in bess_objects}
    bess_names = list(bess_map.keys())
    total_pv_forecast = sum(pv_forecasts.values())
    net_load = load_forecasts - total_pv_forecast.copy()
    p_bess_net = {name: np.zeros(24) for name in bess_names}
    soc_bess = {name: np.zeros(24) for name in bess_names}
    for name, obj in bess_map.items():
        soc_bess[name][0] = obj.initial_soc

    for t in range(24):
        if t > 0:
            for name, obj in bess_map.items():
                power_prev, soc_prev = p_bess_net[name][t-1], soc_bess[name][t-1]
                if power_prev < 0: # Charging
                    soc_bess[name][t] = soc_prev + ((-power_prev * obj.charge_efficiency) / obj.capacity_mwh * 100)
                else: # Discharging
                    soc_bess[name][t] = soc_prev - ((power_prev / obj.discharge_efficiency) / obj.capacity_mwh * 100)
        
        for name, obj in bess_map.items():
            current_soc = soc_bess[name][t]
            if net_load[t] > 0: # Power deficit
                available_energy_mwh = ((current_soc - 20) / 100) * obj.capacity_mwh
                power_from_energy = (available_energy_mwh / 1) * obj.discharge_efficiency
                discharge_power = min(net_load[t], obj.max_p_mw, power_from_energy)
                p_bess_net[name][t] = discharge_power
                net_load[t] -= discharge_power
            elif net_load[t] < 0: # Power surplus
                available_space_mwh = ((90 - current_soc) / 100) * obj.capacity_mwh
                power_to_space = (available_space_mwh / 1) / obj.charge_efficiency
                charge_power = min(-net_load[t], obj.max_p_mw, power_to_space)
                p_bess_net[name][t] = -charge_power
                net_load[t] += charge_power

    res = {'load': load_forecasts}
    for pv_name, forecast in pv_forecasts.items(): res[f'pv_{pv_name}'] = forecast
    for name in bess_names:
        res[f'p_{name}'] = p_bess_net[name]
        res[f'soc_{name}'] = soc_bess[name]
    heuristic_df = pd.DataFrame(res)
    
    pv_total = heuristic_df[[col for col in heuristic_df.columns if 'pv_' in col]].sum(axis=1)
    bess_total = heuristic_df[[col for col in heuristic_df.columns if 'p_bess' in col]].sum(axis=1)
    final_net_power = heuristic_df['load'] - pv_total - bess_total
    heuristic_df['grid_import'] = final_net_power.clip(lower=0)
    heuristic_df['grid_export'] = -final_net_power.clip(upper=0)
    return heuristic_df
    
# --- Main Optimization Model ---
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
    
    model.w1 = pyo.Param(default=0.8, doc="Weight for cost objective")
    model.w2 = pyo.Param(default=0.2, doc="Weight for profit objective")
    model.DR_BUDGET_MAX = pyo.Param(default=500.0, doc="Max daily budget for DR incentives")


    # Control Variables
    model.p_grid = pyo.Var(model.T, domain=pyo.Reals)
    model.q_grid = pyo.Var(model.T, domain=pyo.Reals)
    model.p_flow = pyo.Var(model.T, model.LINE_IDs, domain=pyo.Reals)
    model.q_flow = pyo.Var(model.T, model.LINE_IDs, domain=pyo.Reals)
    model.v_sq = pyo.Var(model.T, model.BUS_IDs, domain=pyo.NonNegativeReals)
    model.p_charge = pyo.Var(model.BESS_IDs, model.T, domain=pyo.NonNegativeReals)
    model.p_discharge = pyo.Var(model.BESS_IDs, model.T, domain=pyo.NonNegativeReals)
    model.soc_bess = pyo.Var(model.BESS_IDs, model.T, domain=pyo.NonNegativeReals, bounds=(20, 90))
    model.dod_bess = pyo.Var(model.BESS_IDs, model.T, domain=pyo.NonNegativeReals, bounds=(0.1, 0.8))

    model.degradation_increment = pyo.Var(model.BESS_IDs, model.T, domain=pyo.NonNegativeReals)

    if enable_dr:
        model.p_curtailment = pyo.Var(model.BUS_IDs, model.T, domain=pyo.NonNegativeReals)
        model.c_incentive = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    else:
        model.p_curtailment = pyo.Param(model.BUS_IDs, model.T, default=0)

    def full_objective_rule(model):
        grid_cost = sum(model.p_grid[t] * grid_price[t] for t in model.T)
        
        degradation_cost = 0
        for b in model.BESS_IDs:
            bess_obj = bess_map[b]
            cost_factor = (bess_obj.replacement_cost * bess_obj.capacity_mwh) / bess_obj.a_b
            degradation_cost += sum(model.degradation_increment[b, t] * cost_factor for t in model.T if t > 0)

        dr_profit = 0
        if enable_dr:
            dr_profit = sum(
                (grid_price[t] - model.c_incentive[t]) * model.p_curtailment[i, t]
                for i in model.BUS_IDs if i in base_loads_p.index for t in model.T
            )
            
        return model.w1 * (grid_cost + degradation_cost) - model.w2 * dr_profit
    model.objective = pyo.Objective(rule=full_objective_rule, sense=pyo.minimize)

    def degradation_positivity_rule(model, b, t):
        if t == 0:
            return model.degradation_increment[b, t] == 0
        
        bess_obj = bess_map[b]
        dod_change_term = (model.dod_bess[b, t]**(1 - bess_obj.b_d) - 
                           model.dod_bess[b, t-1]**(1 - bess_obj.b_d))
        
        return model.degradation_increment[b, t] >= dod_change_term
    model.degradation_positivity_constraint = pyo.Constraint(model.BESS_IDs, model.T, rule=degradation_positivity_rule)

    def active_power_balance_rule(model, t, i):
        flow_out = sum(model.p_flow[t, l] for l in model.LINE_IDs if lines.from_bus[l] == i)
        flow_in = sum(model.p_flow[t, l] for l in model.LINE_IDs if lines.to_bus[l] == i)
        pv_gen = sum(pv_forecasts[pv_id][t] for pv_id in model.PV_IDs if pv_bus_map[pv_id] == i)
        bess_dispatch = sum(model.p_discharge[b, t] - model.p_charge[b, t] for b in model.BESS_IDs if bess_bus_map[b] == i)
        grid_inj = model.p_grid[t] if i == slack_bus_id else 0
        base_load_p = base_loads_p.get(i, 0) * (load_forecasts / load_forecasts.max())[t]
        curtailed_load_p = model.p_curtailment[i, t] if enable_dr else 0
        actual_load_p = base_load_p - curtailed_load_p
        return (pv_gen + bess_dispatch + grid_inj + flow_in - flow_out - actual_load_p) == 0
    model.active_power_balance = pyo.Constraint(model.T, model.BUS_IDs, rule=active_power_balance_rule)

    if enable_dr:
        def max_curtailment_rule(model, i, t):
            if i not in base_loads_p.index: return model.p_curtailment[i, t] == 0
            base_load_p = base_loads_p.get(i, 0) * (load_forecasts / load_forecasts.max())[t]
            return model.p_curtailment[i, t] <= 0.60 * base_load_p
        model.max_curtailment_constraint = pyo.Constraint(model.BUS_IDs, model.T, rule=max_curtailment_rule)

        def total_energy_curtailment_rule(model, i):
            if i not in base_loads_p.index: return pyo.Constraint.Skip
            total_base_energy = sum(base_loads_p.get(i, 0) * (load_forecasts / load_forecasts.max())[t] for t in model.T)
            total_curtailed_energy = sum(model.p_curtailment[i, t] for t in model.T)
            return total_curtailed_energy <= 0.40 * total_base_energy
        model.total_energy_curtailment_constraint = pyo.Constraint(model.BUS_IDs, rule=total_energy_curtailment_rule)

        def consumer_benefit_rule(model, i):
            if i not in base_loads_p.index: return pyo.Constraint.Skip
            phi = 0.5
            incentive_earned = sum(model.c_incentive[t] * model.p_curtailment[i, t] for t in model.T)
            discomfort_cost = sum(
                (pyo.exp(phi * (model.p_curtailment[i, t] / (base_loads_p.get(i, 0) * (load_forecasts / load_forecasts.max())[t] + 1e-6))) - 1)
                for t in model.T
            )
            return incentive_earned >= discomfort_cost
        model.consumer_benefit_constraint = pyo.Constraint(model.BUS_IDs, rule=consumer_benefit_rule)

        def incentive_bounds_rule(model, t):
            return pyo.inequality(0, model.c_incentive[t], grid_price[t])
        model.incentive_bounds = pyo.Constraint(model.T, rule=incentive_bounds_rule)
        
        def total_incentive_budget_rule(model):
            total_incentive_paid = sum(model.c_incentive[t] * model.p_curtailment[i, t] for i in model.BUS_IDs if i in base_loads_p.index for t in model.T)
            return total_incentive_paid <= model.DR_BUDGET_MAX
        model.total_incentive_budget_constraint = pyo.Constraint(rule=total_incentive_budget_rule)

    def dod_soc_link_rule(model, b, t):
        return model.dod_bess[b, t] == 1 - (model.soc_bess[b, t] / 100.0)
    model.dod_soc_link = pyo.Constraint(model.BESS_IDs, model.T, rule=dod_soc_link_rule)
    
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

    def soc_cyclical_rule(model, b_name):
        return model.soc_bess[b_name, forecast_range - 1] >= model.soc_bess[b_name, 0]
    model.soc_cyclical_constraint = pyo.Constraint(model.BESS_IDs, rule=soc_cyclical_rule)

    def reactive_power_balance_rule(model, t, i):
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

    solver = pyo.SolverFactory('ipopt')
    results = solver.solve(model)

    if (results.solver.status == pyo.SolverStatus.ok) and (results.solver.termination_condition == pyo.TerminationCondition.optimal):
        print("\nOptimal coordinated schedule found!")
        
        grid_power_vals = [pyo.value(model.p_grid[t]) for t in model.T]
        res = {'load': load_forecasts, 'grid_price': grid_price, 'grid_import': [max(0, p) for p in grid_power_vals], 'grid_export': [-min(0, p) for p in grid_power_vals]}
        for pv_id, forecast in pv_forecasts.items(): res[f'pv_{pv_id}'] = forecast
        for b_name in model.BESS_IDs:
            p_charge_vals = [pyo.value(model.p_charge[b_name, t]) for t in model.T]; p_discharge_vals = [pyo.value(model.p_discharge[b_name, t]) for t in model.T]
            res[f'p_{b_name}'] = np.array(p_discharge_vals) - np.array(p_charge_vals)
            res[f'soc_{b_name}'] = [pyo.value(model.soc_bess[b_name, t]) for t in model.T]
            res[f'degradation_increment_{b_name}'] = [pyo.value(model.degradation_increment[b_name,t]) for t in model.T]
        
        if enable_dr:
            res['curtailed_load'] = [sum(pyo.value(model.p_curtailment[i, t]) for i in model.BUS_IDs if i in base_loads_p.index) for t in model.T]
            res['c_incentive'] = [pyo.value(model.c_incentive[t]) for t in model.T]
            
        schedule_df = pd.DataFrame(res)

        voltage_data = {}
        for i in model.BUS_IDs:
            voltage_data[f'bus_{i}'] = [np.sqrt(pyo.value(model.v_sq[t, i])) for t in model.T]
        voltage_df = pd.DataFrame(voltage_data, index=model.T)
        
        return schedule_df, voltage_df

    else:
        print("\nSolver failed to find an optimal solution for the full model.")
        print(f"Solver Status: {results.solver.status}")
        print(f"Termination Condition: {results.solver.termination_condition}")
        return None, None

### NEW SECTION: Functions for KPI Calculation and Results Plotting ###

def calculate_kpis(schedule_df, bess_objects, grid_price, base_load_mw):
    """Calculates a dictionary of Key Performance Indicators (KPIs) for a given schedule."""
    kpis = {}
    
    # --- Economic KPIs ---
    kpis['grid_cost'] = (schedule_df['grid_import'] * grid_price).sum()
    
    degradation_cost = 0
    for bess in bess_objects:
        if f'degradation_increment_{bess.name}' in schedule_df.columns:
            cost_factor = (bess.replacement_cost * bess.capacity_mwh) / bess.a_b
            degradation_cost += (schedule_df[f'degradation_increment_{bess.name}'] * cost_factor).sum()
    kpis['degradation_cost'] = degradation_cost
    
    dr_incentive_cost = 0
    if 'curtailed_load' in schedule_df.columns and 'c_incentive' in schedule_df.columns:
        dr_incentive_cost = (schedule_df['curtailed_load'] * schedule_df['c_incentive']).sum()
    kpis['dr_incentive_cost'] = dr_incentive_cost
    
    kpis['total_cost'] = kpis['grid_cost'] + kpis['degradation_cost'] + dr_incentive_cost

    # --- Technical KPIs ---
    kpis['peak_grid_import_kw'] = schedule_df['grid_import'].max() * 1000
    kpis['total_grid_import_kwh'] = schedule_df['grid_import'].sum() * 1000
    
    total_pv_gen = schedule_df[[col for col in schedule_df.columns if 'pv_' in col]].sum().sum()
    total_load = schedule_df['load'].sum()
    
    # Self-Consumption: Percentage of local generation that is used locally
    pv_used_locally = total_pv_gen - schedule_df['grid_export'].sum()
    kpis['self_consumption_rate_pct'] = (pv_used_locally / total_pv_gen) * 100 if total_pv_gen > 0 else 0
    
    # Self-Sufficiency: Percentage of total load covered by local assets
    load_covered_locally = total_load - schedule_df['grid_import'].sum()
    kpis['self_sufficiency_rate_pct'] = (load_covered_locally / total_load) * 100 if total_load > 0 else 0
    
    if 'curtailed_load' in schedule_df.columns:
        kpis['total_curtailed_kwh'] = schedule_df['curtailed_load'].sum() * 1000
        
    return kpis

def plot_cost_comparison(kpi_optimal, kpi_heuristic):
    """Generates a bar chart comparing operational costs."""
    labels = ['Grid Import Cost', 'BESS Degradation Cost', 'DR Incentive Cost']
    optimal_costs = [kpi_optimal.get('grid_cost', 0), kpi_optimal.get('degradation_cost', 0), kpi_optimal.get('dr_incentive_cost', 0)]
    heuristic_costs = [kpi_heuristic.get('grid_cost', 0), kpi_heuristic.get('degradation_cost', 0), 0] # Heuristic has no DR

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, optimal_costs, width, label='Optimal Strategy', color='royalblue')
    rects2 = ax.bar(x + width/2, heuristic_costs, width, label='Heuristic Strategy', color='lightgray')

    ax.set_ylabel('Cost (£)')
    ax.set_title('Comparison of Daily Operational Costs')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.bar_label(rects1, padding=3, fmt='£%.0f')
    ax.bar_label(rects2, padding=3, fmt='£%.0f')
    
    fig.tight_layout()
    plt.savefig("cost_comparison.png")
    plt.show()

def plot_voltage_distribution(voltage_df, title="voltage_distribution.png"):
    """Creates a violin plot of voltage distributions across all buses and times."""
    plt.figure(figsize=(8, 6))
    # Flatten the DataFrame to get all voltage readings in a single series
    all_voltages = voltage_df.values.flatten()
    sns.violinplot(y=all_voltages, inner='quartile', color='skyblue')
    plt.axhline(1.05, color='r', linestyle='--', label='Upper Limit (1.05 p.u.)')
    plt.axhline(0.95, color='r', linestyle='--', label='Lower Limit (0.95 p.u.)')
    plt.title('Distribution of Network Bus Voltages (24h)')
    plt.ylabel('Voltage (p.u.)')
    plt.xlabel('Optimal Coordinated Strategy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(title)
    plt.show()

def plot_dr_analysis(schedule_df):
    """Visualizes the activation of the Demand Response program."""
    if 'curtailed_load' not in schedule_df.columns:
        print("No DR data to plot.")
        return
        
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Subplot 1: Curtailed Load
    ax1.bar(schedule_df.index, schedule_df['curtailed_load'], color='indianred', label='Curtailed Power (MW)')
    ax1.set_ylabel('Power (MW)')
    ax1.set_title('Demand Response Program Activation')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(schedule_df.index, schedule_df['grid_price'], color='black', linestyle='--', marker='o', label='Grid Price (£/MWh)')
    ax1_twin.set_ylabel('Grid Price (£/MWh)')
    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    # Subplot 2: Incentive Price vs Grid Price
    ax2.plot(schedule_df.index, schedule_df['grid_price'], color='black', linestyle='--', marker='o', label='Grid Price (£/MWh)')
    ax2.plot(schedule_df.index, schedule_df['c_incentive'], color='green', marker='.', label='Incentive Price (£/MWh)')
    ax2.set_ylabel('Price (£/MWh)')
    ax2.set_xlabel('Hour of the Day')
    ax2.set_title('Incentive Price vs. Grid Price')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("dr_analysis.png")
    plt.show()


def plot_full_dispatch(df: pd.DataFrame, title_suffix=""):
    # (The dispatch plotting function is unchanged from the previous answer)
    if 'curtailed_load' not in df.columns: df['curtailed_load'] = 0
    df['total_pv'] = df[[col for col in df.columns if 'pv_' in col]].sum(axis=1)
    df['total_bess_p'] = df[[col for col in df.columns if col.startswith('p_bess')]].sum(axis=1)
    df['bess_discharge'] = df['total_bess_p'].clip(lower=0)
    
    hours = df.index
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax1.set_xlabel('Hour of the Day'); ax1.set_ylabel('Power (MW)')
    
    ax1.stackplot(hours, df['grid_import'], df['total_pv'], df['bess_discharge'],
                  labels=['Grid Import', 'PV Generation', 'BESS Discharge'],
                  colors=['#ADD8E6', '#FFD700', '#90EE90'], alpha=0.8)
    
    ax1.plot(hours, df['load'], label='Total Potential Load', color='black', linewidth=2.5, linestyle='--')
    ax1.plot(hours, df['load'] - df['curtailed_load'], label='Realized Load', color='dimgray', linewidth=2.5)
    ax1.set_xlim(0, 23); ax1.grid(True, linestyle='--', alpha=0.6); ax1.set_ylim(bottom=0)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Price (£/MWh) / SoC (%)', color='#8B0000')
    ax2.plot(hours, df['grid_price'], label='Grid Price (£/MWh)', color='#8B0000', linestyle=':', marker='o', markersize=4)
    ax2.tick_params(axis='y', labelcolor='#8B0000')

    soc_cols = [col for col in df.columns if 'soc_' in col]
    if soc_cols:
        for col in soc_cols:
             ax2.plot(hours, df[col], label=f'{col.upper()}', linestyle='-.', linewidth=2.5)
    ax2.set_ylim(0, max(100, df['grid_price'].max() * 1.1))
    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left', ncol=2, frameon=True, framealpha=0.9)
    
    plt.title(f'24-Hour Dispatch: {title_suffix}')
    plt.xticks(np.arange(0, 24, 2)); fig.tight_layout()
    plt.savefig(f"dispatch_schedule_{title_suffix.lower().replace(' ', '_')}.png")
    plt.show()

# --- Main Execution Block ---
if __name__ == "__main__":
    network = nw.case33bw()
    
    pv_silicon = PvSystem(network=network, bus_idx=17, pv_parameters={}, forecast=np.concatenate([np.zeros(6), np.sin(np.linspace(0, np.pi, 12)) * 0.9, np.zeros(6)]), name="silicon_pv")
    pv_emerging = PvSystem(network=network, bus_idx=16, pv_parameters={}, forecast=np.concatenate([np.zeros(6), np.sin(np.linspace(0, np.pi, 12)) * 0.9, np.zeros(6)]), name="emerging_pv")
    
    all_bess = [
        BatterySystem(network=network, bus_idx=16, capacity_mwh=0.5, charge_efficiency=0.95, discharge_efficiency=0.95, max_p_mw=0.2, initial_soc_percent=50, name="bess_1", replacement_cost=150000, lifetime_throughput=3000, b_d_param=0.6, a_b_param=2500),
        BatterySystem(network=network, bus_idx=17, capacity_mwh=0.6, charge_efficiency=0.95, discharge_efficiency=0.95, max_p_mw=0.2, initial_soc_percent=50, name="bess_2", replacement_cost=150000, lifetime_throughput=3000, b_d_param=0.6, a_b_param=2500)
    ]
    
    total_load_forecast = network.load.p_mw.sum() * (np.sin(np.linspace(0, 2*np.pi, 24)) * 0.4 + 0.8)
    grid_price_data = np.array([50, 45, 40, 40, 45, 60, 80, 120, 110, 90, 70, 60, 55, 50, 55, 75, 130, 150, 120, 90, 80, 70, 60, 50])

    print("--- Running Heuristic (Baseline) Scenario ---")
    heuristic_df = calculate_heuristic_schedule(
        pv_objects=[pv_silicon, pv_emerging],
        bess_objects=all_bess,
        load_forecasts=total_load_forecast.copy()
    )
    heuristic_df['grid_price'] = grid_price_data

    print("\n--- Running Optimal (Full Methodology) Scenario ---")
    optimal_df, voltage_df = calculate_full_schedule(
        net=network,
        pv_objects=[pv_silicon, pv_emerging],
        bess_objects=all_bess,
        grid_price=grid_price_data,
        forecast_range=24,
        load_forecasts=total_load_forecast.copy(),
        enable_dr=True
    )

    if optimal_df is not None and heuristic_df is not None:
        print("\n--- Calculating KPIs for Both Scenarios ---")
        kpi_optimal = calculate_kpis(optimal_df, all_bess, grid_price_data, network.load.p_mw.sum())
        kpi_heuristic = calculate_kpis(heuristic_df, all_bess, grid_price_data, network.load.p_mw.sum())
        
        # --- Print KPI Comparison Table ---
        print("\n" + "="*50)
        print("PERFORMANCE AND COST ANALYSIS")
        print("="*50)
        kpi_table = pd.DataFrame({'Optimal Strategy': kpi_optimal, 'Heuristic Baseline': kpi_heuristic}).round(2)
        print(kpi_table)
        print("-"*50)
        savings = kpi_heuristic['total_cost'] - kpi_optimal['total_cost']
        savings_pct = (savings / kpi_heuristic['total_cost']) * 100 if kpi_heuristic['total_cost'] > 0 else 0
        print(f"Total Savings with Optimal Strategy: £{savings:.2f} ({savings_pct:.1f}%)")
        print("="*50 + "\n")
        
        # --- Generate All Plots ---
        print("\n--- Generating Plots ---")
        plot_full_dispatch(optimal_df, title_suffix="Optimal Strategy")
        plot_full_dispatch(heuristic_df, title_suffix="Heuristic Strategy")
        plot_cost_comparison(kpi_optimal, kpi_heuristic)
        plot_voltage_distribution(voltage_df)
        plot_dr_analysis(optimal_df)
        
    else:
        print("\n--- Scenarios failed to solve. No results to analyze. ---")
