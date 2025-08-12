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

    ### NEW SECTION: Variable to represent non-negative degradation increment ###
    model.degradation_increment = pyo.Var(model.BESS_IDs, model.T, domain=pyo.NonNegativeReals)

    if enable_dr:
        model.p_curtailment = pyo.Var(model.BUS_IDs, model.T, domain=pyo.NonNegativeReals)
        model.c_incentive = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    else:
        model.p_curtailment = pyo.Param(model.BUS_IDs, model.T, default=0)

    ### MODIFIED SECTION: Objective Function now uses the degradation_increment variable ###
    def full_objective_rule(model):
        grid_cost = sum(model.p_grid[t] * grid_price[t] for t in model.T)
        
        # F_bess: Sum of degradation increments, scaled by cost factor
        degradation_cost = 0
        for b in model.BESS_IDs:
            bess_obj = bess_map[b]
            cost_factor = (bess_obj.replacement_cost * bess_obj.capacity_mwh) / bess_obj.a_b
            degradation_cost += sum(model.degradation_increment[b, t] * cost_factor for t in model.T if t > 0)

        # F_profit: Profit from the DR program
        dr_profit = 0
        if enable_dr:
            dr_profit = sum(
                (grid_price[t] - model.c_incentive[t]) * model.p_curtailment[i, t]
                for i in model.BUS_IDs if i in base_loads_p.index for t in model.T
            )
            
        return model.w1 * (grid_cost + degradation_cost) - model.w2 * dr_profit
    model.objective = pyo.Objective(rule=full_objective_rule, sense=pyo.minimize)

    # --- Constraints ---
    
    ### NEW SECTION: Constraint to correctly model non-negative degradation cost ###
    def degradation_positivity_rule(model, b, t):
        if t == 0:
            return model.degradation_increment[b, t] == 0
        
        bess_obj = bess_map[b]
        dod_change_term = (model.dod_bess[b, t]**(1 - bess_obj.b_d) - 
                           model.dod_bess[b, t-1]**(1 - bess_obj.b_d))
        
        # This enforces degradation_increment >= max(0, dod_change_term)
        return model.degradation_increment[b, t] >= dod_change_term
    model.degradation_positivity_constraint = pyo.Constraint(model.BESS_IDs, model.T, rule=degradation_positivity_rule)

    # Power balance and other constraints remain the same
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

    # --- Solve and Process Results ---
    solver = pyo.SolverFactory('ipopt')
    results = solver.solve(model, tee=True)

    if (results.solver.status == pyo.SolverStatus.ok) and (results.solver.termination_condition == pyo.TerminationCondition.optimal):
        print("\nOptimal coordinated schedule found!")
        
        grid_power_vals = [pyo.value(model.p_grid[t]) for t in model.T]
        res = {'load': load_forecasts, 'grid_price': grid_price, 'grid_import': [max(0, p) for p in grid_power_vals], 'grid_export': [-min(0, p) for p in grid_power_vals]}
        for pv_id, forecast in pv_forecasts.items(): res[f'pv_{pv_id}'] = forecast
        for b_name in model.BESS_IDs:
            p_charge_vals = [pyo.value(model.p_charge[b_name, t]) for t in model.T]; p_discharge_vals = [pyo.value(model.p_discharge[b_name, t]) for t in model.T]
            res[f'p_{b_name}'] = np.array(p_discharge_vals) - np.array(p_charge_vals)
            res[f'soc_{b_name}'] = [pyo.value(model.soc_bess[b_name, t]) for t in model.T]
        
        if enable_dr:
            res['curtailed_load'] = [sum(pyo.value(model.p_curtailment[i, t]) for i in model.BUS_IDs if i in base_loads_p.index) for t in model.T]
            
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

def plot_full_dispatch(df: pd.DataFrame):
    if 'curtailed_load' not in df.columns: df['curtailed_load'] = 0
    
    df['total_pv'] = df[[col for col in df.columns if 'pv_' in col]].sum(axis=1)
    df['total_bess_p'] = df[[col for col in df.columns if col.startswith('p_bess')]].sum(axis=1)
    df['bess_discharge'] = df['total_bess_p'].clip(lower=0)
    
    hours = df.index
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax1.set_xlabel('Hour of the Day'); ax1.set_ylabel('Power (MW)')
    
    # --- SUPPLY STACK PLOT ---
    ax1.stackplot(hours, df['grid_import'], df['total_pv'], df['bess_discharge'],
                  labels=['Grid Import', 'PV Generation', 'BESS Discharge'],
                  colors=['#ADD8E6', '#FFD700', '#90EE90'], alpha=0.8)
    
    # --- DEMAND PLOTS ---
    ax1.plot(hours, df['load'], label='Total Potential Load', color='black', linewidth=2.5, linestyle='--')
    ax1.plot(hours, df['load'] - df['curtailed_load'], label='Realized Load', color='dimgray', linewidth=2.5)
    ax1.set_xlim(0, 23); ax1.grid(True, linestyle='--', alpha=0.6); ax1.set_ylim(bottom=0)
    
    # --- TWIN AXIS FOR PRICE AND SOC ---
    ax2 = ax1.twinx()
    ax2.set_ylabel('Price (£/MWh) / SoC (%)', color='#8B0000')
    ax2.plot(hours, df['grid_price'], label='Grid Price (£/MWh)', color='#8B0000', linestyle=':', marker='o', markersize=4)
    ax2.tick_params(axis='y', labelcolor='#8B0000')

    soc_cols = [col for col in df.columns if 'soc_' in col]
    if soc_cols:
        for col in soc_cols:
             ax2.plot(hours, df[col], label=f'{col.upper()}', linestyle='-.', linewidth=2.5)
    ax2.set_ylim(0, df['grid_price'].max() * 1.1)
    
    # --- LEGEND ---
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left', ncol=2, frameon=True, framealpha=0.9)
    
    plt.title('24-Hour Coordinated Dispatch (Full Methodology)')
    plt.xticks(np.arange(0, 24, 2)); fig.tight_layout()
    plt.savefig("full_dispatch_schedule_methodology.png")
    plt.show()
    print("\nSaved dispatch plot to 'full_dispatch_schedule_methodology.png'")

if __name__ == "__main__":
    network = nw.case33bw()
    
    pv_silicon = PvSystem(network=network, bus_idx=17, pv_parameters={}, forecast=np.concatenate([np.zeros(6), np.sin(np.linspace(0, np.pi, 12)) * 0.9, np.zeros(6)]), name="silicon_pv")
    pv_emerging = PvSystem(network=network, bus_idx=16, pv_parameters={}, forecast=np.concatenate([np.zeros(6), np.sin(np.linspace(0, np.pi, 12)) * 0.9, np.zeros(6)]), name="emerging_pv")
    
    bess1 = BatterySystem(network=network, bus_idx=16, capacity_mwh=0.5, charge_efficiency=0.95, discharge_efficiency=0.95, max_p_mw=0.2, initial_soc_percent=50, name="bess_1", replacement_cost=150000, lifetime_throughput=3000, b_d_param=0.6, a_b_param=2500)
    bess2 = BatterySystem(network=network, bus_idx=17, capacity_mwh=0.5, charge_efficiency=0.95, discharge_efficiency=0.95, max_p_mw=0.2, initial_soc_percent=50, name="bess_2", replacement_cost=150000, lifetime_throughput=3000, b_d_param=0.6, a_b_param=2500)
    
    total_load_forecast = network.load.p_mw.sum() * (np.sin(np.linspace(0, 2*np.pi, 24)) * 0.4 + 0.8)
    grid_price_data = np.array([50, 45, 40, 40, 45, 60, 80, 120, 110, 90, 70, 60, 55, 50, 55, 75, 130, 150, 120, 90, 80, 70, 60, 50])

    print("Running full network optimization based on the thesis methodology...")
    optimal_df, voltage_df = calculate_full_schedule(
        net=network,
        pv_objects=[pv_silicon, pv_emerging],
        bess_objects=[bess1, bess2],
        grid_price=grid_price_data,
        forecast_range=24,
        load_forecasts=total_load_forecast,
        enable_dr=True
    )

    if optimal_df is not None:
        print("\n--- Plotting Results ---")
        plot_full_dispatch(optimal_df)
    else:
        print("\n--- No results to plot ---")
