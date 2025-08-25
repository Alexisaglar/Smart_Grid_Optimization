import pyomo.environ as pyo
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as nw
import matplotlib.pyplot as plt
import seaborn as sns
import pvlib.pvsystem as pvsystem
from utils.system_parameters import *

# --- CORRECTED PV PARAMETERS ---
# 1. Define a set of validated base parameters from the PVLIB database for a similar panel
# This ensures the underlying physics model is stable.

class PvSystem:
    def __init__(
        self,
        network: pp.pandapowerNet,
        bus_idx: int,
        pv_parameters: dict[str, float],
        name: str,
    ) -> None:
        self.network = network
        self.parameters = pv_parameters.copy() # Use a copy to avoid modifying the original dict
        self.name = name
        self.bus_idx = bus_idx
        self.forecast = None # Forecast will be set later
        
        # # CRITICAL FIX: Convert percentage-based parameters to fractions for calculations
        # self.parameters['sd_t_c'] /= 100.0
        # self.parameters['epv_t_c'] /= 100.0
        # self.parameters['pce_@1sun'] /= 100.0
        # self.parameters['pce_@0sun'] /= 100.0
        #
    def _beta(self, temperature) -> pd.DataFrame:
        return (self.parameters['sd_t_c'] - self.parameters['epv_t_c']) * temperature

    def _phi(self, irradiance) -> pd.DataFrame:
        # This models the change in efficiency based on irradiance
        pce_0 = self.parameters['pce_@0sun']
        pce_1 = self.parameters['pce_@1sun']
        return (pce_0 + ((pce_1 - pce_0) / 1000.0) * irradiance)

    def _delta_mat(self, temperature, irradiance) -> pd.DataFrame:
        # This is the correction factor for the emerging PV model
        return (self._beta(temperature) + self._phi(irradiance) / self.parameters['pce_@1sun'])

    def _single_diode_method(
        self,
        temperature: pd.Series,
        irradiance: pd.Series,
    ) -> pd.DataFrame:
        params = [
            'alpha_sc', 'a_ref', 'I_L_ref', 'I_o_ref', 'R_sh_ref', 'R_s',
            'EgRef', 'dEgdT'
        ]
        desoto_params = {key: self.parameters[key] for key in params}

        light_current, saturation_current, resistance_series, resistance_shunt, nNsVth = pvsystem.calcparams_desoto(
            effective_irradiance=irradiance,
            temp_cell=temperature,
            **desoto_params
        )
        curve_info = pvsystem.singlediode(
            photocurrent=light_current,
            saturation_current=saturation_current,
            resistance_series=resistance_series,
            resistance_shunt=resistance_shunt,
            nNsVth=nNsVth,
            method='lambertw'
        )
        return curve_info['v_mp'] * curve_info['i_mp']

    def _emerging_pv_method(
        self,
        temperature: pd.Series,
        irradiance: pd.Series,
    ) -> pd.Series:
        base_power = self._single_diode_method(temperature, irradiance)
        correction_factor = self._delta_mat(temperature, irradiance)
        return base_power * correction_factor

    def power_generation(
        self,
        temperature: pd.Series,
        irradiance: pd.Series) -> pd.DataFrame:
        irradiance_threshold = 1.0
        irradiance_realistic = irradiance.where(irradiance >= irradiance_threshold, 0)
        power_mw = pd.Series(0.0, index=irradiance.index)
        daylight_hours = irradiance_realistic > 0
        if not daylight_hours.any():
            return power_mw

        temp_daylight = temperature[daylight_hours]
        irrad_daylight = irradiance_realistic[daylight_hours]
        power_w = self._emerging_pv_method(temp_daylight, irrad_daylight)
        total_power_mw = power_w * self.parameters['series_cell'] * self.parameters['parallel_cell'] * 1e-4
        power_mw.loc[daylight_hours] = total_power_mw
        return power_mw.fillna(0)


# --- UNCHANGED CLASSES AND FUNCTIONS (BatterySystem, Schedulers, most plots) ---

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
    model.dod_bess = pyo.Var(model.BESS_IDs, model.T, domain=pyo.NonNegativeReals, bounds=(0.1, 0.9))

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
            return model.p_curtailment[i, t] <= 0.40 * base_load_p
        model.max_curtailment_constraint = pyo.Constraint(model.BUS_IDs, model.T, rule=max_curtailment_rule)

        def total_energy_curtailment_rule(model, i):
            if i not in base_loads_p.index: return pyo.Constraint.Skip
            total_base_energy = sum(base_loads_p.get(i, 0) * (load_forecasts / load_forecasts.max())[t] for t in model.T)
            total_curtailed_energy = sum(model.p_curtailment[i, t] for t in model.T)
            return total_curtailed_energy <= 0.20 * total_base_energy
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
        # print("Optimal coordinated schedule found!") # Reduced verbosity
        grid_power_vals = [pyo.value(model.p_grid[t]) for t in model.T]
        res = {
            'load': load_forecasts,
            'grid_price': grid_price,
            'grid_import': [max(0, p) for p in grid_power_vals],
            'grid_export': [-min(0, p) for p in grid_power_vals],
            'grid_total': grid_power_vals}
        for pv_id, forecast in pv_forecasts.items(): res[f'pv_{pv_id}'] = forecast
        for b_name in model.BESS_IDs:
            p_charge_vals = [pyo.value(model.p_charge[b_name, t]) for t in model.T]; p_discharge_vals = [pyo.value(model.p_discharge[b_name, t]) for t in model.T]
            res[f'p_charge_{b_name}'] = np.array(p_charge_vals)
            res[f'p_discharge_{b_name}'] = np.array(p_discharge_vals)
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
        return None, None

def calculate_kpis(schedule_df, bess_objects, grid_price, base_load_mw):
    kpis = {}
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
    total_pv_gen = schedule_df[[col for col in schedule_df.columns if 'pv_' in col]].sum().sum()
    total_load = schedule_df['load'].sum()
    load_covered_locally = total_load - schedule_df['grid_import'].sum()
    kpis['self_sufficiency_rate_pct'] = (load_covered_locally / total_load) * 100 if total_load > 0 else 0
    return kpis

# --- NEW/UPDATED PLOTTING FUNCTIONS ---

def plot_tech_and_forecast_generation(generation_df):
    """Visualizes the impact of PV technology and forecast quality on generation."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))
    styles = {
        'Actual_Silicon': {'color': 'gray', 'linestyle': '--', 'linewidth': 2.5},
        'Day-Ahead_Silicon': {'color': 'skyblue', 'linestyle': '-', 'linewidth': 2},
        'Actual_Emerging': {'color': 'black', 'linestyle': '--', 'linewidth': 2.5},
        'Day-Ahead_Emerging': {'color': 'seagreen', 'linestyle': '-', 'linewidth': 2}
    }
    plot_cols = ['Actual_Silicon', 'Day-Ahead_Silicon', 'Actual_Emerging', 'Day-Ahead_Emerging']
    for col in plot_cols:
        if col in generation_df.columns:
            label = col.replace('_', ' ')
            ax.plot(generation_df.index, generation_df[col], **styles[col], label=label)
    ax.set_title('Impact of PV Technology and Forecast on Generation', fontsize=16)
    ax.set_xlabel('Hour of the Day', fontsize=12)
    ax.set_ylabel('PV Generation (MW)', fontsize=12)
    ax.legend(fontsize=11)
    ax.set_xlim(0, 23)
    ax.set_ylim(bottom=0)
    plt.xticks(np.arange(0, 24, 2))
    fig.tight_layout()
    plt.savefig("tech_and_forecast_generation_comparison.png")
    plt.show()

def plot_cost_savings_comparison(kpi_df):
    """Creates a grouped bar chart to compare total costs by scenario and forecast model."""
    cost_data = kpi_df.pivot(index='Forecast Model', columns='Scenario', values='total_cost')
    cost_data = cost_data.sort_values(by='Advanced (Emerging PV)', ascending=True)
    ax = cost_data.plot(kind='bar', figsize=(14, 8), color=['gray', 'mediumseagreen'], rot=0)
    ax.set_title('Total Daily Cost by PV Scenario and Forecast Model', fontsize=16)
    ax.set_ylabel('Total Daily Cost (£)', fontsize=12)
    ax.set_xlabel('Forecast Model', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    for container in ax.containers:
        ax.bar_label(container, fmt='£%.2f', label_type='edge', padding=3)
    ax.legend(title='PV Scenario', fontsize=11)
    plt.tight_layout()
    plt.savefig("cost_savings_comparison.png")
    plt.show()



def plot_full_dispatch(df: pd.DataFrame, title_suffix="", filename="dispatch.png"):
    """ --- DEFINITIVELY CORRECTED PLOT --- """
    plt.rcParams.update({'font.size': 20, 'axes.labelsize': 20, 'axes.titlesize': 24, 'xtick.labelsize': 16, 'ytick.labelsize': 16, 'legend.fontsize': 18})

    hours = df.index

    # --- 1. DEFINE ALL ENERGY BALANCE COMPONENTS ---
    # SOURCES (where power comes from)
    grid_import = df['grid_import']
    pv_supply = df[[col for col in df.columns if 'pv_' in col]].sum(axis=1)
    bess_discharge = df[[col for col in df.columns if col.startswith('p_discharge')]].sum(axis=1)
    total_supply = grid_import + pv_supply + bess_discharge

    # SINKS (where power goes to)
    bess_charge = df[[col for col in df.columns if col.startswith('p_charge')]].sum(axis=1)
    grid_export = df.get('grid_export', 0) # Use .get for safety
    realized_load = df['load'] - df.get('curtailed_load', 0)
    potential_load = df['load']

    # This is the key: it must equal total_supply
    total_demand_met = realized_load + bess_charge + grid_export

    # --- 2. SETUP PLOT ---
    fig, ax1 = plt.subplots(figsize=(14, 10))
    ax1.set_xlabel('Time (t)')
    ax1.set_ylabel('Power (kW)')

    # --- 3. PLOT THE SOURCES (STACKED) ---
    ax1.stackplot(hours, grid_import * 1000, pv_supply * 1000, bess_discharge * 1000,
                  labels=[r'$P_{G2H}$', r'$P_{PV}$', r'$P_{BS-}$'],
                  colors=['#40679E', '#FD841F', '#527853'],
                  alpha=0.8)

    # --- 4. PLOT THE DEMAND & ENERGY BALANCE ---
    # Plot potential load (what was originally demanded)
    ax1.plot(hours, potential_load * 1000, color='red', linestyle='--', linewidth=2.5, label=r'$P_{D, pot}$')

    # Plot realized load (what was actually consumed after curtailment)
    # The gap between this and the dashed red line IS the curtailment
    ax1.plot(hours, realized_load * 1000, color='#8B0000', linestyle='-', linewidth=2.5, label=r'$P_{D}$')
    
    # Fill the curtailment area between potential and realized load
    ax1.fill_between(hours, potential_load * 1000, realized_load * 1000,
                     color='#FF6347', alpha=0.7, label=r'$P_{Curt}$')

    # Plot the total demand met line - THIS SHOULD MATCH THE TOP OF THE STACKPLOT
    ax1.plot(hours, total_demand_met * 1000, color='black', linestyle=':', linewidth=2.5, label='Total Demand Met')

    # --- 5. SECONDARY Y-AXIS (Price and SoC) ---
    ax2 = ax1.twinx()
    ax2.set_ylabel('Price (£/MWh) / SoC (%)')
    ax2.plot(hours, df['grid_price'], label='Price', color='darkorange', linestyle=':', marker='o', markersize=4)
    soc_cols = [col for col in df.columns if 'soc_' in col]
    if soc_cols:
        ax2.plot(hours, df[soc_cols[0]], color='royalblue', linestyle='-.', linewidth=2.5, label='SoC')
    ax2.set_ylim(0, max(105, df['grid_price'].max() * 1.1))

    # --- 6. FINAL FORMATTING ---
    ax1.set_ylim(0, total_supply.max() * 1100) # Set y-limit based on max supply
    ax1.set_xlim(0, 23)

    # Let Matplotlib handle the legend automatically - much safer!
    lines, leg_labels = ax1.get_legend_handles_labels()
    lines2, leg_labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, leg_labels + leg_labels2, loc='upper left', ncol=2)

    ax1.set_title(f'Daily Energy Balance: {title_suffix}')
    plt.xticks(np.arange(0, 24, 2))
    fig.tight_layout()
    plt.savefig(filename)
    plt.show()

def plot_demand_fulfillment(df: pd.DataFrame, title_suffix="", filename="demand_fulfillment.png"):
    """
    --- NEW: 'Demand-Centric' Visualization ---
    Shows how the realized demand (Pd) is met by stacking the contributions
    from PV, Battery Discharge, and Grid Imports underneath the demand curve.
    """
    plt.style.use('seaborn-v0_8-whitegrid') # Using a style similar to the example
    plt.rcParams.update({'font.size': 18, 'axes.labelsize': 18, 'axes.titlesize': 22, 
                         'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 16})

    hours = df.index

    # --- 1. DEFINE BASE ENERGY COMPONENTS ---
    pv_supply = df[[col for col in df.columns if 'pv_' in col]].sum(axis=1)
    bess_discharge = df[[col for col in df.columns if col.startswith('p_discharge')]].sum(axis=1)
    grid_import = df['grid_import']
    realized_load = df['load'] - df.get('curtailed_load', 0)

    # --- 2. CALCULATE HOW THE LOAD IS MET (Dispatch Logic) ---
    # Step 1: Meet load with available PV first.
    load_met_by_pv = np.minimum(realized_load, pv_supply)
    remaining_load = realized_load - load_met_by_pv

    # Step 2: Meet remaining load with battery discharge.
    load_met_by_bess = np.minimum(remaining_load, bess_discharge)
    remaining_load -= load_met_by_bess

    # Step 3: Meet the rest with grid import.
    load_met_by_grid = np.minimum(remaining_load, grid_import)

    # --- 3. SETUP PLOT ---
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Power (kW)')
    ax.set_title(f'Daily Demand Fulfillment: {title_suffix}')

    # --- 4. PLOT THE STACKED SOURCES ---
    # The order follows the example: PV at the bottom, then Grid, then Battery.
    ax.stackplot(hours, load_met_by_pv * 1000, load_met_by_grid * 1000, load_met_by_bess * 1000,
                 labels=[r'$P_{PV} \to P_{D}$', r'$P_{G2H}$', r'$P_{BS-} \to P_{D}$'],
                 colors=['#FD841F', '#40679E', '#527853'], # Orange, Blue, Green
                 alpha=0.8)

    # --- 5. PLOT THE DEMAND CURVE ON TOP ---
    ax.plot(hours, realized_load * 1000, color='black', linewidth=2.5, label=r'$P_{D}$')
    
    # --- 6. FINAL FORMATTING ---
    ax.set_xlim(0, 23)
    ax.set_ylim(0)
    ax.legend(loc='upper left')
    plt.xticks(np.arange(0, 24, 2))
    fig.tight_layout()
    plt.savefig(filename)
    plt.show() 

def plot_summary_dashboard(df: pd.DataFrame, title_suffix="", filename="summary_dashboard.png"):
    """
    --- NEW: Two-Part Dashboard Visualization ---
    1. TOP: Shows how realized demand is met, plus curtailment and price.
    2. BOTTOM: Shows surplus energy actions: battery charging and grid exports.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.size': 18, 'axes.labelsize': 18, 'axes.titlesize': 22,
                         'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14})

    hours = df.index

    # --- 1. DEFINE ALL ENERGY COMPONENTS ---
    # Sources
    pv_supply = df[[col for col in df.columns if 'pv_' in col]].sum(axis=1)
    bess_discharge = df[[col for col in df.columns if col.startswith('p_discharge')]].sum(axis=1)
    grid_import = df['grid_import']
    # Sinks
    potential_load = df['load']
    curtailed_load = df.get('curtailed_load', 0)
    realized_load = potential_load - curtailed_load
    bess_charge = df[[col for col in df.columns if col.startswith('p_charge')]].sum(axis=1)
    grid_export = df.get('grid_export', 0)
    
    # --- 2. CALCULATE HOW THE LOAD IS MET (Dispatch Logic) ---
    load_met_by_pv = np.minimum(realized_load, pv_supply)
    remaining_load = realized_load - load_met_by_pv
    load_met_by_bess = np.minimum(remaining_load, bess_discharge)
    remaining_load -= load_met_by_bess
    load_met_by_grid = remaining_load

    # --- 3. SETUP PLOT (2 rows, 1 column, shared X-axis) ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True, 
                                   gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(f'Daily Energy Summary: {title_suffix}', fontsize=24)

    # --- 4. TOP PLOT: DEMAND FULFILLMENT ---
    ax1.set_ylabel('Power (kW)')
    
    # Stacked areas showing how realized demand is met
    ax1.stackplot(hours, load_met_by_pv * 1000, load_met_by_grid * 1000, load_met_by_bess * 1000,
                  labels=[r'$P_{PV} \to P_{D}$', r'$P_{G2H}$', r'$P_{BS-} \to P_{D}$'],
                  colors=['#FD841F', '#40679E', '#527853'], alpha=0.8)
    
    # Plot potential and realized demand lines
    ax1.plot(hours, potential_load * 1000, color='red', linestyle='--', linewidth=2.0, label=r'$P_{D, pot}$')
    ax1.plot(hours, realized_load * 1000, color='black', linewidth=2.5, label=r'$P_{D}$')

    # Fill the area between them to show curtailment
    ax1.fill_between(hours, potential_load * 1000, realized_load * 1000,
                     color='#FF6347', alpha=0.7, label=r'$P_{Curt}$')

    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Add Price on a secondary y-axis for the top plot
    ax1_twin = ax1.twinx()
    ax1_twin.set_ylabel('Price (£/MWh)')
    ax1_twin.plot(hours, df['grid_price'], color='darkorange', linestyle=':', marker='o', markersize=4, label='Price')
    ax1_twin.legend(loc='upper right')
    
    # --- 5. BOTTOM PLOT: SURPLUS ENERGY ACTIONS ---
    ax2.set_ylabel('Power (kW)')
    ax2.set_xlabel('Time (t)')

    # Stacked areas for battery charging and grid exports
    ax2.stackplot(hours, bess_charge * 1000, grid_export * 1000,
                  labels=[r'$P_{BS+}$ (Charge)', r'$P_{H2G}$ (Export)'],
                  colors=['#90EE90', '#A9A9A9'], alpha=0.8) # Light Green, Dark Gray
    
    ax2.legend(loc='upper left')
    ax2.grid(True)
    
    # --- 6. FINAL FORMATTING ---
    ax1.set_xlim(0, 23)
    plt.xticks(np.arange(0, 24, 2))
    fig.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust for suptitle
    plt.savefig(filename)
    plt.show()

def plot_integrated_balance(df: pd.DataFrame, title_suffix="", filename="integrated_balance.png"):
    """
    --- NEW: All-in-One Integrated Visualization ---
    Shows demand fulfillment (positive axis) and surplus energy actions 
    (negative axis) in a single, comprehensive plot.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.size': 18, 'axes.labelsize': 18, 'axes.titlesize': 22,
                         'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14})

    hours = df.index

    # --- 1. DEFINE ALL ENERGY COMPONENTS ---
    # Sources for the load
    pv_supply = df[[col for col in df.columns if 'pv_' in col]].sum(axis=1)
    bess_discharge = df[[col for col in df.columns if col.startswith('p_discharge')]].sum(axis=1)
    grid_import = df['grid_import']
    # Demand components
    potential_load = df['load']
    realized_load = potential_load - df.get('curtailed_load', 0)
    # Surplus actions
    bess_charge = df[[col for col in df.columns if col.startswith('p_charge')]].sum(axis=1)
    grid_export = df.get('grid_export', 0)

    # --- 2. CALCULATE HOW THE LOAD IS MET (Dispatch Logic) ---
    load_met_by_pv = np.minimum(realized_load, pv_supply)
    remaining_load = realized_load - load_met_by_pv
    load_met_by_bess = np.minimum(remaining_load, bess_discharge)
    remaining_load -= load_met_by_bess
    load_met_by_grid = remaining_load

    # --- 3. SETUP PLOT ---
    fig, ax1 = plt.subplots(figsize=(16, 10))
    ax1.set_title(f'Daily Integrated Energy Balance: {title_suffix}')
    ax1.set_xlabel('Time (t)')
    ax1.set_ylabel('Power (kW)')

    # --- 4. PLOT POSITIVE AXIS: DEMAND FULFILLMENT ---
    ax1.stackplot(hours, load_met_by_pv * 1000, load_met_by_grid * 1000, load_met_by_bess * 1000,
                  labels=[r'$P_{PV} \to P_{D}$', r'$P_{G2H}$', r'$P_{BS-} \to P_{D}$'],
                  colors=['#FD841F', '#40679E', '#527853'], alpha=0.8)

    # --- 5. PLOT NEGATIVE AXIS: SURPLUS ACTIONS ---
    # Note the minus sign to plot them below the zero axis
    ax1.stackplot(hours, -bess_charge * 1000, -grid_export * 1000,
                  labels=[r'$P_{BS+}$ (Charge)', r'$P_{H2G}$ (Export)'],
                  colors=['#90EE90', '#A9A9A9'], alpha=0.8)

    # --- 6. PLOT DEMAND & CURTAILMENT LINES ---
    ax1.plot(hours, potential_load * 1000, color='red', linestyle='--', linewidth=2.0, label=r'$P_{D, pot}$')
    ax1.plot(hours, realized_load * 1000, color='black', linewidth=2.5, label=r'$P_{D}$')
    ax1.fill_between(hours, potential_load * 1000, realized_load * 1000,
                     color='#FF6347', alpha=0.7, label=r'$P_{Curt}$')

    # --- 7. ADD PRICE ON SECONDARY AXIS ---
    ax2 = ax1.twinx()
    ax2.set_ylabel('Price (£/MWh)')
    ax2.plot(hours, df['grid_price'], color='darkorange', linestyle=':', marker='o', 
             markersize=4, label='Price')
    # ax2.ticks(np.arange(0,100,20))
    ax2.set_ylim(0, 200)

    # --- 8. FINAL FORMATTING ---
    ax1.grid(True)
    ax1.axhline(0, color='black', linewidth=1.5) # Add a strong zero line
    ax1.set_xlim(0, 23)
    
    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left', ncol=2)
    
    plt.xticks(np.arange(0, 24, 2))
    fig.tight_layout()
    plt.savefig(filename)
    plt.show()

def plot_single_integrated_bars(df: pd.DataFrame, title_suffix="", filename="final_integrated_bars.png"):
    """
    --- FINAL VERSION: Single-Plot Integrated Bar Chart ---
    Combines all energy dispatch actions into a single plot.
    - POSITIVE BARS: Show how the actual load is met (PV, Grid, BESS Discharge).
    - NEGATIVE BARS: Show all other actions (Curtailment, BESS Charge, Grid Export).
    """
    # Use the same style and font settings for consistency
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.size': 18, 'axes.labelsize': 18, 'axes.titlesize': 22,
                         'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 12}) # Slightly smaller legend

    hours = df.index
    bar_width = 0.8

    # --- 1. DEFINE ALL ENERGY COMPONENTS ---
    # Sources for the load
    pv_supply = df[[col for col in df.columns if 'pv_' in col]].sum(axis=1)
    bess_discharge = df[[col for col in df.columns if col.startswith('p_discharge')]].sum(axis=1)
    grid_import = df['grid_import']
    # Demand components
    potential_load = df['load']
    curtailed_load = df.get('curtailed_load', 0)
    realized_load = potential_load - curtailed_load
    # Surplus/Non-Load actions
    bess_charge = df[[col for col in df.columns if col.startswith('p_charge')]].sum(axis=1)
    grid_export = df.get('grid_export', 0)

    # --- 2. CALCULATE HOW THE REALIZED LOAD IS MET ---
    load_met_by_pv = np.minimum(realized_load, pv_supply)
    remaining_load_after_pv = realized_load - load_met_by_pv
    load_met_by_bess = np.minimum(remaining_load_after_pv, bess_discharge)
    remaining_load_after_bess = remaining_load_after_pv - load_met_by_bess
    load_met_by_grid = remaining_load_after_bess

    # --- 3. SETUP PLOT (Single Axes) ---
    fig, ax1 = plt.subplots(figsize=(16, 10))
    ax1.set_title(f'Daily Integrated Energy Dispatch: {title_suffix}')
    ax1.set_xlabel('Time (t)')
    ax1.set_ylabel('Power (kW)')

    # --- 4. PLOT POSITIVE BARS: DEMAND FULFILLMENT ---
    ax1.bar(hours, load_met_by_pv * 1000, width=bar_width,
            label=r'$P_{PV} \to P_{D}$', color='#FD841F')
    ax1.bar(hours, load_met_by_grid * 1000, width=bar_width, bottom=load_met_by_pv * 1000,
            label=r'$P_{G2H}$', color='#40679E')
    ax1.bar(hours, load_met_by_bess * 1000, width=bar_width, bottom=(load_met_by_pv + load_met_by_grid) * 1000,
            label=r'$P_{BS-} \to P_{D}$', color='#527853')

    # --- 5. PLOT NEGATIVE BARS: ALL OTHER ACTIONS (STACKED) ---
    # First negative layer: Curtailment
    ax1.bar(hours, -curtailed_load * 1000, width=bar_width,
            label=r'$P_{Curt}$', color='#FF6347') # Coral/Reddish-pink
    
    # Second negative layer: Battery Charging (stacked below curtailment)
    bottom_for_charge = -curtailed_load * 1000
    ax1.bar(hours, -bess_charge * 1000, width=bar_width, bottom=bottom_for_charge,
            label=r'$P_{BS+}$ (Charge)', color='#90EE90') # Light Green

    # Third negative layer: Grid Export (stacked below charge)
    bottom_for_export = bottom_for_charge - (bess_charge * 1000)
    ax1.bar(hours, -grid_export * 1000, width=bar_width, bottom=bottom_for_export,
            label=r'$P_{H2G}$ (Export)', color='#A9A9A9') # Dark Gray

    # --- 6. OVERLAY LINE PLOTS ---
    ax1.plot(hours, potential_load * 1000, color='red', linestyle='--', linewidth=2.0, label=r'$P^{D, pot}$')
    ax1.plot(hours, realized_load * 1000, color='black', linewidth=2.5, label=r'$P^{D}$')

    # --- 7. ADD PRICE ON SECONDARY AXIS ---
    ax2 = ax1.twinx()
    ax2.set_ylabel('Price (/MWh)')
    ax2.plot(hours, df['grid_price'], color='darkorange', linestyle=':', marker='o',
             markersize=4, label='Price')
    ax2.set_ylim(0, 200)

    # --- 8. FINAL FORMATTING ---
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax1.axhline(0, color='black', linewidth=1.5)
    ax1.set_xlim(-0.5, 23.5)

    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left', ncol=3, frameon=True)

    plt.xticks(np.arange(0, 24, 2))
    fig.tight_layout()
    plt.savefig(filename)
    plt.show()


def plot_thesis_dispatch_bars(df: pd.DataFrame, title_suffix="", filename="dispatch_final.png"):
    """
    Generates a thesis-quality, integrated bar chart for energy dispatch.

    This function addresses all specific formatting requests:
    - Uses stacked bar charts for clarity.
    - Corrects labels to standard LaTeX format with superscripts.
    - Displays power in MW.
    - Sets all axes to start at zero and dynamically adjusts limits to fit all data.
    - Increases font sizes for readability.
    - Adds black edges to bars and ensures gridlines are in the background.
    - Enhances visibility of the price curve.
    """
    # 1. --- Global Style and Font Configuration ---
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 25,
        'axes.labelsize': 20,
        'axes.titlesize': 22,
        'xtick.labelsize': 25,
        'ytick.labelsize': 25,
        'legend.fontsize': 20,
        'figure.titlesize': 24
    })

    hours = df.index
    bar_width = 0.85

    # 2. --- Data Preparation (Units are already in MW) ---
    # Sources
    pv_supply = df[[col for col in df.columns if 'pv_' in col]].sum(axis=1)
    bess_discharge = df[[col for col in df.columns if col.startswith('p_discharge')]].sum(axis=1)
    grid_import = df['grid_import']
    # Demands / Sinks
    potential_load = df['load']
    curtailed_load = df.get('curtailed_load', 0)
    realized_load = potential_load - curtailed_load
    bess_charge = df[[col for col in df.columns if col.startswith('p_charge')]].sum(axis=1)
    grid_export = df.get('grid_export', 0)

    # Calculate stacking components for demand fulfillment
    load_met_by_pv = np.minimum(realized_load, pv_supply)
    remaining_load_after_pv = realized_load - load_met_by_pv
    load_met_by_bess = np.minimum(remaining_load_after_pv, bess_discharge)
    remaining_load_after_bess = remaining_load_after_pv - load_met_by_bess
    load_met_by_grid = remaining_load_after_bess

    # 3. --- Plot Setup ---
    fig, ax1 = plt.subplots(figsize=(18, 10))
    # fig.suptitle(f'Daily Energy Dispatch: {title_suffix}', y=0.96)
    ax1.set_ylabel('Power (MW)')
    ax1.set_xlabel('Time (Hour)')
    
    # Ensure grid is drawn behind plot elements
    ax1.set_axisbelow(True)

    # 4. --- Plot Positive Bars (Demand Fulfillment) ---
    # Stack order: PV -> Grid -> BESS
    ax1.bar(hours, load_met_by_pv, width=bar_width,
            label=r'$P^{PV} \to P^{D}$', color='#FD841F', edgecolor='black', linewidth=0.8)
    
    ax1.bar(hours, load_met_by_grid, width=bar_width, bottom=load_met_by_pv,
            label=r'$P^{g}$', color='#40679E', edgecolor='black', linewidth=0.8)
            
    ax1.bar(hours, load_met_by_bess, width=bar_width, bottom=load_met_by_pv + load_met_by_grid,
            label=r'$P^{BS-}$', color='#527853', edgecolor='black', linewidth=0.8)

    # 5. --- Plot Negative Bars (Surplus/Curtailment Actions) ---
    # Stack order: Curtailment -> BESS Charge -> Grid Export
    ax1.bar(hours, -curtailed_load, width=bar_width,
            label=r'$P^{x}$', color='#FF6347', edgecolor='black', linewidth=0.8)
    
    ax1.bar(hours, -bess_charge, width=bar_width, bottom=-curtailed_load,
            label=r'$P^{BS+}$', color='#90EE90', edgecolor='black', linewidth=0.8)
    
    # ax1.bar(hours, -grid_export, width=bar_width, bottom=-(curtailed_load + bess_charge),
    #         label=r'$P_{\mathrm{H2G}}$', color='#A9A9A9', edgecolor='black', linewidth=0.8)

    # 6. --- Overlay Line Plots for Demand ---
    # zorder ensures lines are plotted on top of the bars
    ax1.plot(hours, potential_load, color='darkred', linestyle='--', linewidth=2.5, label=r'$P_{\mathrm{D}}^{\mathrm{pot}}$', zorder=3)
    ax1.plot(hours, realized_load, color='black', linestyle='-', linewidth=3, label=r'$P_{\mathrm{D}}$', zorder=3)
    
    # 7. --- Secondary Y-Axis for Price ---
    ax2 = ax1.twinx()
    ax2.set_ylabel('Price (£/MWh)')
    price_color = 'darkorange'
    ax2.plot(hours, df['grid_price'], color=price_color, linestyle=':', linewidth=3,
             marker='o', markersize=6, label='Price', zorder=3)
    
    # --- EDIT START: COLOR-MATCH RIGHT AXIS AND REMOVE ITS GRID ---
    # The grid is already tied only to ax1, but this makes the distinction clear.
    # Color the tick labels, the ticks themselves, and the axis label to match the price line.
    ax2.yaxis.label.set_color(price_color)
    ax2.tick_params(axis='y', colors=price_color)
    ax2.grid(False) # Explicitly disable grid for the secondary axis
    # --- EDIT END ---

    # 8. --- Final Formatting ---
    ax1.axhline(0, color='black', linewidth=1.5)
    ax1.set_xlim(-0.5, 23.5)
    
    # Dynamic Y-axis Limits
    positive_max = (load_met_by_pv + load_met_by_grid + load_met_by_bess).max()
    demand_max = potential_load.max()
    upper_power_limit = max(positive_max, demand_max) * 1.15
    negative_max = (curtailed_load + bess_charge + grid_export).max()
    lower_power_limit = -negative_max * 1.15
    ax1.set_ylim(lower_power_limit, upper_power_limit)
    
    price_max = df['grid_price'].max()
    ax2.set_ylim(0, price_max * 1.15)

    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left', ncol=3, frameon=True, facecolor='white', framealpha=0.8)
    
    plt.xticks(np.arange(0, 24, 2))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    network = nw.case33bw()

    all_bess = [
        BatterySystem(network=network, bus_idx=16, capacity_mwh=0.5, charge_efficiency=0.95, discharge_efficiency=0.95, max_p_mw=0.2, initial_soc_percent=50, name="bess_1", replacement_cost=150000, lifetime_throughput=3000, b_d_param=0.6, a_b_param=2500),
        BatterySystem(network=network, bus_idx=17, capacity_mwh=0.6, charge_efficiency=0.95, discharge_efficiency=0.95, max_p_mw=0.2, initial_soc_percent=50, name="bess_2", replacement_cost=150000, lifetime_throughput=3000, b_d_param=0.6, a_b_param=2500)
    ]


    # --- NEW: Calculate Realistic Load Forecast ---
    total_load_forecast = np.zeros(24)
    load_factor_map = {
        'residential': RESIDENTIAL_LOAD_FACTOR,
        'industrial': INDUSTRIAL_LOAD_FACTOR,
        'commercial': COMMERCIAL_LOAD_FACTOR
    }
    for _, load_row in network.load.iterrows():
        bus_idx = load_row['bus']
        base_load_mw = load_row['p_mw']
        node_type = NODE_TYPE.get(bus_idx)
        
        if node_type in load_factor_map:
            time_series_load = base_load_mw * load_factor_map[node_type]
            total_load_forecast += time_series_load
            
    # --- End of New Load Calculation ---
    # total_load_forecast = network.load.p_mw.sum() * (np.sin(np.linspace(0, 2*np.pi, 24)) * 0.4 + 0.8)
    grid_price_data = np.array([50, 45, 40, 40, 45, 60, 80, 120, 110, 90, 70, 60, 55, 50, 55, 75, 130, 150, 120, 90, 80, 70, 60, 50])

    try:
        t2m_df = pd.read_csv("june_t2m.csv", parse_dates=['timestamp']).head(24)
        ghi_df = pd.read_csv("june_GHI.csv", parse_dates=['timestamp']).head(24)
    except FileNotFoundError:
        print("Error: Ensure 'december_t2m.csv' and 'december_GHI.csv' are in the same directory.")
        exit()

    forecast_models = {
        'Actual': 'actual',
        'TFT': 'day_ahead_pred',
        'TFT Rolling Horizon': 'rolling_pred_p50',
        'LSTM': 'lstm_pred',
        'Naive': 'naive_pred'
    }

    scenarios = {
        "Baseline (All Silicon)": [SILICON_PV_PARAMETERS, SILICON_PV_PARAMETERS],
        "Advanced (Emerging PV)": [EMERGING_PV_PARAMETERS, SILICON_PV_PARAMETERS]
    }
    
    all_kpis = []
    schedule_storage = {}

    for scenario_name, param_list in scenarios.items():
        print(f"\n{'='*70}\nRUNNING SIMULATIONS FOR SCENARIO: {scenario_name}\n{'='*70}")
        pv_1 = PvSystem(network=network, bus_idx=16, pv_parameters=param_list[0], name="pv_16")
        pv_2 = PvSystem(network=network, bus_idx=17, pv_parameters=param_list[1], name="pv_17")
        all_pv = [pv_1, pv_2]

        for model_name, col_name in forecast_models.items():
            print(f"\n--- Analyzing Forecast Model: {model_name} ---")
            t_forecast, ghi_forecast = t2m_df[col_name], ghi_df[col_name]
            for pv in all_pv:
                pv.forecast = pv.power_generation(temperature=t_forecast, irradiance=ghi_forecast)

            optimal_df, _ = calculate_full_schedule(
                net=network, pv_objects=all_pv, bess_objects=all_bess, grid_price=grid_price_data,
                forecast_range=24, load_forecasts=total_load_forecast.copy(), enable_dr=True
            )

            if optimal_df is not None:
                kpis = calculate_kpis(optimal_df, all_bess, grid_price_data, network.load.p_mw.sum())
                kpis['Scenario'] = scenario_name
                kpis['Forecast Model'] = model_name
                all_kpis.append(kpis)
                # Store the resulting schedule for later plotting
                schedule_storage[(scenario_name, model_name)] = optimal_df
            else:
                print(f"--- Optimal scenario failed to solve for {model_name} in {scenario_name}. Skipping. ---")

    # --- Final Analysis and Plotting ---
    if all_kpis:
        kpi_results_df = pd.DataFrame(all_kpis)
        
        actual_t, actual_ghi = t2m_df['actual'], ghi_df['actual']
        gen_plot_df = pd.DataFrame({
            'Actual_Silicon': PvSystem(None, 0, SILICON_PV_PARAMETERS, "").power_generation(actual_t, actual_ghi),
            'Actual_Emerging': PvSystem(None, 0, EMERGING_PV_PARAMETERS, "").power_generation(actual_t, actual_ghi),
            'TFT_ROLLING_Silicon': PvSystem(None, 0, SILICON_PV_PARAMETERS, "").power_generation(t2m_df['rolling_pred_p50'], ghi_df['rolling_pred_p50']),
            'TFT_ROLLING_Emerging': PvSystem(None, 0, EMERGING_PV_PARAMETERS, "").power_generation(t2m_df['rolling_pred_p50'], ghi_df['rolling_pred_p50'])
        })
        print(gen_plot_df)

        print("\n" + "="*70)
        print("FINAL KPI COMPARISON ACROSS ALL SCENARIOS AND MODELS (SUMMER)")
        print("="*70)
        print(kpi_results_df[['Scenario', 'Forecast Model', 'total_cost', 'self_sufficiency_rate_pct']].round(2))
        print("="*70 + "\n")

        print("\n--- Generating Final Comparison Plots ---")
        plot_tech_and_forecast_generation(gen_plot_df)
        plot_cost_savings_comparison(kpi_results_df)

        # Plot the energy dispatch for best and worst cases
        advanced_kpis = kpi_results_df[kpi_results_df['Scenario'] == 'Advanced (Emerging PV)']
        
        # Plot the dispatch for the 'Advanced' scenario with the 'TFT Rolling Horizon' forecast
        scenario_to_plot = 'Advanced (Emerging PV)'
        model_to_plot = 'TFT Rolling Horizon'
        
        if (scenario_to_plot, model_to_plot) in schedule_storage:
            print(f"--- Generating final thesis plot for: {scenario_to_plot} with {model_to_plot} forecast ---")
            dispatch_df = schedule_storage[(scenario_to_plot, model_to_plot)]
            
            # --- CALL THE NEW, IMPROVED PLOTTING FUNCTION ---
            plot_thesis_dispatch_bars(
                dispatch_df, 
                title_suffix=f"{scenario_to_plot} ({model_to_plot})", 
                filename="dispatch_thesis_final.png"
            )
        else:
            print(f"Could not find the specified scenario '{scenario_to_plot}' with model '{model_to_plot}' to plot.")

    else:
        print("\n--- All scenarios failed to solve. No results to analyze. ---")



    if not advanced_kpis.empty:
          # Get the 'TFT Rolling Horizon' schedule, as requested
          model_name_to_plot = 'TFT Rolling Horizon'
          
          # Check if this schedule exists in your stored results
          if ('Advanced (Emerging PV)', model_name_to_plot) in schedule_storage:
              print(f"--- Generating DASHBOARD BAR PLOT for: {model_name_to_plot} ---")
              
              # Retrieve the corresponding DataFrame
              best_df = schedule_storage[('Advanced (Emerging PV)', model_name_to_plot)]
              
              # --- CALL THE NEW DASHBOARD FUNCTION HERE ---
              plot_single_integrated_bars(
                  best_df, 
                  f"Best Case ({model_name_to_plot} Forecast)", 
                  "dashboard_bars_best_case.png"
              )
          else:
              print(f"Could not find schedule for '{model_name_to_plot}'. Plotting skipped.")            

             # third
            # best_case = advanced_kpis.loc[advanced_kpis['total_cost'].idxmin()]
            # worst_case = advanced_kpis.loc[advanced_kpis['total_cost'].idxmax()]
            #
            # print(f"--- Generating dispatch plot for BEST case: {best_case['Forecast Model']} forecast ---")
            # best_df = schedule_storage[('Advanced (Emerging PV)', best_case['Forecast Model'])]
            # # --- Call the NEW integrated function here ---
            # plot_integrated_balance(best_df, f"Best Case ({best_case['Forecast Model']} Forecast)", "integrated_best.png")
            #
            # print(f"--- Generating dispatch plot for WORST case: {worst_case['Forecast Model']} forecast ---")
            # worst_df = schedule_storage[('Advanced (Emerging PV)', worst_case['Forecast Model'])]
            # # --- And call the NEW integrated function here ---
            # plot_integrated_balance(worst_df, f"Worst Case ({worst_case['Forecast Model']} Forecast)", "integrated_worst.png")

            # second
            # best_case = advanced_kpis.loc[advanced_kpis['total_cost'].idxmin()]
            # worst_case = advanced_kpis.loc[advanced_kpis['total_cost'].idxmax()]
            #
            # print(f"--- Generating dispatch plot for BEST case: {best_case['Forecast Model']} forecast ---")
            # best_df = schedule_storage[('Advanced (Emerging PV)', best_case['Forecast Model'])]
            # # --- Call the NEW function here ---
            # plot_demand_fulfillment(best_df, f"Best Case ({best_case['Forecast Model']} Forecast)", "demand_fulfillment_best.png")
            #
            # print(f"--- Generating dispatch plot for WORST case: {worst_case['Forecast Model']} forecast ---")
            # worst_df = schedule_storage[('Advanced (Emerging PV)', worst_case['Forecast Model'])]
            # # --- And call the NEW function here ---
            # plot_demand_fulfillment(worst_df, f"Worst Case ({worst_case['Forecast Model']} Forecast)", "demand_fulfillment_worst.png")
#
            # main
            # best_case = advanced_kpis.loc[advanced_kpis['total_cost'].idxmin()]
            # worst_case = advanced_kpis.loc[advanced_kpis['total_cost'].idxmax()]
            #
            # print(f"--- Generating dispatch plot for BEST case: {best_case['Forecast Model']} forecast ---")
            # best_df = schedule_storage[('Advanced (Emerging PV)', best_case['Forecast Model'])]
            # plot_full_dispatch(best_df, f"Best Case ({best_case['Forecast Model']} Forecast)", "dispatch_best_case.png")
            #
            # print(f"--- Generating dispatch plot for WORST case: {worst_case['Forecast Model']} forecast ---")
            # worst_df = schedule_storage[('Advanced (Emerging PV)', worst_case['Forecast Model'])]
            # plot_full_dispatch(worst_df, f"Worst Case ({worst_case['Forecast Model']} Forecast)", "dispatch_worst_case.png")
            
    else:
        print("\n--- All scenarios failed to solve. No results to analyze. ---")
