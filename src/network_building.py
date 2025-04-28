import sys
from typing import Tuple
from numpy._core.multiarray import ndarray
import pandas as pd
import pandapower as pp
import pandapower.networks as pn
import logging
import argparse 
import numpy as np
import matplotlib.pyplot as plt
import time

# battery configuration settings
BATTERY_DISCHARGE_EFF, BATTERY_CHARGE_EFF = 0.95, 0.95
POWER_BATTERY_DISCHARGE_MAX, POWER_BATTERY_CHARGE_MAX = 0.03, 0.03
DELTA_T = 60
BATTERY_TOTAL_CAPACITY = 0.5
network = 33
horizon = 24
state = ''
power_battery = np.zeros((network, horizon))
power_battery_available = np.zeros((network, horizon))
power_battery_required = np.zeros((network, horizon))
power_bus = np.zeros((network, 2, horizon))
state_of_charge = np.zeros((network, horizon))
remain_power = np.zeros((network, horizon))

# Default values
EV_BUS = [8, 15, 28]
PV_BUS = [11, 17, 22, 24, 32]
BESS_BUS = [9, 8, 15]
BUS_CLASS = pd.Series([ # Load Type: 0 = slack, 1 = Residential, 2 = Commercial, 3 = Industrial
    0, 1,  1, 1, 2, 1, 3, 3, 3, 2,
    1, 1, 1, 2, 1, 3, 3, 1, 2, 2,
    1, 1, 3, 2, 1, 3, 2, 3, 2, 1,
    1, 2, 1
])
RESIDENTIAL_LOAD_FACTOR = np.array([
    0.30, 0.40, 0.44, 0.46, 0.50, 0.70, 0.72, 0.80, 0.70, 0.63, 0.50, 0.48,
    0.43, 0.50, 0.44, 0.55, 0.70, 0.85, 1.00, 0.85, 0.75, 0.65, 0.50, 0.44,
])
INDUSTRIAL_LOAD_FACTOR = np.array([
    0.65, 0.60, 0.65, 0.70, 0.80, 0.65, 0.65, 0.60, 0.60, 0.55, 0.50, 0.50, 
    0.50, 0.55, 0.60, 0.65, 0.60, 0.55, 0.68, 0.87, 0.90, 1.00, 0.90, 0.70,
])
COMMERCIAL_LOAD_FACTOR = np.array([
    0.40, 0.38, 0.34, 0.32, 0.36, 0.47, 0.63, 0.84, 0.94, 1.00, 0.97, 0.88,
    0.82, 0.80, 0.72, 0.73, 0.75, 0.65, 0.60, 0.52, 0.44, 0.49, 0.43, 0.42,
])

def parse_args() -> argparse.Namespace():
    parser = argparse.ArgumentParser(
        description='Arguments for Smart Grid Optimisation - EV, PV, BESS and DR program is applied in this work',
    )
    parser.add_argument(
        '--pv-bus', nargs='+', type=int, default=PV_BUS, 
        help='Provide a list of PV buses i.e.: --pv-bus 2 5 32'
    )
    parser.add_argument(
        '--ev-bus', nargs='+', type=int, default=EV_BUS, 
        help='Provide a list of EV buses i.e.: --ev-bus 2 5 32'
    )
    parser.add_argument(
        '--bess-bus', nargs='+', type=int, default=EV_BUS, 
        help='Provide a list of BESS buses i.e.: --bess-bus 2 5 32'
    )
    return parser.parse_args()

# Setup logger.
logging.basicConfig(
    level = logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('smartgridopt.log')

    ],
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

def generate_loads(
        net: pp.pandapowerNet,
        bus_class: list
) -> np.ndarray:
    try:
        factor_map = {
            1: RESIDENTIAL_LOAD_FACTOR,
            2: COMMERCIAL_LOAD_FACTOR,
            3: INDUSTRIAL_LOAD_FACTOR,
        }

        n_buses = len(bus_class)
        n_hours = RESIDENTIAL_LOAD_FACTOR.size
        print(f'n_hours = {n_hours}, n_buses = {n_buses}')

        # mapping factors to classes
        factors = np.vstack([
            factor_map.get(cls, np.zeros(n_hours))
            for cls in bus_class
        ])

        # added 0's as p_mw and q_mvar don't return values for slack bus
        p_base, q_base = np.array(0), np.array(0)

        p_base = np.append(p_base, net.load.p_mw.to_numpy()).reshape(n_buses, 1)
        q_base = np.append(q_base, net.load.q_mvar.to_numpy()).reshape(n_buses, 1)

        # this multiplies a p_base which is of length 33*1, multiplying 33*24
        p_time = p_base * factors
        q_time = p_base * factors

        # make 3d stack (buses, values, time)
        loads = np.stack([p_time, q_time], axis=1)

    except Exception as e:
        logging.exception(f'Error while creating load factors: \n{e}')

    return loads 

def network_results(
    net: pp.pandapowerNet,
    horizon: int,
    power_demand: ndarray,
    # ev_bus: list,
    ev_data: pd.DataFrame,
    pv_bus: list,
) -> None:
    pv_power = [ 
        0, 0, 0, 0, 0, 0, 0.01, 0.02, 0.04, 0.06, 0.1, 0.15,
        0.12, 0.1, 0.05, 0.045, 0.03, 0.01, 0, 0, 0, 0, 0, 0
    ]


    for t in range(horizon):
        power_bus = power_demand
        # Determine the resource state of each bus:
        for bus in pv_bus:
            power_battery_available[bus, t] = (state_of_charge[bus, t-1] * BATTERY_TOTAL_CAPACITY) / (BATTERY_DISCHARGE_EFF * DELTA_T) 
            power_battery_required[bus, t] = ((100 - state_of_charge[bus, t-1]) * BATTERY_TOTAL_CAPACITY * BATTERY_CHARGE_EFF) / DELTA_T 
            print(f'Power_required: {power_battery_required[bus,t]}, power_available: {power_battery_available[bus,t]}, previous_soc: {state_of_charge[bus, t]}')

            # Discharging state
            if pv_power[t] <= power_demand[bus, 0, t]:
                # This will always be a negative value as pv power generation is not enough
                state = 'discharge'
                power_battery[bus, t] = max(
                    (pv_power[t] - power_demand[bus, 0, t]),
                    -power_battery_available[bus, t],
                    -POWER_BATTERY_DISCHARGE_MAX
                )
                print(f'State: {state}, power_battery: {power_battery[bus,t]}')

            # Charge state
            if pv_power[t] > power_demand[bus, 0, t]:
                state = 'charge'
                power_battery[bus, t] = min(
                    (pv_power[t] - power_demand[bus, 0, t]),
                    power_battery_required[bus, t],
                    POWER_BATTERY_CHARGE_MAX, 
                )
                print(f'State: {state}, power_battery: {power_battery[bus,t]}')

            state_of_charge[bus, t] = update_battery_values(
                state_of_charge[bus, t-1],
                power_battery[bus, t],
                state,
            ) 

            power_bus[bus, 0, t] = power_demand[bus, 0, t] - pv_power[t] + power_battery[bus, t]

        for ev in ev_data.index:
            # how do we determine that the EV is in the charging station?
            # we create the available range by arrival and departure values
            ev_power = np.zeros((24, 33))
            if t <= ev_data.departure.iloc[ev] or t >= ev_data.arrival.iloc[ev]:
                print(f'time: {t}')
                # time.sleep(1)
                # power_ev_available[bus, t] = (state_of_charge[bus, t-1] * BATTERY_TOTAL_CAPACITY) / (BATTERY_DISCHARGE_EFF * DELTA_T) 
                # power_ev_required[bus, t] = ((100 - state_of_charge[bus, t-1]) * BATTERY_TOTAL_CAPACITY * BATTERY_CHARGE_EFF) / DELTA_T 

            #
            #
            # if ev in station:
            #     pass
            #
            # if horizon in np.arange(ev_data.iloc[bus].arrival, ev_data.departure).any(): 
            #     power_ev_available[bus, t] = (state_of_charge[bus, t-1] * BATTERY_TOTAL_CAPACITY) / (BATTERY_DISCHARGE_EFF * DELTA_T) 
            #     power_ev_required[bus, t] = ((100 - state_of_charge[bus, t-1]) * BATTERY_TOTAL_CAPACITY * BATTERY_CHARGE_EFF) / DELTA_T 
            #     if ev_data.soc_at_arrival != 100:
            #         state = 'charge'
            #         power_ev = update_battery_values(ev_data.soc_at_arrival, ev_data.battery_capacity,)
            #
            #     if V2G_enables == True :
            #         state = 'discharge'
            # pass
            #




        net.load.p_mw = power_demand[1:, 0, t]
        pp.runpp(net)

        print('*' * 50 + f' Power flow results at time: {t} ' +'*' * 50)
        print(f'p_mw: {net.res_load.p_mw.T}, voltage_pu: {net.res_bus.vm_pu.T}')

def update_battery_values(
    previous_soc: float,
    power_battery: float,
    state: str,
    delta_t: float = DELTA_T,
    charging_eff: float = BATTERY_CHARGE_EFF,
    discharge_eff: float = BATTERY_DISCHARGE_EFF,
    battery_capacity: float = BATTERY_TOTAL_CAPACITY,
) -> float:
    match state:
        case 'charge':
            state_of_charge = previous_soc + ((power_battery * delta_t / battery_capacity * discharge_eff)/100)
            return state_of_charge

        case 'discharge':
            state_of_charge = previous_soc + ((power_battery * delta_t * charging_eff / battery_capacity)/100)
            return state_of_charge

def generate_ev_data(
    ev_bus: list,
    random_profile: bool,
)-> pd.DataFrame:
    ev_data = pd.DataFrame()
    ev_data['bus_idx'] = ev_bus

    if random_profile:
        ev_data['arrival'] = np.random.randint(low=16, high=20, size=len(ev_bus))
        ev_data['departure'] = np.random.randint(low=6, high=10, size=len(ev_bus))
        # ev_data['range_at_station'] = [np.arange(ev_data['arrival'].iloc[i], ev_data['departure'].iloc[i]) for i in range(len(ev_data))]
        ev_data['battery_capacity'] = np.random.randint(low=50, high=70, size=len(ev_bus))
        ev_data['soc_at_arrival'] = np.random.randint(low=0, high=50, size=len(ev_bus))

    # For testing purposes generate a stable constant arrival/departure/soc
    else:
        ev_data['arrival'] = np.full(shape=len(ev_bus), fill_value=17)
        ev_data['departure'] = np.full(shape=len(ev_bus), fill_value=8)
        # ev_data['range_at_station'] = [np.arange(ev_data['arrival'].iloc[i], ev_data['departure'].iloc[i], 1) for i in range(len(ev_data))]
        ev_data['battery_capacity'] = np.full(shape=len(ev_bus), fill_value=17)
        ev_data['soc_at_arrival'] = np.full(shape=len(ev_bus), fill_value=50)
        pass

    return ev_data



def main():
    try:
        args = parse_args()
        net = pn.case33bw()
        loads = generate_loads(net, BUS_CLASS)
        ev_data = generate_ev_data(args.ev_bus, random_profile=True)
        network_results(net, horizon, loads, ev_data, args.pv_bus)

    except Exception as e:
        logger.exception(f'Error while creating network structure: \n{e}')

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.exception(f'Error while running main loop: \n{e}')
        sys.exit(1)
