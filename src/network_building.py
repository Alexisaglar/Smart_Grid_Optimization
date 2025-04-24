import sys
from numpy._core.multiarray import ndarray
import pandas as pd
import pandapower as pp
import pandapower.networks as pn
import logging
import argparse 
import numpy as np
import matplotlib.pyplot as plt
import time


# Default values
EV_BUS = [2, 5, 32]
PV_BUS = [9, 8, 15]
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
    level = logging.DEBUG,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('smartgridopt.log')

    ],
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

def attach_distributed_sources(
    net: pp.pandapowerNet,
    ev_bus: list,
    pv_bus: list,
    bess_bus: list,
    loads: ndarray,
) -> pp.pandapowerNet:
       

    pass
    # return network_structure

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
    loads: ndarray,
    ev_bus: list,
    pv_bus: list,
) -> None:
    pv_power = [ 
        0, 0, 0, 0, 0, 0, 0.01, 0.02, 0.04, 0.06, 0.1, 0.15,
        0.12, 0.1, 0.05, 0.045, 0.03, 0.01, 0, 0, 0, 0, 0, 0
    ]

    network = 33

    # battery configuration settings
    power_battery = np.zeros((network, horizon))
    power_battery_available = np.zeros((network, horizon))
    power_battery_required = np.zeros((network, horizon))
    power_battery_discharge_max = 0.03
    power_battery_charge_max = 0.03
    battery_total_capacity = 0.5
    battery_discharge_eff = 0.9
    battery_charge_eff = 0.9
    state_of_charge = np.zeros((network, horizon))
    surplus_power = np.zeros((network, horizon))
    delta_t = 60

    for t in range(horizon):
        # Determine the resource state of each bus:
        for bus in range(network):
            # calculate bss state and total demand
            if bus in pv_bus:
                power_battery_available[bus, t] = (state_of_charge[bus, t-1] * battery_total_capacity) / (delta_t * battery_discharge_eff)
                power_battery_required[bus, t] = ((state_of_charge[bus, t-1] - 100) * battery_total_capacity) * battery_charge_eff / delta_t

                # Discharging state
                if pv_power[t] < loads[bus, 0, t]:
                    power_battery[bus, t] = max(
                        (pv_power[t] - loads[bus, 0, t]),
                        -power_battery_available[bus, t],
                        -power_battery_discharge_max
                    )
                    if power_battery_available[bus, t] > abs(power_battery[bus, t]):
                        state_of_charge[bus, t] = state_of_charge[bus, t-1] + (power_battery[bus, t] / battery_total_capacity)
                        loads[bus, 0, t] = power_battery[bus, t] - power_battery_available[bus, t] 

                    else:
                        state_of_charge[bus, t] = 0
                        loads[bus, 0, t] = abs(power_battery[bus, t]) - power_battery_available[bus, t] 

                    print('*' * 50 + f' Bus: {bus}  Time: {t} ' + '*' * 50)
                    print(f'Power coming from PV is not enough to provide for power demand.\n')
                    print(f'PV_power: {pv_power[t]}\n')
                    print(f'Power demand: {loads[bus, 0, t]}\n')
                    print(f'State of Charge: {state_of_charge[bus, t]}\n')
                    print(f'Power battery: {power_battery[bus, t]}\n')

                # Charging state
                if pv_power[t] > loads[bus, 0, t]:
                    power_battery[bus, t] = min(
                        (pv_power[t] - loads[bus, 0, t]),
                        power_battery_required[bus, t],
                        power_battery_charge_max, 
                    )
                    if power_battery_required[bus, t] < power_battery[bus, t]:
                        state_of_charge[bus, t] = 100
                        loads[bus, 0, t] = power_battery[bus, t] + power_battery_required

                    else:
                        loads[bus, 0, t] = 0
                        state_of_charge[bus, t] = state_of_charge[bus, t-1] + (power_battery[bus, t] / battery_total_capacity)
                    # if state_of_charge[bus, t] < 100:
                    #     if power_battery[bus, t] > power_battery_required[bus, t]:
                    #         state_of_charge[bus, t] = 100
                    #     else:
                    print('*' * 50 + f' Bus: {bus}  Time: {t} ' + '*' * 50)
                    print(f'There is a surplus in energy generation.\n')
                    print(f'PV_power: {pv_power[t]}\n')
                    print(f'Power demand: {loads[bus, 0, t]}\n')
                    print(f'State of Charge: {state_of_charge[bus, t]}\n')
                    print(f'Power battery: {power_battery[bus, t]}\n')

            if bus in ev_bus:
                pass

    plt.plot(loads[9, 0, :], label='power demand')
    plt.plot(power_battery[9, :], label='power battery')
    # plt.plot(state_of_charge[9, :], label='SoC')
    plt.plot(pv_power, label='PV')
    plt.legend()
    plt.show()

            # pp.runpf(net)

def main():
    args = parse_args()
    try:
        net = pn.case33bw()
        loads = generate_loads(net, BUS_CLASS)
        network_results(net, 24, loads, args.ev_bus, args.pv_bus)
        # network_structure = attach_distributed_sources(net, args.ev_bus, args.pv_bus, args.bess_bus, loads)
    except Exception as e:
        logger.exception(f'Error while creating network structure: \n{e}')

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.exception(f'Error while running main loop: \n{e}')
        sys.exit(1)
