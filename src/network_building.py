import sys
from numpy.core.multiarray import ndarray
import pandas as pd
import pandapower as pp
import pandapower.networks as pn
import logging
import argparse 
import numpy as np


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




def main():
    args = parse_args()
    try:
        net = pn.case33bw()
        loads = generate_loads(net, BUS_CLASS)
        network_structure = attach_distributed_sources(net, args.ev_bus, args.pv_bus, args.bess_bus, loads)
    except Exception as e:
        logger.exception(f'Error while creating network structure: \n{e}')

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.exception(f'Error while running main loop: \n{e}')
        sys.exit(1)
