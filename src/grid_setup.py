import pandapower as pp
import pandapower.networks as pn
import numpy as np
import utils.config as config

def setup_pandapower_network() -> pp.pandapowerNet:
    """
    Loads pandapower case33bw network and sets base values.
    Adds parking lot identifier to buses columns
    """

    net = pn.case33bw()

    net.bus['ev_id'] = False
    for _, bus_index in enumerate(config.EV_INDICES):
        if bus_index in net.bus.index:
            net.bus.loc[bus_index, 'ev_id'] = True
        else:
            print(f"Warning: Parking lot bus index {bus_index} not found in the network")

    # Store original loads
    net.load['original_p_mw'] = net.load.p_mw.copy()
    net.load['original_q_mvar'] = net.load.q_mvar.copy()

    print("Pandapower network case33bw loaded")
    print(f"Parking lots assigned to buses: {config.EV_INDICES}")
    return net

if __name__ == '__main__':
    network = setup_pandapower_network()
    print(network.bus)
    print(network.load)
