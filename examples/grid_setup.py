import pandapower as pp
import pandapower.networks as pn
import numpy as np
import config

def setup_pandapower_network():
    """
    Loads the pandapower case33bw network and sets base values.
    Adds parking lot identifiers to buses.
    """
    # Load the 33-bus network
    net = pn.case33bw()

    # Set base MVA
    net.sn_mva = config.POWER_BASE_MVA

    # Add parking lot info to bus table for easier access
    net.bus['parking_lot_id'] = np.nan
    for idx, bus_index in enumerate(config.PARKING_LOT_BUS_INDICES):
        if bus_index in net.bus.index:
            net.bus.loc[bus_index, 'parking_lot_id'] = idx
        else:
            print(f"Warning: Parking lot bus index {bus_index} not found in network.")

    # Store original load values
    net.load['original_p_mw'] = net.load.p_mw.copy()
    net.load['original_q_mvar'] = net.load.q_mvar.copy()

    print("Pandapower network case33bw loaded.")
    print(f"Parking lots assigned to buses: {config.PARKING_LOT_BUS_INDICES}")
    return net

if __name__ == '__main__':
    # Example usage
    network = setup_pandapower_network()
    print(network.bus)
    print(network.load)
