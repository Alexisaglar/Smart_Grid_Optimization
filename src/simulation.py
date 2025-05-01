import pandas as pd
import numpy as np
import pandapower as pp
import cvxpy as cp

import src.grid_setup as grid_setup
import utils.config as config
import src.component_models as component_models

def run_simulation():
    """
    Runs the Smart Grid Optimisation simulation.
    """
    net = grid_setup.setup_pandapower_network()
    time_steps = range(config.NUM_TIME_STEPS)

    # Generate EV Data
    ev_data, ev_present_matrix = component_models.generate_ev_data(
        config.EV_INDICES,
        config.TOTAL_EV,
        config.NUM_TIME_STEPS
    )


if __name__ == '__main__':
    run_simulation()
