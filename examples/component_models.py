import numpy as np
import pandas as pd
import config

def calculate_pv_power_mw(time_steps, parking_lot_id):
    """
    Calculates the available PV power generation for a specific parking lot
    over the time steps based on irradiance and temperature.
    Args:
        time_steps (range): Range object for the simulation time steps.
        parking_lot_id (int): Index of the parking lot.
    Returns:
        np.array: Available PV power in MW for each time step.
    """
    pv_power = np.zeros(len(time_steps))
    rated_capacity_kw = config.PV_RATED_CAPACITY_MW[parking_lot_id] 
    if rated_capacity_kw <= 0:
        return pv_power

    # Calculate number of modules based on rated capacity and max module power
    # Note: Sbase is not directly needed here if rated_capacity_kw is correctly defined
    num_modules = np.ceil(rated_capacity_kw * 1000 / config.PV_MODULE_MAX_POWER_W)

    for t in time_steps:
        irradiance_kw_m2 = config.SOLAR_IRRADIANCE_KW_PER_M2[t]
        if irradiance_kw_m2 <= 0:
            pv_power[t] = 0
            continue

        # Cell Temperature Calculation (simplified from MATLAB)
        cell_temp_c = config.PV_AMBIENT_TEMP_C + irradiance_kw_m2 * (config.PV_NOT_C - 20) / 0.8

        # Module Power Calculation (simplified from MATLAB)
        Ig = irradiance_kw_m2 * (config.PV_ISC + config.PV_KI * (cell_temp_c - 25))
        Vg = config.PV_VOC - config.PV_KV * cell_temp_c
        module_power_w = config.PV_FF * Vg * Ig

        # Total PV Power (convert W to MW)
        total_power_mw = num_modules * module_power_w / 1e6

        # Ensure power does not exceed rated capacity (as a simple check)
        pv_power[t] = min(total_power_mw, config.PV_RATED_CAPACITY_MW[parking_lot_id])

    return pv_power


def generate_ev_data(num_lots, evs_per_lot, num_time_steps):
    """
    Generates synthetic data for EVs including arrival/departure times,
    initial SOC, battery capacity, and max power.
    Args:
        num_lots (int): Number of parking lots.
        evs_per_lot (int): Number of EVs per parking lot.
        num_time_steps (int): Number of simulation time steps.
    Returns:
        pd.DataFrame: DataFrame containing EV parameters and schedules.
    """
    num_total_evs = num_lots * evs_per_lot
    num_ev_types = len(config.EV_TYPES_CAPACITY_MWH)

    # Assign EV types cyclically
    type_indices = np.tile(np.arange(num_ev_types), evs_per_lot // num_ev_types + 1)[:evs_per_lot]
    all_type_indices = np.tile(type_indices, num_lots)

    capacities = config.EV_TYPES_CAPACITY_MWH[all_type_indices]
    max_powers = config.EV_TYPES_MAX_POWER_MW[all_type_indices]
    min_powers = config.EV_MIN_POWER_MW_FACTOR * max_powers

    # Generate arrival times
    arrival_times = np.random.normal(config.EV_ARR_MEAN, config.EV_ARR_STD_DEV, num_total_evs)
    arrival_times = np.round(arrival_times).astype(int)
    arrival_times = np.clip(arrival_times, config.EV_ARR_MIN, config.EV_ARR_MAX)

    # Generate departure times
    departure_times = np.random.normal(config.EV_DEP_MEAN, config.EV_DEP_STD_DEV, num_total_evs)
    departure_times = np.round(departure_times).astype(int)
    departure_times = np.clip(departure_times, config.EV_DEP_MIN, config.EV_DEP_MAX)
    # Ensure departure is after arrival
    departure_times = np.maximum(departure_times, arrival_times + 1) # Must stay at least one time step

    # Generate initial SOCs
    initial_socs = np.random.normal(config.EV_INITIAL_SOC_MEAN, config.EV_INITIAL_SOC_STD_DEV, num_total_evs)
    initial_socs = np.clip(initial_socs, config.EV_INITIAL_SOC_MIN, config.EV_INITIAL_SOC_MAX)
    # Ensure initial SOC is not above target SOC
    initial_socs = np.minimum(initial_socs, config.EV_TARGET_SOC - 0.01) # Small margin

    # Assign parking lot IDs
    lot_ids = np.repeat(np.arange(num_lots), evs_per_lot)

    # Create DataFrame
    ev_data = pd.DataFrame({
        'ev_id': np.arange(num_total_evs),
        'lot_id': lot_ids,
        'capacity_mwh': capacities,
        'max_power_mw': max_powers,
        'min_power_mw': min_powers, # Negative for V2G
        'arrival_time': arrival_times,
        'departure_time': departure_times,
        'initial_soc': initial_socs,
        'target_soc': np.full(num_total_evs, config.EV_TARGET_SOC)
    })

    # Create EV presence matrix [num_total_evs, num_time_steps]
    ev_present_matrix = np.zeros((num_total_evs, num_time_steps), dtype=bool)
    for i, ev in ev_data.iterrows():
        # Note: Time steps are 1 to T-1. Arrival/Departure times from config are 1-based.
        # EV is present from arrival_time up to (but not including) departure_time.
        
        # Ensure times are integers
        arrival_time = int(ev['arrival_time'])
        departure_time = int(ev['departure_time'])

        # Convert 1-based arrival/departure times to 0-based indices for slicing
        start_idx = arrival_time - 1 
        end_idx = departure_time - 1 # The EV leaves *at* the departure hour, so it's present up to end_idx

        # Ensure indices are within valid range [0, num_time_steps - 1]
        start_idx = max(0, start_idx)
        # end_idx can be up to num_time_steps for slicing up to the end

        if start_idx < end_idx: # Normal case (e.g., arrive 8 (idx 7), depart 17 (idx 16))
             # Slice from start_idx up to end_idx (exclusive)
             slice_start = min(start_idx, num_time_steps)
             slice_end = min(end_idx, num_time_steps)
             if slice_start < slice_end:
                 ev_present_matrix[i, slice_start:slice_end] = True
        elif start_idx >= end_idx: # Overnight case (e.g., arrive 17 (idx 16), depart 8 (idx 7))
             # Part 1: From arrival to end of horizon
             slice_start_1 = min(start_idx, num_time_steps)
             if slice_start_1 < num_time_steps:
                 ev_present_matrix[i, slice_start_1:num_time_steps] = True
             # Part 2: From beginning of horizon to departure
             slice_end_2 = min(end_idx, num_time_steps)
             if slice_end_2 > 0:
                 ev_present_matrix[i, 0:slice_end_2] = True

    print(f"Generated data for {num_total_evs} EVs.")
    return ev_data, ev_present_matrix

if __name__ == '__main__':
    # Example usage
    time_steps = range(config.NUM_TIME_STEPS)
    pv_power_lot0 = calculate_pv_power_mw(time_steps, 0)
    print("PV Power Lot 0 (MW):", pv_power_lot0)

    ev_data, ev_presence = generate_ev_data(config.NUM_PARKING_LOTS, config.NUM_EVS_PER_LOT, config.NUM_TIME_STEPS)
    print("\nEV Data Sample:")
    print(ev_data.head())
    print("\nEV Presence Matrix Sample (First 5 EVs, First 10 Steps):")
    print(ev_presence[:5, :10])

