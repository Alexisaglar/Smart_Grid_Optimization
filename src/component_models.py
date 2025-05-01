import numpy as np
import pandas as pd
import utils.config as config

def generate_ev_data(
    ev_idx: int,
    ev_per_lot: int,
    num_time_steps: int,
) -> pd.DataFrame:

    num_total_evs = ev_per_lot * len(ev_idx)
    ev_data = pd.DataFrame()
    ev_data['bus_idx'] = ev_idx

    arrival = np.random.randint(low=16, high=20, size=len(ev_idx))
    departure = np.random.randint(low=6, high=10, size=len(ev_idx))
    capacity_mwh = np.random.randint(low=50, high=70, size=len(ev_idx)) / 1000
    soc_at_arrival = np.random.randint(low=0, high=50, size=len(ev_idx))
    lot_id = np.zeros((len(ev_idx)))

    # Create DataFrame
    ev_data = pd.DataFrame({
        'ev_id': ev_idx,
        'lot_id': lot_id,
        'capacity_mwh': capacity_mwh,
        # 'max_power_mw': max_powers,
        # 'min_power_mw': min_powers, # Negative for V2G
        'arrival_time': arrival,
        'departure_time': departure,
        'soc_at_arrival': soc_at_arrival,
        # 'target_soc': np.full(num_total_evs, config.EV_TARGET_SOC)
    })

    # Create EV presence matrix [num_total_evs, num_time_steps]
    ev_present_matrix = np.zeros((num_total_evs, num_time_steps), dtype=bool)
    print(ev_present_matrix)
    for i, ev in ev_data.iterrows():
        arrival_time = int(ev['arrival_time'])
        departure_time = int(ev['departure_time'])

        # Note that time steps are 1 to T-1. Arrival/Departure has to be converted to 0 based ranges
        arrival_idx = arrival_time - 1
        departure_idx = departure_time - 1

        start_idx = max(0, arrival_idx)

        if arrival_idx < departure_idx: 
            # slice from the arrival to end (exclusive)
            slice_start = min(arrival_idx, num_time_steps)
            slice_end = min(departure_idx, num_time_steps)
            if slice_start < slice_end:
                ev_present_matrix[i, slice_start:slice_end] = True
                # print(ev_present_matrix[i, :])

        elif arrival_idx >= departure_idx: # Overnight case (e.g., arrive 17 (idx 16), depart 8 (idx 7))
             # Part 1: From arrival to end of horizon
             slice_start_1 = min(arrival_idx, num_time_steps)
             if slice_start_1 < num_time_steps:
                 ev_present_matrix[i, slice_start_1:num_time_steps] = True
             # Part 2: From beginning of horizon to departure
             slice_end_2 = min(departure_idx, num_time_steps)
             if slice_end_2 > 0:
                 ev_present_matrix[i, 0:slice_end_2] = True

    print(f"Generated data for {num_total_evs} EVs.")
    return ev_data, ev_present_matrix


if __name__ == '__main__':
    # example usage
    time_steps = range(config.NUM_TIME_STEPS)
    ev_data, ev_presence = generate_ev_data(config.EV_INDICES, config.NUM_EVS_PER_LOT, config.NUM_TIME_STEPS)
    print(ev_presence)

