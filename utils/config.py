import numpy as np

# ----- Simulation parameteres -----
NUM_TIME_STEPS = 24 # Number of horizon time steps
TIME_STEP_DURATION_H = 1.0 # Duration of each time step in hours

# ----- Grid Parameters -----
# Define the buses where EV are located
EV_INDICES = [7, 14, 20, 29, 22]
TOTAL_EV = len(EV_INDICES)

# ----- Pricing -----
# Hourly electriciy prices (£/MWh) 
ELECTRICITY_PRICE_PER_MW = np.array([
    50, 45, 42, 40, 40, 45, 55, 60, 70, 75, 80, 85,
    80, 75, 70, 65, 60, 65, 75, 85, 90, 80, 70, 60
]) * TIME_STEP_DURATION_H # Convert £/MWh to MW per time step

# ----- PV Parameters -----
PV_RATED_CAPACITY = []

# --- BESS Parameters ---
# Calculating only one BESS per parking lot
# BESS_CAPACITY_MWH = np.array([0.024, 0.022, 0.028, 0.085, 0.024]) # Example values based on C_BSS
# BESS_MAX_POWER_MW = np.array([0.0072, 0.0066, 0.010, 0.0172, 0.0066]) # Example values based on Ch_BSS
# BESS_MIN_POWER_MW = -BESS_MAX_POWER_MW # Assuming symmetric charge/discharge
# BESS_CHARGING_EFFICIENCY = 0.90 # effC
# BESS_DISCHARGING_EFFICIENCY = 0.90 # effD
# BESS_MIN_SOC = 0.10 # minSOCb
# BESS_MAX_SOC = 0.95 # maxSOCb
# Initial SOC (example, can be randomized like MATLAB)
# BESS_INITIAL_SOC = np.full(NUM_PARKING_LOTS, 0.5) # SOCbi

# --- EV Parameters ---
NUM_EVS_PER_LOT = 1 # Vpl * num_types # CHANGED FROM 10 to 1
EV_CHARGING_EFFICIENCY = 0.90 # effC
EV_DISCHARGING_EFFICIENCY = 0.90 # effD (for V2G)
EV_MIN_SOC = 0.30 # minSOC0
EV_TARGET_SOC = 0.90 # maxSOC0 (used as target)


