import numpy as np

from examples.config import BESS_CAPACITY_MWH

# ----- Simulation parameteres -----
NUM_TIME_STEPS = 24 # Number of horizon time steps
TIME_STEP_DURATION_H = 1.0 # Duration of each time step in hours

# ----- Grid Parameters -----
# Define the buses where EV are located
EV_INDICES = [7, 14, 20, 29, 22]
TOTAL_NUM_EVS = len(EV_INDICES)

# ----- Pricing -----
# Hourly electriciy prices (£/MWh) 
ELECTRICITY_PRICE_PER_MWH = np.array([
    50, 45, 42, 40, 40, 45, 55, 60, 70, 75, 80, 85,
    80, 75, 70, 65, 60, 65, 75, 85, 80, 80, 70, 60
]) * TIME_STEP_DURATION_H # Convert £/MWh to MW per time step

# ----- PV Parameters -----
PV_BUSES = [3, 4, 5, 6, 7, 8, 9, 10, 11, 17, 22, 24, 32]
# PV_RATED_CAPACITY = []

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
# EV_MIN_SOC = np.full(TOTAL_NUM_EVS, 0.30) # minSOC0
# EV_MAX_SOC = np.full(TOTAL_NUM_EVS, 0.90) # minSOC0
EV_MIN_SOC = 0.30 # minSOC0
EV_MAX_SOC = 0.90 # minSOC0
EV_TARGET_SOC = 0.90 # maxSOC0 (used as target)


# --- BESS Parameters ---
# Assume one BESS per parking lot
# BESS_CAPACITY_MWH = np.array([0.024, 0.022, 0.028, 0.085, 0.024])
BESS_CAPACITY_MWH = np.full(len(PV_BUSES), 0.085)
BESS_MAX_POWER_MW = np.full(len(PV_BUSES), 0.066)
BESS_MIN_POWER_MW = -BESS_MAX_POWER_MW # Assuming symmetric charge/discharge
BESS_CHARGING_EFFICIENCY = 0.90 # effC
BESS_DISCHARGING_EFFICIENCY = 0.90 # effD
BESS_MIN_SOC = np.full(len(PV_BUSES), 0.10) # minSOCb
BESS_MAX_SOC = np.full(len(PV_BUSES), 0.95) # maxSOCb
# Initial SOC (example, can be randomized like MATLAB)
BESS_INITIAL_SOC = np.full(len(PV_BUSES), 0.5) # SOCbi

# --- EV Parameters ---
NUM_EVS_PER_LOT = 1 # Vpl * num_types # CHANGED FROM 10 to 1
# EV battery capacities (MWh) - based on MATLAB 'Capacity'
# Example: 24/(10*1000) * 10 = 0.024 MWh
EV_TYPES_CAPACITY_MWH = np.array([24, 22, 28, 85, 24, 23, 27, 16, 17, 24]) / 10000 # kWh -> MWh
# EV max charging powers (MW) - based on MATLAB 'Charging'
# Example: 7.2 / 1000 = 0.0072 MW
EV_MAX_POWER_MW = np.array([7.2, 6.6, 10, 17.2, 6.6]) / 1000 # kW -> MW
EV_MIN_POWER_MW = -EV_MAX_POWER_MW

EV_CHARGING_EFFICIENCY = 0.90 # effC
EV_DISCHARGING_EFFICIENCY = 0.90 # effD (for V2G)
EV_MIN_POWER_MW_FACTOR = -1.0 # Factor for V2G capability (minP = factor * maxP)
EV_MIN_SOC = 0.30 # minSOC0
EV_TARGET_SOC = 0.90 # maxSOC0 (used as target)

# EV arrival/departure/SOC randomization parameters 
EV_ARR_MEAN = 8
EV_ARR_STD_DEV = 3
EV_ARR_MIN = 1
EV_ARR_MAX = 20
EV_DEP_MEAN = 17
EV_DEP_STD_DEV = 3
EV_DEP_MIN = 11
EV_DEP_MAX = 24
EV_INITIAL_SOC_MEAN = 0.5
EV_INITIAL_SOC_STD_DEV = 0.25
EV_INITIAL_SOC_MIN = 0.3
EV_INITIAL_SOC_MAX = 0.9 # Should be <= EV_TARGET_SOC

# --- Optimization Parameters ---
SOLVER = 'SCS' # Change solver to SCS
OSQP_MAX_ITER = 50000 # Keep this for reference or if switching back

# --- Load Parameters ---
# Example load profile (relative to base load in pandapower network)
LOAD_PROFILE = np.array([
    0.6, 0.55, 0.5, 0.5, 0.55, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.9,
    0.85, 0.8, 0.75, 0.7, 0.75, 0.85, 0.95, 1.0, 0.9, 0.8, 0.7, 0.65
])/10
