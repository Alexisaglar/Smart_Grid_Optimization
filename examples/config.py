import numpy as np

# --- Simulation Parameters ---
NUM_TIME_STEPS = 24  # T: Time horizon (e.g., 24 hours)
POWER_BASE_MVA = 10  # Sbase: Base MVA for per unit calculations (adjust as needed)
TIME_STEP_DURATION_H = 1.0 # Duration of each time step in hours

# --- Grid Parameters ---
# Define the buses where parking lots/VPPs are located (0-based index)
# Original MATLAB posPL=[8 15 21 30 23] maps to 7, 14, 20, 29, 22 in 0-based indexing for case33bw
PARKING_LOT_BUS_INDICES = [7, 14, 20, 29, 22]
NUM_PARKING_LOTS = len(PARKING_LOT_BUS_INDICES)

# --- Pricing ---
# Example hourly electricity prices ($/MWh) - should match MATLAB 'Price' scaled by Sbase if needed
# Price(t)*Sbase in MATLAB was $/hour. Price here is $/MWh.
# Example: if Price(t) was 50 $/pu*hr and Sbase=10MVA, then price is 50/10 = 5 $/MWh. Adjust accordingly.
ELECTRICITY_PRICE_PER_MWH = np.array([
    50, 45, 42, 40, 40, 45, 55, 60, 70, 75, 80, 85,
    80, 75, 70, 65, 60, 65, 75, 85, 90, 80, 70, 60
]) * TIME_STEP_DURATION_H # Convert $/MWh to $/MW per time step

# --- PV Parameters ---
# Assume rated PV capacity per parking lot (in MW)
# This replaces 'ratedPs' which was an optimization variable in MATLAB
PV_RATED_CAPACITY_MW = np.array([0.5, 0.4, 0.6, 0.3, 0.5]) 

# PV model parameters (example values, adjust based on MATLAB)
PV_AMBIENT_TEMP_C = 25.0  # Ta
PV_NOT_C = 45.0          # Nominal Operating Cell Temperature
PV_ISC = 3.8             # Short-circuit current (A)
PV_KI = 0.004            # Temperature coefficient of Isc (A/°C)
PV_VOC = 45.0            # Open-circuit voltage (V)
PV_KV = -0.16            # Temperature coefficient of Voc (V/°C)
PV_FF = 0.75             # Fill Factor
PV_MODULE_MAX_POWER_W = 300 # Psmax: Max power of a single module (W)

# Example hourly solar irradiance (kW/m^2) - replace with MATLAB 'muS'
SOLAR_IRRADIANCE_KW_PER_M2 = np.array([
    0, 0, 0, 0, 0, 0.0983, 0.2380, 0.4094, 0.5607, 0.6636, 0.6949, 0.6841,
    0.5894, 0.4349, 0.2596, 0.1039, 0.1030, 0, 0, 0, 0, 0, 0, 0
])

# --- BESS Parameters ---
# Assume one BESS per parking lot
# Original Cb(n,b)*Sbase was kWh. Capacity here is MWh.
# Example: C_BSS=[24,...]/(Sbase*1000) -> 24/(10*1000) = 0.0024 pu. Capacity = 0.0024 * 10 MVA = 0.024 MWh
BESS_CAPACITY_MWH = np.array([0.024, 0.022, 0.028, 0.085, 0.024]) # Example values based on C_BSS
# Original maxPb(n,b)*Sbase was kW. Power here is MW.
# Example: Ch_BSS=[7.2,...]/(Sbase*1000) -> 7.2/(10*1000)=0.00072 pu. Power = 0.00072 * 10 MVA = 0.0072 MW
BESS_MAX_POWER_MW = np.array([0.0072, 0.0066, 0.010, 0.0172, 0.0066]) # Example values based on Ch_BSS
BESS_MIN_POWER_MW = -BESS_MAX_POWER_MW # Assuming symmetric charge/discharge
BESS_CHARGING_EFFICIENCY = 0.90 # effC
BESS_DISCHARGING_EFFICIENCY = 0.90 # effD
BESS_MIN_SOC = 0.10 # minSOCb
BESS_MAX_SOC = 0.95 # maxSOCb
# Initial SOC (example, can be randomized like MATLAB)
BESS_INITIAL_SOC = np.full(NUM_PARKING_LOTS, 0.5) # SOCbi

# --- EV Parameters ---
NUM_EVS_PER_LOT = 1 # Vpl * num_types # CHANGED FROM 10 to 1
# EV battery capacities (MWh) - based on MATLAB 'Capacity'
# Example: 24/(10*1000) * 10 = 0.024 MWh
EV_TYPES_CAPACITY_MWH = np.array([24, 22, 28, 85, 24, 23, 27, 16, 17, 24]) / 1000 # kWh -> MWh
# EV max charging powers (MW) - based on MATLAB 'Charging'
# Example: 7.2 / 1000 = 0.0072 MW
EV_TYPES_MAX_POWER_MW = np.array([7.2, 6.6, 10, 17.2, 6.6, 6.6, 6.6, 3.3, 3.3, 6.6]) / 1000 # kW -> MW

EV_CHARGING_EFFICIENCY = 0.90 # effC
EV_DISCHARGING_EFFICIENCY = 0.90 # effD (for V2G)
EV_MIN_POWER_MW_FACTOR = -1.0 # Factor for V2G capability (minP = factor * maxP)
EV_MIN_SOC = 0.30 # minSOC0
EV_TARGET_SOC = 0.90 # maxSOC0 (used as target)

# EV arrival/departure/SOC randomization parameters (similar to MATLAB)
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
])
