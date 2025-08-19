import pandas as pd
import numpy as np

TEMPERATURE = pd.Series([25,25,25,25,25,25,25,25,25,25,25])
IRRADIANCE = pd.Series([0,100,200,300,400,500,600,700,800,900,1000])

PV_CAPACITY = 0.5
BATTERY_CAPACITY = 0.5
MIN_SOC_CHARGE, MAX_SOC_CHARGE = 0.1, 0.45
CHARGE_EFFICIENCY, DISCHARGE_EFFICIENCY = 0.95, 0.95
MAX_P_BESS, MIN_P_BESS = 0.05, 0.05

SILICON_PV_PARAMETERS = {
    'Name': 'SunPower SPR-305E-WHT-D',
    'BIPV': 'N',
    'Date': '10/5/2009',
    'T_NOCT': 42.4,
    'A_c': 1.7,
    'N_s': 96,
    'I_sc_ref': 5.96,
    'V_oc_ref': 64.2,
    'I_mp_ref': 5.58,
    'V_mp_ref': 54.7,
    'alpha_sc': 0.061745,
    'beta_oc': -0.2727,
    'a_ref': 2.3373,
    'I_L_ref': 5.9657,
    'I_o_ref': 6.3076e-12,
    'R_s': 0.37428,
    'R_sh_ref': 393.2054,
    'Adjust': 8.7,
    'gamma_r': -0.476,
    'series_cell': 5,
    'parallel_cell': 3,
    'Version': 'MM106',
    'EgRef': 1.121,
    'dEgdT': -0.0002677,
    'PTC': 200.1,
    'Technology': 'Mono-c-Si',
    'series_cell': 5,
    'parallel_cell': 3,
    'sd_t_c': -0.38,  # %/ºC
    'epv_t_c': -0.38,  # %/ºC
    'pce_@1sun': 20,
    'pce_@0sun': 18,
    'peak_power_kW': 5,
}

EMERGING_PV_PARAMETERS= {
    'Name': r'$\mu_{epv}$',
    'BIPV': 'N',
    'Date': '10/5/2009',
    'T_NOCT': 42.4,
    'A_c': 1.7,
    'N_s': 96,
    'I_sc_ref': 5.96,
    'V_oc_ref': 64.2,
    'I_mp_ref': 5.58,
    'V_mp_ref': 54.7,
    'alpha_sc': 0.061745,
    'beta_oc': -0.2727,
    'a_ref': 2.3373,
    'I_L_ref': 5.9657,
    'I_o_ref': 6.3076e-12,
    'R_s': 0.37428,
    'R_sh_ref': 393.2054,
    'Adjust': 8.7,
    'gamma_r': -0.476,
    'series_cell': 5,
    'parallel_cell': 3,
    'Version': 'MM106',
    'EgRef': 1.121,
    'dEgdT': -0.0002677,
    'PTC': 200.1,
    'Technology': 'Mono-c-Si',
    'series_cell': 5,
    'parallel_cell': 3,
    'sd_t_c': -0.38,  # %/ºC
    'epv_t_c': -0.38,  # %/ºC
    'pce_@0sun': 21,
    'pce_@1sun': 15,
    'peak_power_kW': 5,
}

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

# NODE_TYPE = np.array([
#     'slack', 'residential',  'residential', 'residential', 'commercial', 'residential', 'industrial', 'industrial', 'industrial', 'commercial',
#     'residential', 'residential', 'residential', 'commercial', 'residential', 'industrial', 'industrial', 'residential', 'commercial', 'commercial',
#     'residential', 'residential', 'industrial', 'commercial', 'residential', 'industrial', 'commercial', 'industrial', 'commercial', 'residential',
#     'residential', 'commercial', 'residential'
# ])
NODE_TYPE = {
    0: 'slack', 1: 'residential', 2: 'residential', 3: 'residential', 4: 'commercial', 5: 'residential', 
    6: 'industrial', 7: 'industrial', 8: 'industrial', 9: 'commercial', 10: 'residential', 11: 'residential', 
    12: 'residential', 13: 'commercial', 14: 'residential', 15: 'industrial', 16: 'industrial', 17: 'residential', 
    18: 'commercial', 19: 'commercial', 20: 'residential', 21: 'residential', 22: 'industrial', 23: 'commercial', 
    24: 'residential', 25: 'industrial', 26: 'commercial', 27: 'industrial', 28: 'commercial', 29: 'residential',
    30: 'residential', 31: 'commercial', 32: 'residential'
}

CLASS_MAPPING = {
    "slack": 0,
    "residential": 1,
    "commercial": 2,
    "industrial": 3,
}


