import pandas as pd

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
}

