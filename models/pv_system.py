from pvlib import pvsystem
import pandas as pd
from typing import Dict
import numpy as np

class PvSystem:
    def __init__(
        self,
        pv_parameters: Dict[str, float],
        peak_power_kW: float,
        temperature_c: pd.Series,
        irradiance_wm2: pd.Series
    ) -> None:
        self.parameters = pv_parameters
        self.peak_power = peak_power_kW 
        self.temperature = temperature_c 
        self.irradiance = irradiance_wm2 


    def beta(self) -> pd.DataFrame:
        return (self.parameters['sd_t_c'] - self.parameters['epv_t_c']) * self.temperature 

    def phi(self) -> pd.DataFrame:
        return (self.parameters['pce_@0sun'] + ((self.parameters['pce_@1sun'] - self.parameters['pce_@0sun']) / 1000 ) * self.irradiance)

    def delta_mat(self) -> pd.DataFrame:
        return (self.beta() + self.phi()/ self.parameters['pce_@1sun'])

    def single_diode_method(self) -> pd.DataFrame:
        # Preallocate the output array
        light_current, saturation_current, resistance_series, resistance_shunt, nNsVth = pvsystem.calcparams_desoto(
            effective_irradiance=self.irradiance,
            temp_cell=self.temperature,
            alpha_sc=self.parameters['alpha_sc'],
            a_ref=self.parameters['a_ref'],
            I_L_ref=self.parameters['I_L_ref'],
            I_o_ref=self.parameters['I_o_ref'],
            R_sh_ref=self.parameters['R_sh_ref'],
            R_s=self.parameters['R_s'],
            EgRef=1.121,
            dEgdT=-0.0002677
        )
        
        curve_info = pvsystem.singlediode(
            photocurrent=light_current,
            saturation_current=saturation_current,
            resistance_series=resistance_series,
            resistance_shunt=resistance_shunt,
            nNsVth=nNsVth,
            method='lambertw'
        )

        sd_output = curve_info['v_mp'] * curve_info['i_mp']

        return sd_output

    def power_generation(self) -> pd.Series:
        return single_diode_method() * delta_mat()

