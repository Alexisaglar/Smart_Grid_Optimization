from pvlib import pvsystem
import pandas as pd
from typing import Dict
import numpy as np
import pandapower as pp

class PvSystem:
    def __init__(
        self,
        net: pp.pandapowerNet,
        bus_idx: int,
        name: str,
        pv_parameters: Dict[str, float],
        peak_power_kW: float,
    ) -> None:
        self.parameters = pv_parameters
        self.peak_power = peak_power_kW 

        self.pv_idx = pp.create_sgen(
            net,
            bus=18,
            p_mw=0.0,
            name=name
        )

    def _beta(self, temperature) -> pd.DataFrame:
        return (self.parameters['sd_t_c'] - self.parameters['epv_t_c']) * temperature 

    def _phi(self, irradiance) -> pd.DataFrame:
        return (self.parameters['pce_@0sun'] + ((self.parameters['pce_@1sun'] - self.parameters['pce_@0sun']) / 1000 ) * irradiance)

    def _delta_mat(self, temperature, irradiance) -> pd.DataFrame:
        return (self._beta(temperature) + self._phi(irradiance) / self.parameters['pce_@1sun'])

    def _single_diode_method(
        self,
        temperature: pd.Series,
        irradiance: pd.Series,
    ) -> pd.DataFrame:
        # Preallocate the output array
        light_current, saturation_current, resistance_series, resistance_shunt, nNsVth = pvsystem.calcparams_desoto(
            effective_irradiance=irradiance,
            temp_cell=temperature,
            alpha_sc=self.parameters['alpha_sc'],
            a_ref=self.parameters['a_ref'],
            I_L_ref=self.parameters['I_L_ref'],
            I_o_ref=self.parameters['I_o_ref'],
            R_sh_ref=self.parameters['R_sh_ref'],
            R_s=self.parameters['R_s'],
            EgRef=self.parameters['EgRef'],
            dEgdT=self.parameters['dEgdT']
        )
        curve_info = pvsystem.singlediode(
            photocurrent=light_current,
            saturation_current=saturation_current,
            resistance_series=resistance_series,
            resistance_shunt=resistance_shunt,
            nNsVth=nNsVth,
            method='lambertw'
        )

        return (curve_info['v_mp'] * curve_info['i_mp'])

    def _emerging_pv_method(
        self,
        temperature: pd.Series,
        irradiance: pd.Series,
    ) -> pd.Series:
        return self._single_diode_method(temperature, irradiance) * self._delta_mat(temperature, irradiance)

    def power_generation(
        self,
        temperature: pd.Series,
        irradiance: pd.Series) -> pd.DataFrame:
        '''
        This method calculates the power output of a PV system based on the emerging PV model 
        References:
            [1] A. A. Celis, H. Sun, C. Groves, and P. Harsh, "Energy
                Self-Sufficiency Assessment of a Novel Low-Light Enhanced
                Photovoltaic Model in the Residential Sector," 2024 IEEE
                Energy Conversion Congress and Exposition (ECCE), 2024,
                pp. 494-500. doi: 10.1109/ECCE55643.2024.10860893.
        '''
        return self._emerging_pv_method(temperature, irradiance) * self.parameters['series_cell'] * self.parameters['parallel_cell']
