import pandapower as pp

class BatterySystem:
    def __init__(
        self,
        net: pp.pandapowerNet,
        bus_idx: int,
        capacity_mwh: float,
        max_energy_mwh: float,
        min_energy_mwh: float,
        charge_efficiency: float,
        discharge_efficiency: float,
        max_p_mw: float,
        min_p_mw: float,
        initial_soc_percent: float,
        name: str = None,
    ) -> None:
        self.capacity_mwh = capacity_mwh
        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency
        self.soc_history = [initial_soc_percent]
        self.net = net

        self.bess_idx = pp.create_storage(
            net,
            bus=bus_idx,
            p_mw=0.0,
            max_e_mwh=max_energy_mwh,
            min_e_mwh=min_energy_mwh,
            soc_percent=initial_soc_percent,
            max_p_mw=max_p_mw,
            min_p_mw=min_p_mw,
            q_mvar=0.0,
            name=name,
            in_service=True,
        )

    def update_power(self, p_mw: float) -> None:
        """Sets the charging/discharging power of the BESS in the pandapower network."""
        self.net.storage.p_mw[self.bess_idx] = p_mw

    def get_current_soc(self) -> float:
        """Gets the current SoC from the pandapower results after a power flow."""
        return self.net.res_storage.soc_percent[self.bess_idx]
