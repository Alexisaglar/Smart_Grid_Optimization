import pandapower as pp

class BatterySystem:
    def __init__(
        self,
        network: pp.pandapowerNet,
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
        # self.bus_idx = bus_idx
        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency
        self.soc_history = [initial_soc_percent]
        self.max_e_mwh = max_energy_mwh
        self.min_e_mwh = min_energy_mwh
        self.initial_soc = initial_soc_percent
        self.max_p_mw = max_p_mw
        self.min_p_mw = min_p_mw
        self.name = name
        # self.net = net

        self.bess_idx = pp.create_storage(
            network,
            bus=bus_idx,
            p_mw=0.0,
            max_e_mwh=self.max_e_mwh,
            min_e_mwh=self.min_e_mwh,
            soc_percent=self.initial_soc,
            max_p_mw=self.max_p_mw,
            min_p_mw=self.min_p_mw,
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
