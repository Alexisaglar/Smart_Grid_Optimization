class BessSystem:
    def __init__(
        self,
        capacity_kWh: float,
        max_charge_kW: float,
        min_charge_kW: float,
        efficiency: float,
        initial_SoC: int
    ) -> None:
        self.capacity = capacity_kWh
        self.max_charge = max_charge_kW
        self.min_charge = min_charge_kW
        self.efficiency = efficiency
        self.soc_history = pd.DataFrame([initial_SoC], columns=['SoC'])

