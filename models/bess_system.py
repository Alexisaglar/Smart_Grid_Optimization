from ast import Name
import pandas as pd
import numpy as np

class BatterySystem:
    def __init__(
        self,
        capacity_kWh: float,
        max_charge_percentage: float,
        min_charge_percentage: float,
        charge_efficiency: float,
        discharge_efficiency: float,
        initial_SoC: int
    ) -> None:
        self.capacity = capacity_kWh
        self.max_soc = max_charge_percentage
        self.min_soc = min_charge_percentage
        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency
        self.soc_history = [initial_SoC]

    def update_soc(
        self,
        new_soc: float
    ) -> None:
        if not isinstance(new_soc, (int, float)):
            raise TypeError(f"New SoC is not a number: {new_soc}")

        if new_soc > 100 or new_soc < 0:
            raise ValueError(f"The new SoC has to be in [0, 100] range; new SoC: {new_soc}")
        self.soc_history.append(new_soc)
    
    def actual_soc(self) -> None:
        if not self.soc_history:
            raise ValueError("Array is empty")
        print(f"Actual SoC: {self.soc_history[-1]}")


