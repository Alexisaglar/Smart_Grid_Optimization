import pandapower as pp
import pandapower.networks as nw
import pandas as pd
from models.pv_system import PvSystem
from models.bess_system import BatterySystem

class Bus:
    def __init__(
        self,
        bus_id: int,
        load_profile: pd.Series,
        pv: PvSystem,
        bess: BatterySystem,
    ) -> None:
        self.id = bus_id
        self.load_profile = load_profile
        self.pv = pv 
        self.bess = bess 

