import pypower as pp
import pypower.network as pn

class Bus:
    def __init__(
        self,
        bus_id: int,
        load_profile: pd.Series,
        pv: PvSystem,
        bess: BessSystem,
    ) -> None:
        self.id = bus_id
        self.load_profile = load_profile
        self.pv = pv 
        self.bess = bess 

