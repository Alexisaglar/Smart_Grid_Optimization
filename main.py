import pandas as pd
from models.pv_system import PvSystem
from models.bess_system import BessSystem
from models.smart_grid import Bus 
from utils.pv_parameters import SILICON_PV_PARAMETERS, EMERGING_PV_PARAMETERS

TEMPERATURE = pd.Series([25,25,25,25,25,25,25,25,25,25,25])
IRRADIANCE = pd.Series([0,100,200,300,400,500,600,700,800,900,1000])
PV_CAPACITY = 5






if __name__ == "__main__":
    a = PvSystem(EMERGING_PV_PARAMETERS, PV_CAPACITY)
    b = PvSystem(SILICON_PV_PARAMETERS, PV_CAPACITY)
    print(a.power_generation(TEMPERATURE, IRRADIANCE))
    print(b.power_generation(TEMPERATURE, IRRADIANCE))
