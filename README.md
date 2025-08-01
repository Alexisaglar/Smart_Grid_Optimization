# Smart Grid Optimization with Virtual Power Plants and Machine Learning

![Status: Active Development](https://img.shields.io/badge/status-active%20development-green)
![Framework: PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=pytorch&logoColor=white)
![Optimization: Pyomo](https://img.shields.io/badge/Optimization-Pyomo-blue)
![Power Systems: PandaPower](https://img.shields.io/badge/Power%20Systems-PandaPower-orange)

This repository contains a comprehensive smart grid optimization framework that integrates advanced machine learning forecasting models with energy management systems. The system addresses both strategic planning and real-time operational challenges in modern distribution networks with high penetration of distributed energy resources (DERs).

---

## 🎯 Project Overview

This project implements a multi-layered approach to smart grid optimization, combining:

1. **Strategic Planning Phase**: Optimal placement and sizing of PV systems and Battery Energy Storage Systems (BESS) in distribution networks
2. **Operational Phase**: Real-time dispatch and coordination of distributed energy resources including BESS and PV systems
3. **Forecasting Integration**: Advanced machine learning models for renewable energy and load forecasting to handle uncertainty

The framework is designed to minimize operational costs while maintaining grid stability and power quality constraints.

---

## 🏗️ System Architecture

### Core Components

* **🔮 Forecasting Engine**: 
  - **Temporal Fusion Transformer (TFT)** for multi-horizon probabilistic forecasting
  - Specialized models for solar irradiance and temperature prediction
  - Integration with real weather data and historical patterns

* **🔋 Energy Management System (EMS)**:
  - Battery Energy Storage System optimization with degradation modeling
  - PV generation forecasting and integration
  - Heuristic and optimization-based dispatch algorithms

* **⚡ Grid Integration**:
  - PandaPower-based network modeling and power flow analysis
  - Voltage and thermal constraint enforcement
  - Distribution system state estimation

* **🎯 Optimization Engine**:
  - Mathematical optimization for BESS dispatch
  - Heuristic algorithms for self-consumption maximization
  - Multi-objective optimization balancing cost, reliability, and sustainability

### Operational Framework

The system operates using a **Model Predictive Control (MPC)** approach:

1. **Forecast Generation**: TFT models generate probabilistic forecasts for the next 24-48 hours
2. **Optimization Planning**: Solve constrained optimization problems for optimal resource dispatch
3. **Real-time Execution**: Implement decisions with rolling horizon updates (15-minute intervals)
4. **Feedback Integration**: Update models based on actual system performance

---

## 📁 Repository Structure

```
Smart_Grid_Optimization/
├── main.py                     # Main execution script with heuristic scheduling
├── models/                     # Core system models
│   ├── smart_grid.py          # Grid topology and bus modeling
│   ├── pv_system.py           # Photovoltaic system models
│   ├── bess_system.py         # Battery Energy Storage System models
│   ├── TemporalFusionTransformer.py  # Custom TFT implementation
│   ├── irradiance_model.ckpt  # Trained solar irradiance model
│   └── temperature_model.ckpt # Trained temperature model
├── utils/                      # Utility functions and parameters
│   ├── pv_parameters.py       # Solar panel specifications
│   ├── system_parameters.py   # System configuration parameters
│   └── run_tft.py            # TFT training and inference utilities
├── data/                       # Data management
│   └── processed_data/
│       └── combined_data.csv  # Preprocessed time series data
├── lightning_logs/            # Training logs and model checkpoints
│   ├── tft_run/              # TFT model training outputs
│   └── version_*/            # Multiple training experiment versions
└── src/                       # Additional source code modules
```

---

## 🚀 Key Features

### Machine Learning Integration
- **Custom TFT Architecture**: Adapted for power systems forecasting with covariates
- **Probabilistic Forecasting**: Uncertainty quantification for robust decision-making
- **Multi-target Prediction**: Simultaneous forecasting of multiple system variables

### Energy Management Systems
- **BESS Optimization**: Lifecycle-aware battery dispatch with degradation modeling
- **Self-Consumption Strategies**: Heuristic algorithms for maximizing renewable energy utilization
- **Grid Integration**: Seamless integration with distribution network constraints

### Grid-Aware Optimization
- **Power Flow Constraints**: AC power flow modeling with voltage and thermal limits
- **Network Topology**: Realistic distribution system modeling
- **Scalable Architecture**: Modular design supporting different network configurations

---

## � Current Implementation Status

### ✅ Completed Features
- [x] Basic grid modeling with PandaPower integration
- [x] TFT model architecture and training pipeline
- [x] BESS dispatch algorithms (heuristic and optimization-based)
- [x] PV generation modeling and forecasting
- [x] Data preprocessing and feature engineering pipeline
- [x] Trained irradiance and temperature forecasting models

### 🔄 In Progress
- [ ] Real-time MPC implementation
- [ ] Advanced energy management strategies
- [ ] Multi-objective optimization framework
- [ ] Grid stability constraint integration

### 📋 Planned Enhancements  
- [ ] Reinforcement learning for adaptive control
- [ ] Distributed optimization algorithms
- [ ] Market integration and pricing strategies
- [ ] Comprehensive performance benchmarking

---

## 🛠️ Getting Started

### Prerequisites
```bash
# Core dependencies
pip install pandas numpy matplotlib seaborn
pip install pandapower pyomo
pip install pytorch-lightning pytorch-forecasting
pip install torch torchvision
```

### Quick Start

1. **Basic Grid Simulation**:
```python
from models.smart_grid import Bus
from models.pv_system import PvSystem
from models.bess_system import BatterySystem
import main

# Run heuristic BESS scheduling
results = main.calculate_heuristic_schedule(pv_objects, bess_objects, load_forecasts)
```

2. **TFT Forecasting**:
```python
from utils.run_tft import train_tft_model
# Train forecasting models
model = train_tft_model(data_path="data/processed_data/combined_data.csv")
```

---

## 📈 Results and Validation

The framework has been tested on realistic distribution network scenarios with:
- **Distributed PV installations** with different orientations and technologies
- **Grid-scale BESS** for energy arbitrage and grid support services
- **Real weather data** for accurate renewable generation modeling
- **Multiple training experiments** tracked in lightning_logs

Key performance metrics include:
- Operational cost reduction: Target 15-25% compared to baseline operations
- Grid constraint violations: <1% of operational time steps
- Forecast accuracy: MAPE <10% for 24-hour horizon PV forecasting
- Model convergence: Successfully trained TFT models with multiple versions

---

## 📚 Related Work

This project builds upon research in:
- **Stochastic optimization** for power systems under uncertainty
- **Energy Management Systems** for distributed energy resources
- **Machine Learning** applications in renewable energy forecasting
- **Distribution system** planning and operation

---

## 🤝 Contributing

This is an active research project. For collaboration opportunities or questions about the methodology, please feel free to reach out.

---

## 📄 License

This project is part of ongoing academic research. Please cite appropriately if using any components in your work.

---

## 🔗 References

- Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting
- Energy Management Systems in smart grids
- Stochastic programming for renewable energy integration
- PandaPower: Convenient Power System Modelling and Analysis

---

*Last updated: August 2025*
