# Real-Time Resource Allocation in Smart Grids using Temporal Fusion Transformers

![Status: In Progress](https://img.shields.io/badge/status-in%20progress-yellow)
![Framework: PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=pytorch&logoColor=white)

This repository is under active development. It contains the implementation of
a real-time resource allocation framework for smart grids, building upon the
planning-phase models developed previously.

---

## 🎯 Project Goal

While previous work focused on the strategic *planning* of PV systems in
distribution networks, this project addresses the critical challenge of
real-time *operation* and *coordination*. The goal is to develop an advanced
**Energy Management System (EMS)** that minimizes total operational costs by
optimally dispatching all distributed energy resources (DERs) in real-time.

This framework is designed to handle the inherent intermittency of renewable
energy sources by leveraging state-of-the-art probabilistic forecasting.

---

## ⚙️ Proposed Framework

The system operates on a rolling-horizon basis, often called **Model Predictive
Control (MPC)**. The core components are:

* **Forecasting Core**: A **Temporal Fusion Transformer (TFT)** provides
multi-horizon, probabilistic forecasts for PV generation (per material type)
and load demand. This gives the EMS crucial foresight into future grid
conditions.
* **Optimization Engine**: A constrained optimization model co-optimises the
dispatch of **Battery Energy Storage Systems (BESS)**, a **Demand Response
program**, and power exchange with the main grid.
* **Decision Making**: The EMS uses the forecasts to generate an optimal
dispatch schedule for the next 24 hours, which is then implemented in short
intervals (e.g., every 15 minutes) as the horizon rolls forward.

The overall workflow is illustrated below:

![Framework Diagram]

---

## 🚧 Current Status

This repository is currently a work in progress. The code and documentation
will be updated as the research and implementation for this chapter of my
thesis advance.
