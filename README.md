﻿# ANFIS-based Fuzzy Logic System for WRR Weight Adjustment

Welcome to the **ANFIS-based Fuzzy Logic System**! This project implements a Fuzzy Rule-Based System (FRBS) with an **Adaptive Neuro-Fuzzy Inference System (ANFIS)** for dynamically adjusting Weighted Round Robin (WRR) weights based on buffer occupancies.

This repository integrates the ANFIS library developed by Tim Meggs ([twmeggs/anfis](https://github.com/twmeggs/anfis)) to enhance the decision-making capabilities of a fuzzy logic system. 

**Author**: [Flávio Rocha](https://github.com/flaviogeraldo)

## Overview

The project leverages **ANFIS** to improve the performance of a fuzzy logic system in a networking environment, particularly to optimize the allocation of WRR weights based on buffer states. This results in more adaptive and dynamic control in distributed systems.

- **ANFIS** is used to train the system to learn from historical data and predict WRR weights adjustments.
- The project uses **skfuzzy** to implement fuzzy logic and to visualize how membership functions evolve during training.
- **Membership Functions** are automatically adjusted based on predictions made by the ANFIS system.

## Key Features

- **Adaptive Membership Functions**: Automatically adjust based on real-time predictions.
- **ANFIS Integration**: Uses ANFIS to train the fuzzy system and improve predictions.
- **WRR Weight Adjustment**: Dynamically adjusts WRR weights based on buffer conditions.
- **Customizable Fuzzy Logic Rules**: Easily modify the fuzzy rules to define the decision logic.

## Project Structure

```
.
├── ANFIS_adjust.py               # ANFIS integration for adjusting membership functions
├── Main_Sim.py                   # Main for simulation and obtaining table and plot results
├── FRBS_with_ANFIS_call.py       # Fuzzy logic control system with ANFIS call for adjustment
├── Multi_Queue_Tandem_Fed_Sim.py # Multi-queue simulation with federated learning
├── README.md                     # Project documentation
├── requirements.txt              # Dependencies
└── tests.py                      # Unit tests for core functionality
```

## Getting Started

### Prerequisites

Before running the project, ensure that you have the following dependencies installed:

- Python 3.8+
- NumPy
- scikit-fuzzy (`skfuzzy`)
- scikit-learn
- Matplotlib

You can install all dependencies with the following command:

```bash
pip install -r requirements.txt
```

### Installation

1. Clone this repository:

```bash
git clone https://github.com/flaviogeraldo/fuzzy_wrr.git
```

### Usage

To train the ANFIS model and adjust WRR weights based on buffer occupancy, use the following code:

TODO
```

### Visualizing Membership Functions

You can visualize how the membership functions have been adjusted with the following code:

```python
ANFIS_adjust.show_adjusted_membership_functions(BufferOccupancy1, BufferOccupancy2)
```

### Running Tests

The project comes with unit tests that validate its functionality. To run the tests, use the following command:

```bash
python -m TODO
```

## Contributing

Feel free to contribute to the project! Whether it’s reporting issues, suggesting new features, or submitting pull requests, your contributions are welcome. Make sure to follow the repository's contribution guidelines.

- **Main Repository**: [Flávio Rocha GitHub](https://github.com/flaviogeraldo)
- **ANFIS Library**: [Tim Meggs' ANFIS GitHub](https://github.com/twmeggs/anfis)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


