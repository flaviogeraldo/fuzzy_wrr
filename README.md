# ANFIS-based Fuzzy Logic System for WRR Weight Adjustment

Welcome to the **ANFIS-based Fuzzy Logic System**! This project implements a Fuzzy Rule-Based System (FRBS) with an **Adaptive Neuro-Fuzzy Inference System (ANFIS)** for adjusting Weighted Round Robin (WRR) weights dynamically based on buffer occupancies. 

## Overview

This project integrates **ANFIS** to improve the decision-making of a fuzzy logic system in a networking environment. The system adjusts WRR weights based on the current buffer occupancies, allowing for adaptive and dynamic control.

- **ANFIS** is used to train the system to learn from the data and predict WRR weights adjustments.
- The project leverages **skfuzzy** for implementing fuzzy logic and visualizing membership functions.
- Functions of pertinence are dynamically adjusted based on predicted values from the ANFIS system.

## Key Features

- **Adaptive Membership Functions:** Automatically adjust the membership functions based on real-time data.
- **ANFIS Integration:** Leverage ANFIS for training the system to improve predictions.
- **Weighted Round Robin Adjustment:** Dynamically adjust WRR weights based on buffer conditions.
- **Fuzzy Logic Rules:** Easily configurable fuzzy rules to determine actions based on buffer occupancy.

## Project Structure

