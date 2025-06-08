# ESG-Driven ROA Prediction Optimization Project

## Project Overview
This project implements three optimization algorithms—**Genetic Algorithm (GA)**, **Particle Swarm Optimization (PSO)**, and **Adaptive Guided Differential Evolution (AdaGuiDE)**—to predict **Return on Assets (ROA)** for companies in three industries: software, pharmaceuticals, and banking. The goal is to optimize weights for financial and ESG (Environmental, Social, Governance) factors and select mathematical operators (+, -, *, /) to construct a linear combination that minimizes the Mean Squared Error (MSE) between predicted and actual ROA values.

The project includes:
- `data.py`: Contains datasets for software, pharmaceutical, and banking industries.
- `genetic_algorithm.py`: Implements the GA to optimize weights and operators.
- `pso_algorithm.py`: Implements the PSO algorithm for the same task.
- `adaguide_algorithm.py`: Implements the AdaGuiDE algorithm, an adaptive differential evolution approach.

Each algorithm dynamically adapts to the number of factors in the selected dataset, supports flexible operator selection, and generates convergence and prediction plots to evaluate performance.

## Datasets
The `data.py` file provides three datasets:
- **Software Industry (`software_data`)**:
  - 5 factors: E1 (Financing), S1 (Human), S2 (Consumer), S3 (Access), G1 (Governance).
  - 4 operators needed.
  - 10 companies with ROA and factor values.
- **Pharmaceutical Industry (`pharma_data`)**:
  - 5 factors: E1 (Financing), S1 (Human), S2 (Consumer), S3 (Access), G1 (Governance).
  - 4 operators needed.
  - 10 companies with ROA and factor values.
- **Banking Industry (`banks_data`)**:
  - 6 factors: E1 (Financing), S1 (Human), S2 (Consumer), S3 (Access), S4 (Privacy), G1 (Governance).
  - 5 operators needed.
  - 10 companies with ROA and factor values.

Each dataset is a dictionary with keys: `Company` (list of company names), `ROA` (list of ROA values), and factor keys (e.g., `E1`, `S1`, etc., with corresponding values). The algorithms dynamically detect the number of factors and operators based on the dataset.

## How It Works
1. **Dataset Selection**:
   - Each algorithm script (`genetic_algorithm.py`, `pso_algorithm.py`, `adaguide_algorithm.py`) allows selecting a dataset via a `dataset_key` variable (`'software'`, `'pharma'`, or `'banks'`).
   - The script loads the chosen dataset from `data.py` and determines the number of factors (excluding `Company` and `ROA`) and operators (factors - 1).

2. **Optimization Process**:
   - **Objective**: Minimize MSE between predicted and actual ROA, with a penalty for weights not summing to 1.
   - **Variables**:
     - Weights: One for each factor, normalized to sum to 1.
     - Operators: Mathematical operators (+, -, *, /) applied between weighted factors.
   - **Algorithms**:
     - **GA**: Uses selection, crossover, and mutation to evolve a population of solutions.
     - **PSO**: Updates particle positions and velocities based on personal and global best solutions.
     - **AdaGuiDE**: Employs adaptive differential evolution with multiple mutation strategies and boundary revision.

3. **Output**:
   - Prints MSE, optimized weights, operators, and sample predictions vs. actual ROA.
   - Generates two plots:
     - **Convergence Plot**: Shows MSE vs. function evaluations.
     - **Predicted vs. Actual Plot**: Scatter plot of predicted ROA vs. actual ROA with a perfect fit line.

## Installation
### Prerequisites
- Python 3.6 or later.
- `pip` for installing dependencies.

### Dependencies
The project requires the following Python libraries:
- `numpy`: For numerical computations.
- `scikit-learn`: For calculating MSE.
- `matplotlib`: For plotting results.

Install dependencies using `pip`:
```bash
pip install numpy scikit-learn matplotlib
```

### Recommended: Use a Virtual Environment
To avoid conflicts with other Python projects, use a virtual environment:
```bash
python -m venv venv
source venv
