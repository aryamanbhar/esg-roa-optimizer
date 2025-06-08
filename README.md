# ESG-Driven ROA Optimization

# Project Description
This project applies three metaheuristic optimization algorithms—**Genetic Algorithm (GA)**, **Particle Swarm Optimization (PSO)**, and **Adaptive Guided Differential Evolution (AdaGuiDE)**—to optimize **Return on Assets (ROA)** predictions for companies in the software, pharmaceutical, and banking industries. By assigning weights to industry-specific Environmental, Social, and Governance (ESG) factors and selecting mathematical operators (+, -, *, /), the algorithms minimize the Mean Squared Error (MSE) between predicted and actual ROA, revealing key ESG-financial relationships.

Files included:
- `data.py`: Datasets for software, pharmaceutical, and banking industries.
- `genetic_algorithm.py`: GA implementation.
- `pso_algorithm.py`: PSO implementation.
- `adaguide_algorithm.py`: AdaGuiDE implementation.

Each algorithm dynamically adapts to the dataset’s factor count, optimizes weights and operators, and generates plots to evaluate performance, offering actionable insights for sustainable investing.

# Datasets
The `data.py` file contains datasets for 30 companies in each of three industries, collected in 2024:
- **Software (`software_data`)**:
  - Factors: 5 (E1: Opportunities in Clean Tech, E2: Carbon Emissions, S1: Human Capital Development, S2: Privacy and Data Security, G1: Governance).
  - Operators: 4.
  - Companies: 30, with ROA and factor values.
- **Pharmaceutical (`pharma_data`)**:
  - Factors: 5 (E1: Toxic Emissions and Waste, S1: Product Safety and Quality, S2: Human Capital Development, S3: Access to Healthcare, G1: Governance).
  - Operators: 4.
  - Companies: 30.
- **Banking (`banks_data`)**:
  - Factors: 6 (E1: Financing Environmental Impact, S1: Human Capital Development, S2: Consumer Privacy and Data, S3: Access to Finance, S4: Privacy and Data Security, G1: Governance).
  - Operators: 5.
  - Companies: 30.

Data was sourced from the **MSCI Index** for 2024 ESG performance scores and **Yahoo Finance** for ROA values. The MSCI Index provided factor-level ESG ratings to identify material factors for each industry (e.g., Carbon Emissions for software, Product Safety for pharmaceuticals) and assess company performance, normalized (0 for laggards, 0.5 for average, 1 for leaders) for optimization. Each dataset is a dictionary with keys: `Company` (company names), `ROA` (ROA values), and factors (e.g., `E1`, `S1`). Algorithms dynamically detect factors and operators.

# How It Works
1. **Select Dataset**:
   - Set `dataset_key` in the script to `'software'`, `'pharma'`, or `'banks'`.
   - The script loads the dataset and calculates factors and operators.

2. **Optimize**:
   - **Goal**: Minimize MSE, with a penalty if weights don’t sum to 1.
   - **Variables**:
     - Weights: One per factor, normalized to sum to 1.
     - Operators: +, -, *, / between weighted factors.
   - **Algorithms**:
     - GA: Evolves solutions via selection, crossover, mutation.
     - PSO: Updates particles based on best positions.
     - AdaGuiDE: Uses adaptive differential evolution with mutation strategies.

3. **Output**:
   - Console: MSE, weights, operators, sample predictions vs. actual ROA.
   - Plots:
     - Convergence (MSE vs. evaluations).
     - Predicted vs. actual ROA (scatter plot).

# Installation
## Requirements
- Python 3.6+.
- `pip` for installing dependencies.

## Install Dependencies
Run:
```bash
pip install numpy scikit-learn matplotlib
```

## Optional: Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
pip install numpy scikit-learn matplotlib
```

# Usage
1. **Clone Repository**:
   ```bash
   git clone https://github.com/your-username/esg-roa-opt.git
   cd esg-roa-opt
   ```

2. **Check Files**:
   - `data.py`
   - `genetic_algorithm.py`
   - `pso_algorithm.py`
   - `adaguide_algorithm.py`

3. **Change Dataset**:
   - Open a script (e.g., `adaguide_algorithm.py`) in a text editor.
   - Modify the `dataset_key` line to choose a dataset:
     ```python
     dataset_key = 'banks'  # Change to 'software' or 'pharma' if needed
     ```

4. **Run Script**:
   ```bash
   python adaguide_algorithm.py
   ```
   Repeat for other scripts and datasets.

5. **View Results**:
   - Console shows MSE, weights, operators, and predictions.
   - Plots appear in a window (save manually if needed).

# Expected Results
## Plots
- **Convergence Plot**:
  - X-axis: Function evaluations.
  - Y-axis: Best MSE.
  - Title: E.g., “Convergence of AdaGuiDE Algorithm (Banks Data)”.
  - Shows MSE decreasing over time.
- **Predicted vs. Actual ROA**:
  - X-axis: Actual ROA.
  - Y-axis: Predicted ROA.
  - Title: E.g., “Predicted vs Actual ROA (Banks Data)”.
  - Scatter plot with a red dashed line (perfect fit).

## Results
- **MSE**: Lower is better. `banks_data` may have higher MSE (e.g., ~0.452) due to 6 factors and wider ROA range (-0.16 to 3.01) compared to software (~0.009) or pharmaceuticals (~0.008).
- **Weights**: Sum to ~1, show factor importance (e.g., Financing Environmental Impact dominates in banking).
- **Operators**: E.g., `['+', '-', '*', '/']` for 5 factors, revealing interactions like multiplicative effects in software.
- **Predictions**: Compare actual vs. predicted ROA for 30 companies.

# Notes
- **Tuning**: For `banks_data`, try `pop_size=50` or `max_gen=200` for better MSE.
- **Add Datasets**: Update `data.py` and `dataset_map` in scripts.
- **Troubleshooting**:
  - Save files in UTF-8 (VS Code, Notepad++).
  - Check for `�` or odd characters.
  - Ensure `data.py` has valid datasets.
- **Data Source**: MSCI ESG ratings and Yahoo Finance ROA data were collected for 2024, ensuring relevance and accuracy.

# License
MIT License. See [LICENSE](LICENSE) for details.

# Contact
Open a GitHub issue or email [your-email@example.com].