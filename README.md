ROA Prediction Optimization Project
Project Overview
This project implements three optimization algorithms—Genetic Algorithm (GA), Particle Swarm Optimization (PSO), and Adaptive Guided Differential Evolution (AdaGuiDE)—to predict Return on Assets (ROA) for companies in three industries: software, pharmaceuticals, and banking. The goal is to optimize weights for financial and ESG (Environmental, Social, Governance) factors and select mathematical operators (+, -, *, /) to construct a linear combination that minimizes the Mean Squared Error (MSE) between predicted and actual ROA values.
The project includes:

data.py: Contains datasets for software, pharmaceutical, and banking industries.
genetic_algorithm.py: Implements the GA to optimize weights and operators.
pso_algorithm.py: Implements the PSO algorithm for the same task.
adaguide_algorithm.py: Implements the AdaGuiDE algorithm, an adaptive differential evolution approach.

Each algorithm dynamically adapts to the number of factors in the selected dataset, supports flexible operator selection, and generates convergence and prediction plots to evaluate performance.
Datasets
The data.py file provides three datasets:

Software Industry (software_data):
5 factors: E1 (Opportunities in clean tech), E2 (Carbon emissions), S1 (Human capital development), S2 (Privacy & Data Security), G1 (Governance).
4 operators needed.
10 companies with ROA and factor values.


Pharmaceutical Industry (pharma_data):
5 factors: E1 (Toxic emissions & waste), S1 (Product safety & quality), S2 (Human capital development), S3 (Access to healthcare), G1 (Governance).
4 operators needed.
10 companies with ROA and factor values.


Banking Industry (banks_data):
6 factors: E1 (Financing Environmental Impact), S1 (Human Capital Development), S2 (Consumer Financial Protection), S3 (Access to Finance), S4 (Privacy & Data Security), G1 (Governance).
5 operators needed.
10 companies with ROA and factor values.



Each dataset is a dictionary with keys: Company (list of company names), ROA (list of ROA values), and factor keys (e.g., E1, S1, etc., with corresponding values). The algorithms dynamically detect the number of factors and operators based on the dataset.
How It Works

Dataset Selection:

Each algorithm script (genetic_algorithm.py, pso_algorithm.py, adaguide_algorithm.py) allows selecting a dataset via a dataset_key variable ('software', 'pharma', or 'banks').
The script loads the chosen dataset from data.py and determines the number of factors (excluding Company and ROA) and operators (factors - 1).


Optimization Process:

Objective: Minimize MSE between predicted and actual ROA, with a penalty for weights not summing to 1.
Variables:
Weights: One for each factor, normalized to sum to 1.
Operators: Mathematical operators (+, -, *, /) applied between weighted factors.


Algorithms:
GA: Uses selection, crossover, and mutation to evolve a population of solutions.
PSO: Updates particle positions and velocities based on personal and global best solutions.
AdaGuiDE: Employs adaptive differential evolution with multiple mutation strategies and boundary revision.




Output:

Prints MSE, optimized weights, operators, and sample predictions vs. actual ROA.
Generates two plots:
Convergence Plot: Shows MSE vs. function evaluations.
Predicted vs. Actual Plot: Scatter plot of predicted ROA vs. actual ROA with a perfect fit line.





Installation
Prerequisites

Python 3.6 or later.
pip for installing dependencies.

Dependencies
The project requires the following Python libraries:

numpy: For numerical computations.
scikit-learn: For calculating MSE.
matplotlib: For plotting results.

Install dependencies using pip:
pip install numpy scikit-learn matplotlib

Recommended: Use a Virtual Environment
To avoid conflicts with other Python projects, use a virtual environment:
python -m venv venv
source venv/bin/activate  # On Linux/macOS
venv\Scripts\activate     # On Windows
pip install numpy scikit-learn matplotlib

Usage

Clone the Repository:
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name


Ensure Files Are Present:

data.py
genetic_algorithm.py
pso_algorithm.py
adaguide_algorithm.py


Run an Algorithm:

Open the desired script (e.g., adaguide_algorithm.py) in a text editor.
Set the dataset_key to 'software', 'pharma', or 'banks':dataset_key = 'banks'


Run the script:python adaguide_algorithm.py


Repeat for other scripts (genetic_algorithm.py, pso_algorithm.py) and datasets.


Output:

Console output includes:
MSE of the best solution.
Optimized weights and operators.
Sum of weights (should be close to 1).
Sample predictions vs. actual ROA for the first 10 companies.


Two plots are displayed and can be saved manually:
Convergence plot (Convergence of [Algorithm] ([Dataset] Data)).
Predicted vs. actual ROA scatter plot.





Expected Plots and Results
Plots

Convergence Plot:

X-axis: Number of function evaluations.
Y-axis: Best MSE achieved so far.
Title: E.g., “Convergence of AdaGuiDE Algorithm (Banks Data)”.
Description: Shows how the algorithm’s best solution improves over evaluations. A downward trend indicates convergence.


Predicted vs. Actual ROA Plot:

X-axis: Actual ROA values.
Y-axis: Predicted ROA values.
Title: E.g., “Predicted vs Actual ROA (Banks Data)”.
Description: Scatter plot with a red dashed line (perfect fit). Points closer to the line indicate better predictions.



Results

MSE: Lower values indicate better model performance. Expected MSE varies by dataset:
software_data and pharma_data: Typically lower MSE due to 5 factors and narrower ROA ranges.
banks_data: May have higher MSE due to 6 factors and wider ROA range (-0.16 to 3.01).


Weights: Normalized to sum to approximately 1, representing the importance of each factor.
Operators: Sequence of +, -, *, / (e.g., ['+', '-', '*', '/'] for 5 factors).
Sample Predictions: Compare predicted ROA to actual ROA for the first 10 companies, showing model accuracy.

Performance Notes

Banks Dataset: May require tuning (e.g., increase pop_size to 50 or max_gen to 200) for better MSE due to its complexity.
Algorithm Comparison: GA, PSO, and AdaGuiDE may yield different MSEs. AdaGuiDE often performs well due to adaptive mutation strategies.

Additional Information
Project Structure
your-repo-name/
├── data.py                  # Datasets for software, pharma, and banks
├── genetic_algorithm.py     # Genetic Algorithm implementation
├── pso_algorithm.py         # Particle Swarm Optimization implementation
├── adaguide_algorithm.py    # AdaGuiDE implementation
├── README.md                # Project documentation

Customization

Add New Datasets:
Update data.py with a new dataset dictionary (e.g., new_data).
Add an entry to dataset_map in each algorithm script:dataset_map = {
    ...
    'new': {'data': new_data, 'name': 'New Industry'}
}




Tune Parameters:
Adjust pop_size and max_gen in each script for better convergence.
Modify penalty_factor in fitness_function to change the weight sum penalty.



Troubleshooting

Encoding Issues:
Ensure all .py files are saved in UTF-8 encoding (use VS Code, PyCharm, or Notepad++).
Check for invalid characters (e.g., �) in files using a text editor.


Module Not Found:
Verify dependencies are installed in the active Python environment.
Run pip list to confirm numpy, scikit-learn, and matplotlib are present.


Dataset Errors:
Ensure data.py contains valid datasets with Company, ROA, and factor columns.
Check dataset_key is set to a valid option ('software', 'pharma', 'banks').



Contributing

Fork the repository and submit pull requests for improvements.
Report issues or suggest features via GitHub Issues.

License
This project is licensed under the MIT License. See LICENSE for details.
Contact
For questions or feedback, contact [your-email@example.com] or open an issue on GitHub.
