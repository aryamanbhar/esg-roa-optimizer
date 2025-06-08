import random
import numpy as np
import operator
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from data import software_data, pharmaceuticals_data, banks_data

# Set fixed random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Select dataset using a string identifier
dataset_key = 'banks'  # Options: 'software', 'pharmaceuticals', 'banks'

# Map dataset keys to datasets and display names
dataset_map = {
    'software': {'data': software_data, 'name': 'Software'},
    'pharmaceuticals': {'data': pharmaceuticals_data, 'name': 'Pharmaceuticals'},
    'banks': {'data': banks_data, 'name': 'Banks'}
}

# Validate dataset selection
if dataset_key not in dataset_map:
    raise ValueError(f"Invalid dataset key: {dataset_key}. Choose from {list(dataset_map.keys())}")

dataset = dataset_map[dataset_key]['data']
dataset_name = dataset_map[dataset_key]['name']

# Dynamically determine number of factors (excluding 'Company' and 'ROA')
factor_keys = [k for k in dataset.keys() if k not in ['Company', 'ROA']]
NUM_FACTORS = len(factor_keys)
NUM_OPERATORS = NUM_FACTORS - 1
DIM = NUM_FACTORS + NUM_OPERATORS  # Total dimensions (weights + operators)

# Prepare data
ESG_scores = np.array([list(v) for v in zip(*[dataset[k] for k in factor_keys])])
ROA_actual = np.array(dataset['ROA'])

operator_map = {0: operator.add, 1: operator.sub, 2: operator.mul, 3: lambda x, y: x / (y + 1e-6)}

# PSO Parameters
POP_SIZE = 200
MAX_EVALUATIONS = 40000
W = 0.8  # Inertia weight
C1 = 0.5  # Cognitive coefficient (personal best)
C2 = 0.5  # Social coefficient (global best)
NUM_RUNS = 10

# PSO Implementation
class PSO:
    def __init__(self, dim=DIM, bounds_weights=[0, 1], bounds_operators=[0, 3]):
        self.dim = dim  # Dimensions: NUM_FACTORS weights + NUM_OPERATORS operators
        self.bounds_weights = bounds_weights
        self.bounds_operators = bounds_operators
        self.pop_size = POP_SIZE
        self.max_evals = MAX_EVALUATIONS
        self.w = W
        self.c1 = C1
        self.c2 = C2
        
        self.X = np.zeros((self.pop_size, self.dim))
        self.V = np.zeros((self.pop_size, self.dim))
        self.pbest_X = np.zeros((self.pop_size, self.dim))
        self.pbest_fitness = np.full(self.pop_size, float('inf'))
        self.gbest_X = np.zeros(self.dim)
        self.gbest_fitness = float('inf')
        
        self.eval_count = 0
        self.convergence_data = []
        self.best_history = []

    def normalize_weights(self, weights):
        weights = np.maximum(weights, 0)
        weight_sum = np.sum(weights)
        if weight_sum == 0:
            return np.ones_like(weights) / len(weights)
        return weights / weight_sum

    def initialize(self):
        for i in range(self.pop_size):
            weights = np.random.uniform(self.bounds_weights[0], self.bounds_weights[1], NUM_FACTORS)
            self.X[i, :NUM_FACTORS] = self.normalize_weights(weights)
            self.X[i, NUM_FACTORS:] = np.random.uniform(self.bounds_operators[0], self.bounds_operators[1], NUM_OPERATORS)
            self.V[i, :NUM_FACTORS] = np.random.uniform(-0.1, 0.1, NUM_FACTORS)
            self.V[i, NUM_FACTORS:] = np.random.uniform(-1, 1, NUM_OPERATORS)
        
        for i in range(self.pop_size):
            fitness = self.evaluate(self.X[i])
            self.pbest_X[i] = self.X[i].copy()
            self.pbest_fitness[i] = fitness
            if fitness < self.gbest_fitness:
                self.gbest_fitness = fitness
                self.gbest_X = self.X[i].copy()
            self.eval_count += 1
            self.convergence_data.append((self.eval_count, self.gbest_fitness))
            if i == 0 or fitness < min([h[1] for h in self.best_history]):
                self.best_history.append((self.eval_count, fitness, self.X[i].copy()))

    def evaluate(self, particle):
        weights = self.normalize_weights(particle[:NUM_FACTORS])
        op_indices = [int(x) % 4 for x in particle[NUM_FACTORS:]]
        ROA_predicted = []
        for stock in ESG_scores:
            try:
                result = stock[0] * weights[0]
                for i in range(1, NUM_FACTORS):
                    op = operator_map[op_indices[i-1]]
                    result = op(result, stock[i] * weights[i])
                ROA_predicted.append(result)
            except ZeroDivisionError:
                ROA_predicted.append(0)
        return mean_squared_error(ROA_actual, ROA_predicted)

    def run(self):
        self.initialize()
        
        while self.eval_count < self.max_evals:
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                self.V[i] = (self.w * self.V[i] + 
                             self.c1 * r1 * (self.pbest_X[i] - self.X[i]) + 
                             self.c2 * r2 * (self.gbest_X - self.X[i]))
                
                self.X[i] += self.V[i]
                self.X[i, :NUM_FACTORS] = self.normalize_weights(self.X[i, :NUM_FACTORS])
                self.X[i, NUM_FACTORS:] = np.clip(self.X[i, NUM_FACTORS:], self.bounds_operators[0], self.bounds_operators[1])
                
                fitness = self.evaluate(self.X[i])
                self.eval_count += 1
                
                if fitness < self.pbest_fitness[i]:
                    self.pbest_fitness[i] = fitness
                    self.pbest_X[i] = self.X[i].copy()
                
                if fitness < self.gbest_fitness:
                    self.gbest_fitness = fitness
                    self.gbest_X = self.X[i].copy()
                    self.best_history.append((self.eval_count, fitness, self.X[i].copy()))
                
                self.convergence_data.append((self.eval_count, self.gbest_fitness))
                
                if self.eval_count >= self.max_evals:
                    break
        
        return self.gbest_X, self.gbest_fitness, self.convergence_data, self.best_history

# Run PSO multiple times
best_overall = None
best_fitness = float('inf')
best_convergence = None
best_run_history = None

for run in range(NUM_RUNS):
    random.seed(42 + run)
    np.random.seed(42 + run)
    
    pso = PSO()
    gbest_X, gbest_fitness, convergence, history = pso.run()
    
    print(f"Run {run+1} best fitness: {gbest_fitness:.6f}")
    
    if gbest_fitness < best_fitness:
        best_fitness = gbest_fitness
        best_overall = gbest_X.copy()
        best_convergence = convergence
        best_run_history = history

# Display results
print("\nBest Overall Solution:")
print(f"MSE: {best_fitness:.6f}")
print("Weights:", [f"{w:.4f}" for w in best_overall[:NUM_FACTORS]])
print("Operators:", [operator_map[int(i)%4].__name__ if hasattr(operator_map[int(i)%4], '__name__') else "div" 
                     for i in best_overall[NUM_FACTORS:]])
print(f"Sum of weights: {np.sum(best_overall[:NUM_FACTORS]):.6f}")

# Calculate predictions
def calculate_predictions(particle, data):
    weights = particle[:NUM_FACTORS]
    op_indices = [int(x) % 4 for x in particle[NUM_FACTORS:]]
    predictions = []
    for stock in data:
        try:
            result = stock[0] * weights[0]
            for i in range(1, NUM_FACTORS):
                op = operator_map[op_indices[i-1]]
                result = op(result, stock[i] * weights[i])
            predictions.append(result)
        except ZeroDivisionError:
            predictions.append(0)
    return predictions

predictions = calculate_predictions(best_overall, ESG_scores)

# Show model performance
print("\nModel Performance:")
print(f"MSE: {mean_squared_error(ROA_actual, predictions):.6f}")
print("\nSample predictions vs actual:")
for i in range(min(10, len(ROA_actual))):
    print(f"Company: {dataset['Company'][i]}, Actual: {ROA_actual[i]:.4f}, Predicted: {predictions[i]:.4f}")

# Plot convergence
evals, fitnesses = zip(*best_convergence)
plt.figure(figsize=(10, 6))
plt.plot(evals, fitnesses, 'b-', label='Best MSE')
plt.xlabel('Function Evaluations')
plt.ylabel('MSE')
plt.title(f'Convergence of PSO ({dataset_name} Data - Best Run)')
plt.legend()
plt.grid(True)
plt.show()

# Plot predicted vs actual
plt.figure(figsize=(10, 6))
plt.scatter(ROA_actual, predictions, c='blue', label='Predicted vs Actual')
plt.plot([min(ROA_actual), max(ROA_actual)], [min(ROA_actual), max(ROA_actual)], 'r--', label='Perfect Fit')
plt.xlabel('Actual ROA')
plt.ylabel('Predicted ROA')
plt.title(f'Predicted vs Actual ROA ({dataset_name} Data)')
plt.legend()
plt.grid(True)
plt.show()

# Display best solution history
print("\nBest Solution History (Evaluations, Fitness, Weights):")
for evals, fit, ind in best_run_history[:5]:
    print(f"Evals: {evals}, Fitness: {fit:.6f}, Weights: {[f'{w:.4f}' for w in ind[:NUM_FACTORS]]}")