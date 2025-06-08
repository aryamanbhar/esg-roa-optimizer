import random
import numpy as np
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
factors = [k for k in dataset.keys() if k not in ['Company', 'ROA']]
NUM_FACTORS = len(factors)
NUM_OPERATORS = NUM_FACTORS - 1

# Prepare data
factor_arrays = {k: np.array(dataset[k]) for k in factors}
ROA_actual = np.array(dataset['ROA'])

# Define operators pool
OPERATORS = {
    '+': lambda x, y: x + y,
    '-': lambda x, y: x - y,
    '*': lambda x, y: x * y,
    '/': lambda x, y: x / (y + 1e-8)  # Avoid division by zero
}

# DE operator pool for mutation (inspired by ADAGUIDE)
DE_OPERATORS = {
    'CSDE_CurrentToPbest': lambda x, pbest, r1, r2, F: x + F * (pbest - x) + F * (r1 - r2),
    'CSDE_PbestToRand': lambda x, pbest, r1, r3, F: r1 + F * (pbest - r3),
    'SADE_CurrentToRand': lambda x, r1, r2, r3, F: r1 + F * (r2 - r3),
    'AMPO_Local': lambda x, pbest, r1, F: x + F * (pbest - r1)  # Simplified local search
}

class AdaGuiDE:
    def __init__(self, pop_size, max_gen, bound=(0, 1)):
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.dim = NUM_FACTORS  # Number of factors (weights)
        self.num_operators = NUM_OPERATORS  # Number of operators
        self.bound = bound
        self.population = self.initialize_population()
        self.success_rates = {op: 0.5 for op in DE_OPERATORS.keys()}  # Initial success rates
        self.best_solution = None
        self.best_fitness = float('inf')
        self.fitness_history = []
        self.evaluation_count = 0

    def compute_roa(self, weights, operators):
        result = weights[0] * factor_arrays[factors[0]]
        for i in range(1, self.dim):
            result = OPERATORS[operators[i-1]](result, weights[i] * factor_arrays[factors[i]])
        return result

    def fitness_function(self, weights, operators, penalty_factor=0.1):
        ROA_predicted = self.compute_roa(weights, operators)
        mse = np.mean((ROA_actual - ROA_predicted) ** 2)
        sum_penalty = abs(np.sum(weights) - 1) * penalty_factor
        self.evaluation_count += 1
        return mse + sum_penalty

    def initialize_population(self):
        population = []
        for _ in range(self.pop_size):
            weights = np.random.uniform(*self.bound, self.dim)
            weights = weights / np.sum(weights)  # Normalize to sum to 1
            operators = [random.choice(list(OPERATORS.keys())) for _ in range(self.num_operators)]
            population.append((weights, operators))
        return population

    def apply_de_operator(self, op_name, x, pbest, r1, r2, r3, F):
        if op_name == 'SADE_CurrentToRand':
            return DE_OPERATORS[op_name](x, r1, r2, r3, F)
        elif op_name == 'CSDE_PbestToRand':
            return DE_OPERATORS[op_name](x, pbest, r1, r3, F)
        elif op_name == 'AMPO_Local':
            return DE_OPERATORS[op_name](x, pbest, r1, F)
        else:  # CSDE_CurrentToPbest
            return DE_OPERATORS[op_name](x, pbest, r1, r2, F)

    def mutate(self, target, pbest, F):
        r1, r2, r3 = [self.population[i][0] for i in np.random.choice(len(self.population), 3, replace=False)]
        op_name = random.choices(list(DE_OPERATORS.keys()), weights=[self.success_rates[op] for op in DE_OPERATORS.keys()], k=1)[0]
        mutated_weights = self.apply_de_operator(op_name, target[0], pbest, r1, r2, r3, F)
        mutated_weights = np.clip(mutated_weights, 0, None)  # Ensure non-negative
        mutated_weights = mutated_weights / np.sum(mutated_weights) if np.sum(mutated_weights) > 0 else np.ones(self.dim) / self.dim
        mutated_operators = [random.choice(list(OPERATORS.keys())) if random.random() < 0.1 else op for op in target[1]]
        return mutated_weights, mutated_operators, op_name

    def crossover(self, target, mutant, CR):
        weights = np.where(np.random.rand(self.dim) < CR, mutant[0], target[0])
        weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones(self.dim) / self.dim
        operators = [mutant[1][i] if random.random() < CR else target[1][i] for i in range(self.num_operators)]
        return weights, operators

    def revise_boundaries(self, threshold=0.1):
        weights = np.array([ind[0] for ind in self.population])
        lower = np.maximum(self.bound[0], np.min(weights, axis=0) - threshold)
        upper = np.minimum(self.bound[1], np.max(weights, axis=0) + threshold)
        self.bound = (lower, upper)

    def evolve(self):
        max_evaluations = self.max_gen * self.pop_size
        for gen in range(self.max_gen):
            if self.evaluation_count >= max_evaluations:
                break
            F = 0.5 + 0.5 * (1 - gen / self.max_gen)  # Dynamic F
            CR = 0.7 + 0.2 * (gen / self.max_gen)    # Dynamic CR
            pbest_idx = np.argmin([self.fitness_function(*ind) for ind in self.population])
            pbest = self.population[pbest_idx][0]

            new_population = []
            for target in self.population:
                mutant, mutant_ops, op_name = self.mutate(target, pbest, F)
                trial = self.crossover(target, (mutant, mutant_ops), CR)
                trial_fitness = self.fitness_function(*trial)
                target_fitness = self.fitness_function(*target)

                if trial_fitness < target_fitness:
                    new_population.append(trial)
                    self.success_rates[op_name] += 0.01
                else:
                    new_population.append(target)

            self.population = new_population
            current_best_fitness = min([self.fitness_function(*ind) for ind in self.population])
            if current_best_fitness < self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_solution = self.population[np.argmin([self.fitness_function(*ind) for ind in self.population])]

            self.fitness_history.append((self.evaluation_count, self.best_fitness))
            print(f"Generation {gen+1}, Evaluations: {self.evaluation_count}, Best Fitness: {self.best_fitness:.6f}")

            if gen % 10 == 0:  # Revise boundaries every 10 generations
                self.revise_boundaries()

        total = sum(self.success_rates.values())
        self.success_rates = {k: v / total for k, v in self.success_rates.items()}
        return self.best_solution, self.fitness_history

# Run AdaGuiDE
optimizer = AdaGuiDE(pop_size=30, max_gen=100)
best_solution, fitness_history = optimizer.evolve()

# Calculate predictions
best_weights, best_operators = best_solution
predictions = optimizer.compute_roa(best_weights, best_operators)

# Show model performance
print("\nModel Performance:")
print(f"MSE: {mean_squared_error(ROA_actual, predictions):.6f}")
print("\nBest Overall Solution:")
print("Best Weights:", [f"{w:.4f}" for w in best_weights])
print("Best Operators:", best_operators)
print(f"Sum of Weights: {np.sum(best_weights):.6f}")
print("\nSample predictions vs actual:")
for i in range(min(10, len(ROA_actual))):
    print(f"Company: {dataset['Company'][i]}, Actual: {ROA_actual[i]:.4f}, Predicted: {predictions[i]:.4f}")

# Plot convergence
evals, fitnesses = zip(*fitness_history)
plt.figure(figsize=(10, 6))
plt.plot(evals, fitnesses, 'b-', label='Best MSE')
plt.xlabel('Function Evaluations')
plt.ylabel('MSE')
plt.title(f'Convergence of AdaGuiDE Algorithm ({dataset_name} Data)')
plt.legend()
plt.grid(True)
plt.show()

# Plot predicted vs actual ROA
plt.figure(figsize=(10, 6))
plt.scatter(ROA_actual, predictions, c='blue', label='Predicted vs Actual')
plt.plot([min(ROA_actual), max(ROA_actual)], [min(ROA_actual), max(ROA_actual)], 'r--', label='Perfect Fit')
plt.xlabel('Actual ROA')
plt.ylabel('Predicted ROA')
plt.title(f'Predicted vs Actual ROA ({dataset_name} Data)')
plt.legend()
plt.grid(True)
plt.show()