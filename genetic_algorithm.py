import random
import numpy as np
import operator
from deap import base, creator, tools, algorithms
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from data import software_data, pharmaceuticals_data, banks_data

# Set fixed random seeds for reproducibility
random.seed(42)
np.random.seed(42)

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

# Prepare data
ESG_scores = np.array([list(v) for v in zip(*[dataset[k] for k in factor_keys])])
ROA_actual = np.array(dataset['ROA'])

operator_map = {0: operator.add, 1: operator.sub, 2: operator.mul, 3: lambda x, y: x / (y + 1e-6)}

# Clear any existing DEAP classes
if 'FitnessMin' in creator.__dict__:
    del creator.FitnessMin
if 'Individual' in creator.__dict__:
    del creator.Individual

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

def normalize_weights(weights):
    weights = np.maximum(weights, 0)
    weight_sum = np.sum(weights)
    if weight_sum == 0:
        return np.ones_like(weights) / len(weights)
    return weights / weight_sum

def evaluate(individual):
    weights = normalize_weights(individual[:NUM_FACTORS])
    op_indices = [int(x) % 4 for x in individual[NUM_FACTORS:]]
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
    mse = mean_squared_error(ROA_actual, ROA_predicted)
    return (mse,)

def create_individual():
    weights = [random.uniform(0, 1) for _ in range(NUM_FACTORS)]
    weights = normalize_weights(weights)
    operators = [random.randint(0, 3) for _ in range(NUM_OPERATORS)]
    return creator.Individual(list(weights) + operators)

def custom_crossover(ind1, ind2):
    tools.cxTwoPoint(ind1, ind2)
    ind1[:NUM_FACTORS] = normalize_weights(ind1[:NUM_FACTORS])
    ind2[:NUM_FACTORS] = normalize_weights(ind2[:NUM_FACTORS])
    return ind1, ind2

def perturbation_mutation(individual, pert_rate=0.1, indpb=0.2):
    for i in range(NUM_FACTORS):
        if random.random() < indpb:
            perturbation = random.uniform(-pert_rate, pert_rate)
            individual[i] = max(0, individual[i] * (1 + perturbation))
    individual[:NUM_FACTORS] = normalize_weights(individual[:NUM_FACTORS])
    for i in range(NUM_FACTORS, len(individual)):
        if random.random() < indpb:
            individual[i] = random.randint(0, 3)
    return individual,

toolbox = base.Toolbox()
toolbox.register("individual", create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", custom_crossover)
toolbox.register("mutate", perturbation_mutation, pert_rate=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# Optimization settings
POP_SIZE = 200
MAX_EVALUATIONS = 40000
CXPB, MUTPB = 0.7, 0.2
NUM_RUNS = 10

def eaSimpleWithEvalCount(population, toolbox, cxpb, mutpb, max_evals, stats=None, verbose=False):
    logbook = tools.Logbook()
    logbook.header = ['evals', 'gen'] + (stats.fields if stats else [])
    
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    
    eval_count = len(population)
    gen = 0
    best_history = []
    convergence_data = []
    
    best = tools.selBest(population, 1)[0]
    best_history.append((eval_count, best.fitness.values[0], best[:]))
    
    while eval_count < max_evals:
        gen += 1
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        eval_count += len(invalid_ind)
        population[:] = offspring
        
        if stats:
            record = stats.compile(population)
            logbook.record(evals=eval_count, gen=gen, **record)
            if verbose:
                print(logbook.stream)
        
        current_best = tools.selBest(population, 1)[0]
        if current_best.fitness.values[0] < best.fitness.values[0]:
            best = current_best
            best_history.append((eval_count, best.fitness.values[0], best[:]))
        convergence_data.append((eval_count, best.fitness.values[0]))
    
    return population, logbook, best_history, convergence_data

# Track best overall solution
best_overall = None
best_fitness = float('inf')
best_run_history = None
best_convergence = None

for run in range(NUM_RUNS):
    random.seed(42 + run)
    np.random.seed(42 + run)
    
    pop = toolbox.population(n=POP_SIZE)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    
    pop, logbook, history, convergence = eaSimpleWithEvalCount(
        pop, toolbox, cxpb=CXPB, mutpb=MUTPB, max_evals=MAX_EVALUATIONS, 
        stats=stats, verbose=False
    )
    
    best_ind_run = tools.selBest(pop, 1)[0]
    print(f"Run {run+1} best fitness: {best_ind_run.fitness.values[0]:.6f}")
    
    if best_ind_run.fitness.values[0] < best_fitness:
        best_fitness = best_ind_run.fitness.values[0]
        best_overall = best_ind_run.copy()
        best_run_history = history
        best_convergence = convergence

# Display results
print("\nBest Overall Solution:")
print(f"MSE: {best_fitness:.6f}")
print("Weights:", [f"{w:.4f}" for w in best_overall[:NUM_FACTORS]])
print("Operators:", [operator_map[int(i)%4].__name__ if hasattr(operator_map[int(i)%4], '__name__') else "div" 
                     for i in best_overall[NUM_FACTORS:]])
print(f"Sum of weights: {np.sum(best_overall[:NUM_FACTORS]):.6f}")

# Calculate predictions
def calculate_predictions(individual, data):
    weights = normalize_weights(individual[:NUM_FACTORS])
    op_indices = [int(x) % 4 for x in individual[NUM_FACTORS:]]
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
plt.title(f'Convergence of Genetic Algorithm ({dataset_name} Data - Best Run)')
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