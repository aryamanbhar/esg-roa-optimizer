import random
import numpy as np
import operator
from deap import base, creator, tools, algorithms
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Set fixed random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# ESG factors and ROA from actual data (banks_data)
# banks_data = {
#     'Company': ['JPM', 'BAC', 'WFC', 'C', 'PNC', 'USB', 'TFC', 'MTB', 'FITB', 'HBAN', 
#                 'CFG', 'FITB', 'ALLY', 'RF', 'DFS', 'KEY', 'FCNCA', 'Royal bank of Canada', 
#                 'Capital One', 'Toronto-Dominion', 'Bank of Montreal', 'National Bank of Canada', 
#                 'BNP', 'Barclays', 'Societe Generale', 'Standard Chartered', 'HSBC', 'Mizuho', 
#                 'Synchrony Financial', 'CM'],
#     'E1 (Financing)': [1, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 0.5, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     'S1 (Human)': [0, 1, 0, 0, 0.5, 0.5, 1, 0.5, 1, 0.5, 0.5, 1, 0.5, 0.5, 0.5, 0, 0, 0, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 0.5, 1, 0.5],
#     'S2 (Consumer)': [1, 1, 0.5, 1, 0.5, 1, 1, 0.5, 1, 0.5, 0, 0, 0.5, 0.5, 0.5, 1, 0.5, 0.5, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0],
#     'S3 (Access)': [0, 0, 0, 0, 1, 0.5, 0.5, 0.5, 0, 0.5, 0.5, 1, 0, 0, 0.5, 0.5, 0.5, 0.5, 1, 0.5, 0.5, 0.5, 1, 1, 1, 1, 1, 0, 0, 0.5],
#     'S4 (Privacy)': [1, 0.5, 0.5, 1, 1, 0.5, 0.5, 0.5, 1, 0.5, 0.5, 0.5, 1, 0.5, 0.5, 1, 1, 0, 1, 0.5, 0.5, 0.5, 1, 1, 1, 1, 0.5, 1, 1, 0.5],
#     'G1': [0, 0, 1, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 0.5, 0.5, 0.5, 0.5, 1, 0.5, 0.5, 0.5, 1, 0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     'ROA': [1.42, 0.78, 0.96, 0.49, 0.98, 0.87, 0.84, 1.17, 1.01, 0.88, 0.63, 1.01, 0.29, 1.12, 3.01, -0.16, 1.21, 0.80, 0.90, 0.40, 0.49, 0.80, 0.40, 0.35, 0.22, 0.42, 0.75, 0.32, 2.86, 0.65]
# }

# ESG_scores = np.array([list(v) for v in zip(*[banks_data[k] for k in list(banks_data.keys())[1:-1]])])
# ROA_actual = np.array(banks_data['ROA'])

# operator_map = {0: operator.add, 1: operator.sub, 2: operator.mul, 3: lambda x, y: x / (y + 1e-6)}

# # Clear any existing DEAP classes
# if 'FitnessMin' in creator.__dict__:
#     del creator.FitnessMin
# if 'Individual' in creator.__dict__:
#     del creator.Individual

# creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
# creator.create("Individual", list, fitness=creator.FitnessMin)

# def normalize_weights(weights):
#     """Normalize weights to sum to 1, ensuring non-negative values."""
#     weights = np.maximum(weights, 0)
#     weight_sum = np.sum(weights)
#     if weight_sum == 0:
#         return np.ones_like(weights) / len(weights)
#     return weights / weight_sum

# def evaluate(individual):
#     weights = normalize_weights(individual[:6])  # Normalize weights before evaluation
#     op_indices = [int(x) % 4 for x in individual[6:]]
#     ROA_predicted = []
#     for stock in ESG_scores:
#         try:
#             result = stock[0] * weights[0]
#             for i in range(1, 6):
#                 op = operator_map[op_indices[i-1]]
#                 result = op(result, stock[i] * weights[i])
#             ROA_predicted.append(result)
#         except ZeroDivisionError:
#             ROA_predicted.append(0)
#     mse = mean_squared_error(ROA_actual, ROA_predicted)
#     return (mse,)

# def create_individual():
#     weights = [random.uniform(0, 1) for _ in range(6)]
#     weights = normalize_weights(weights)  # Normalize weights at creation
#     operators = [random.randint(0, 3) for _ in range(5)]
#     return creator.Individual(list(weights) + operators)

# def custom_crossover(ind1, ind2):
#     """Custom crossover that normalizes weights after swapping."""
#     tools.cxTwoPoint(ind1, ind2)
#     ind1[:6] = normalize_weights(ind1[:6])
#     ind2[:6] = normalize_weights(ind2[:6])
#     return ind1, ind2

# def perturbation_mutation(individual, pert_rate=0.1, indpb=0.2):
#     """Perturbation mutation with normalization."""
#     for i in range(6):  # Perturb weights
#         if random.random() < indpb:
#             perturbation = random.uniform(-pert_rate, pert_rate)
#             individual[i] = max(0, individual[i] * (1 + perturbation))
#     individual[:6] = normalize_weights(individual[:6])  # Normalize after perturbation
#     for i in range(6, len(individual)):  # Mutate operators
#         if random.random() < indpb:
#             individual[i] = random.randint(0, 3)
#     return individual,

# toolbox = base.Toolbox()
# toolbox.register("individual", create_individual)
# toolbox.register("population", tools.initRepeat, list, toolbox.individual)
# toolbox.register("mate", custom_crossover)  # Use custom crossover
# toolbox.register("mutate", perturbation_mutation, pert_rate=0.1, indpb=0.2)
# toolbox.register("select", tools.selTournament, tournsize=3)
# toolbox.register("evaluate", evaluate)

# # Optimization settings
# POP_SIZE = 200
# MAX_EVALUATIONS = 40000
# CXPB, MUTPB = 0.7, 0.2
# NUM_RUNS = 10

# def eaSimpleWithEvalCount(population, toolbox, cxpb, mutpb, max_evals, stats=None, verbose=False):
#     logbook = tools.Logbook()
#     logbook.header = ['evals', 'gen'] + (stats.fields if stats else [])
    
#     fitnesses = list(map(toolbox.evaluate, population))
#     for ind, fit in zip(population, fitnesses):
#         ind.fitness.values = fit
    
#     eval_count = len(population)
#     gen = 0
#     best_history = []
#     convergence_data = []
    
#     best = tools.selBest(population, 1)[0]
#     best_history.append((eval_count, best.fitness.values[0], best[:]))
    
#     while eval_count < max_evals:
#         gen += 1
#         offspring = toolbox.select(population, len(population))
#         offspring = list(map(toolbox.clone, offspring))
        
#         for child1, child2 in zip(offspring[::2], offspring[1::2]):
#             if random.random() < cxpb:
#                 toolbox.mate(child1, child2)
#                 del child1.fitness.values
#                 del child2.fitness.values
        
#         for mutant in offspring:
#             if random.random() < mutpb:
#                 toolbox.mutate(mutant)
#                 del mutant.fitness.values
        
#         invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
#         fitnesses = map(toolbox.evaluate, invalid_ind)
#         for ind, fit in zip(invalid_ind, fitnesses):
#             ind.fitness.values = fit
        
#         eval_count += len(invalid_ind)
#         population[:] = offspring
        
#         if stats:
#             record = stats.compile(population)
#             logbook.record(evals=eval_count, gen=gen, **record)
#             if verbose:
#                 print(logbook.stream)
        
#         current_best = tools.selBest(population, 1)[0]
#         if current_best.fitness.values[0] < best.fitness.values[0]:
#             best = current_best
#             best_history.append((eval_count, best.fitness.values[0], best[:]))
#         convergence_data.append((eval_count, best.fitness.values[0]))
    
#     return population, logbook, best_history, convergence_data

# # Track best overall solution
# best_overall = None
# best_fitness = float('inf')
# best_run_history = None
# best_convergence = None

# for run in range(NUM_RUNS):
#     random.seed(42 + run)
#     np.random.seed(42 + run)
    
#     pop = toolbox.population(n=POP_SIZE)
#     stats = tools.Statistics(lambda ind: ind.fitness.values)
#     stats.register("avg", np.mean)
#     stats.register("min", np.min)
    
#     pop, logbook, history, convergence = eaSimpleWithEvalCount(
#         pop, toolbox, cxpb=CXPB, mutpb=MUTPB, max_evals=MAX_EVALUATIONS, 
#         stats=stats, verbose=False
#     )
    
#     best_ind_run = tools.selBest(pop, 1)[0]
#     print(f"Run {run+1} best fitness: {best_ind_run.fitness.values[0]:.6f}")
    
#     if best_ind_run.fitness.values[0] < best_fitness:
#         best_fitness = best_ind_run.fitness.values[0]
#         best_overall = best_ind_run.copy()
#         best_run_history = history
#         best_convergence = convergence

# # Display results
# print("\nBest Overall Solution:")
# print(f"MSE: {best_fitness:.6f}")
# print("Weights:", [f"{w:.4f}" for w in best_overall[:6]])
# print("Operators:", [operator_map[int(i)%4].__name__ if hasattr(operator_map[int(i)%4], '__name__') else "div" 
#                      for i in best_overall[6:]])
# print(f"Sum of weights: {np.sum(best_overall[:6]):.6f}")

# # Calculate predictions
# def calculate_predictions(individual, data):
#     weights = normalize_weights(individual[:6])  # Normalize weights for consistency
#     op_indices = [int(x) % 4 for x in individual[6:]]
#     predictions = []
#     for stock in data:
#         try:
#             result = stock[0] * weights[0]
#             for i in range(1, 6):
#                 op = operator_map[op_indices[i-1]]
#                 result = op(result, stock[i] * weights[i])
#             predictions.append(result)
#         except ZeroDivisionError:
#             predictions.append(0)
#     return predictions

# predictions = calculate_predictions(best_overall, ESG_scores)

# # Show model performance
# print("\nModel Performance:")
# print(f"MSE: {mean_squared_error(ROA_actual, predictions):.6f}")
# print("\nSample predictions vs actual:")
# for i in range(min(10, len(ROA_actual))):
#     print(f"Company: {banks_data['Company'][i]}, Actual: {ROA_actual[i]:.4f}, Predicted: {predictions[i]:.4f}")

# # Plot convergence
# evals, fitnesses = zip(*best_convergence)
# plt.figure(figsize=(10, 6))
# plt.plot(evals, fitnesses, 'b-', label='Best MSE')
# plt.xlabel('Function Evaluations')
# plt.ylabel('MSE')
# plt.title('Convergence of Genetic Algorithm (Best Run)')
# plt.legend()
# plt.grid(True)
# plt.show()

# Plot predicted
      

# software_data = {
#     'Company': ['MSFT', 'CRM', 'ORCL', 'ADBE', 'ACN', 'NOW', 'IBM', 'INTU', 'PLTR', 'PANW', 
#                 'S2HO34', 'APP', 'CRWD', 'FTNT', 'MSTR', 'CSU', 'SNPS', 'WDAY', 'CDNS', 'TEAM', 
#                 'ADSK', 'DSY', 'SNOW', 'FICO', 'NET', 'HUBS', 'ZS', 'ANSS', 'ZM', 'TYL'],
#     'E1 (Opportun)': [1, 1, 1, 0.5, 1, 1, 1, 0.5, 0.5, 0, 0.5, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 0, 
#                       1, 1, 1, 0, 1, 0, 1, 1, 0],
#     'E2 (Carbon)': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0.5, 0.5, 
#                     0.5, 0.5, 0, 0.5, 0.5, 0.5, 0, 0.5, 0.5, 0],
#     'S1 (Human)': [0.5, 1, 0, 1, 0.5, 1, 0, 1, 1, 1, 0.5, 0, 0.5, 0.5, 0.5, 0, 0, 0.5, 1, 0.5, 
#                    1, 0.5, 0.5, 1, 0.5, 1, 1, 1, 0.5, 0.5, 1],
#     'S2 (Privacy)': [1, 1, 1, 1, 0.5, 1, 1, 0.5, 0, 0.5, 0.5, 0, 0, 0, 1, 0, 1, 1, 1, 1, 
#                     1, 0.5, 1, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5],
#     'G1': [0.5, 0, 0, 1, 0.5, 0, 0.5, 1, 0.5, 0.5, 0.5, 0, 0.5, 0.5, 0, 0.5, 0.5, 0.5, 1, 1, 0.5, 
#            1, 1, 0.5, 0.5, 0.5, 0.5, 1, 0, 0.5, 0],
#     'ROA': [0.181, 0.0602, 0.0825, 0.184, 0.135, 0.0699, 0.0439, 
#             0.0908, 0.0729, 0.0628, 0.145, 0.269, 0.0191, 0.179, 
#             -0.0451, 0.0539, 0.161, 0.0985, 0.118, -0.0664, 
#             0.103, 0.0714, -0.137, 0.317, -0.0239, 0.00122, 
#             -0.00771, 0.0715, 0.0919, 0.0508]
# }

pharma_data = {
    'Company': ['LLY', 'JNJ', 'ABBV', 'MRK', 'TMO', 'DHR', 'AMGN', 'PFE', 'VRTX', 'BMY', 'AZN', 'NOVO B', 'NOVN', 'GILD', 'REGN', 'ZTS', 'GSK', 'CSL', 'SAN', 'MRNA', 'BIIB', 'BAYN', 'VTRS', '4568', '4502', 'GMAB', 'RPRX', 'BMRN', '4503', 'SUNPHARMA'],
    'E1 (Toxic)': [0.5, 1, 1, 1, 0.5, 0.5, 1, 1, 0.5, 0.5, 0.5, 0.5, 1, 1, 0.5, 1, 1, 0.5, 1, 0.5, 0.5, 0, 0.5, 1, 0.5, 0.5, 0, 0.5, 1, 0.5],
    'S1 (Product)': [0, 0, 0, 0, 0.5, 0, 0.5, 0, 0, 0, 0, 0.5, 0, 0, 0.5, 1, 0, 0.5, 0, 0, 0.5, 1, 0, 1, 0, 0.5, 0, 0.5, 0.5, 0],
    'S2 (Human)': [1, 1, 1, 1, 0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.5],
    'S3 (Access)': [1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0.5, 1, 1],
    'G1 (Govern)': [0.5, 0, 0.5, 0.5, 0, 0, 1, 1, 1, 1, 1, 0.5, 1, 1, 0, 0.5, 1, 1, 1, 0.5, 1, 0.5, 0.5, 1, 1, 1, 1, 1, 1, 0.5],
    'ROA': [
        0.135,  # 0.1345361552 -> 0.135
        0.0781, # 0.07809932039 -> 0.0781
        0.0314, # 0.03135519861 -> 0.0314
        0.146,  # 0.1461667207 -> 0.146
        0.0651, # 0.06509386463 -> 0.0651
        0.0503, # 0.05028242759 -> 0.0503
        0.0445, # 0.04453445704 -> 0.0445
        0.0376, # 0.03763425744 -> 0.0376
        -0.0238,# -0.02376937142 -> -0.0238
        -0.0966,# -0.09662753906 -> -0.0966
        0.0676, # 0.06762147354 -> 0.0676
        0.217,  # 0.2168078232 -> 0.217
        0.117,  # 0.1167869648 -> 0.117
        0.00814,# 0.008136282736 -> 0.00814
        0.117,  # 0.1168609671 -> 0.117
        0.175,  # 0.1746154386 -> 0.175
        0.0433, # 0.04330423961 -> 0.0433
        0.225,  # 0.2250971358 -> 0.225
        0.00684,# 0.006844553942 -> 0.00684
        -0.252, # -0.2518031396 -> -0.252
        0.0582, # 0.05819040047 -> 0.0582
        -0.0230,# -0.02302210194 -> -0.0230
        -0.0153,# -0.0152815963 -> -0.0153
        0.0662, # 0.06619244843 -> 0.0662
        0.0138, # 0.01377112081 -> 0.0138
        0.171,  # 0.1712252516 -> 0.171
        0.0471, # 0.04713803624 -> 0.0471
        0.0611, # 0.06107635779 -> 0.0611
        0.00478,# 0.00477504081 -> 0.00478
        0.134   # 0.1337860367 -> 0.134
    ]
}

    
    
ESG_scores = np.array([list(v) for v in zip(*[pharma_data[k] for k in list(pharma_data.keys())[1:-1]])])
ROA_actual = np.array(pharma_data['ROA'])

operator_map = {0: operator.add, 1: operator.sub, 2: operator.mul, 3: lambda x, y: x / (y + 1e-6)}

# Clear any existing DEAP classes
if 'FitnessMin' in creator.__dict__:
    del creator.FitnessMin
if 'Individual' in creator.__dict__:
    del creator.Individual

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

def normalize_weights(weights):
    """Normalize weights to sum to 1, ensuring non-negative values."""
    weights = np.maximum(weights, 0)
    weight_sum = np.sum(weights)
    if weight_sum == 0:
        return np.ones_like(weights) / len(weights)
    return weights / weight_sum

def evaluate(individual):
    weights = normalize_weights(individual[:5])  # 5 weights for 5 ESG factors
    op_indices = [int(x) % 4 for x in individual[5:]]  # 5 operators
    ROA_predicted = []
    for stock in ESG_scores:
        try:
            result = stock[0] * weights[0]
            for i in range(1, 5):  # 5 factors, 4 operations
                op = operator_map[op_indices[i-1]]
                result = op(result, stock[i] * weights[i])
            ROA_predicted.append(result)
        except ZeroDivisionError:
            ROA_predicted.append(0)
    mse = mean_squared_error(ROA_actual, ROA_predicted)
    return (mse,)

def create_individual():
    weights = [random.uniform(0, 1) for _ in range(5)]  # 5 weights
    weights = normalize_weights(weights)
    operators = [random.randint(0, 3) for _ in range(5)]  # 5 operators
    return creator.Individual(list(weights) + operators)

def custom_crossover(ind1, ind2):
    """Custom crossover that normalizes weights after swapping."""
    tools.cxTwoPoint(ind1, ind2)
    ind1[:5] = normalize_weights(ind1[:5])
    ind2[:5] = normalize_weights(ind2[:5])
    return ind1, ind2

def perturbation_mutation(individual, pert_rate=0.1, indpb=0.2):
    """Perturbation mutation with normalization."""
    for i in range(5):  # Perturb 5 weights
        if random.random() < indpb:
            perturbation = random.uniform(-pert_rate, pert_rate)
            individual[i] = max(0, individual[i] * (1 + perturbation))
    individual[:5] = normalize_weights(individual[:5])  # Normalize after perturbation
    for i in range(5, len(individual)):  # Mutate 5 operators
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
print("Weights:", [f"{w:.4f}" for w in best_overall[:5]])
print("Operators:", [operator_map[int(i)%4].__name__ if hasattr(operator_map[int(i)%4], '__name__') else "div" 
                     for i in best_overall[5:]])
print(f"Sum of weights: {np.sum(best_overall[:5]):.6f}")

# Calculate predictions
def calculate_predictions(individual, data):
    weights = normalize_weights(individual[:5])
    op_indices = [int(x) % 4 for x in individual[5:]]
    predictions = []
    for stock in data:
        try:
            result = stock[0] * weights[0]
            for i in range(1, 5):
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
    print(f"Company: {pharma_data['Company'][i]}, Actual: {ROA_actual[i]:.4f}, Predicted: {predictions[i]:.4f}")

# Plot convergence
evals, fitnesses = zip(*best_convergence)
plt.figure(figsize=(10, 6))
plt.plot(evals, fitnesses, 'b-', label='Best MSE')
plt.xlabel('Function Evaluations')
plt.ylabel('MSE')
plt.title('Convergence of Genetic Algorithm (Best Run)')
plt.legend()
plt.grid(True)
plt.show()

# Plot predicted vs actual
plt.figure(figsize=(10, 6))
plt.scatter(ROA_actual, predictions, c='blue', label='Predicted vs Actual')
plt.plot([min(ROA_actual), max(ROA_actual)], [min(ROA_actual), max(ROA_actual)], 'r--', label='Perfect Fit')
plt.xlabel('Actual ROA')
plt.ylabel('Predicted ROA')
plt.title('Predicted vs Actual ROA')
plt.legend()
plt.grid(True)
plt.show()
    
    
    
####################################################################################################################   
    

    
# ESG_scores = np.array([list(v) for v in zip(*[pharma_data[k] for k in list(pharma_data.keys())[1:-1]])])
# ROA_actual = np.array(pharma_data['ROA'])

# operator_map = {0: operator.add, 1: operator.sub, 2: operator.mul, 3: lambda x, y: x / (y + 1e-6)}

# # Clear any existing DEAP classes
# if 'FitnessMin' in creator.__dict__:
#     del creator.FitnessMin
# if 'Individual' in creator.__dict__:
#     del creator.Individual

# creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
# creator.create("Individual", list, fitness=creator.FitnessMin)

# def evaluate(individual):
#     weights = individual[:5]  # Now 5 weights instead of 6
#     op_indices = [int(x) % 4 for x in individual[5:]]  # Now 4 operators
#     ROA_predicted = []
#     for stock in ESG_scores:
#         try:
#             result = stock[0] * weights[0]
#             for i in range(1, 5):  # Loop up to 5 factors
#                 op = operator_map[op_indices[i-1]]
#                 result = op(result, stock[i] * weights[i])
#             ROA_predicted.append(result)
#         except ZeroDivisionError:
#             ROA_predicted.append(0)
#     mse = mean_squared_error(ROA_actual, ROA_predicted)
#     return (mse,)

# def create_individual():
#     weights = [random.uniform(0, 1) for _ in range(5)]  # 5 weights
#     operators = [random.randint(0, 3) for _ in range(4)]  # 4 operators
#     return creator.Individual(weights + operators)

# # Perturbation mutation with non-negative constraint
# def perturbation_mutation(individual, pert_rate=0.1, indpb=0.2):
#     for i in range(5):  # Perturb 5 weights
#         if random.random() < indpb:
#             perturbation = random.uniform(-pert_rate, pert_rate)
#             individual[i] = max(0, individual[i] * (1 + perturbation))
#     for i in range(5, len(individual)):  # Mutate operators
#         if random.random() < indpb:
#             individual[i] = random.randint(0, 3)
#     return individual,

# toolbox = base.Toolbox()
# toolbox.register("individual", create_individual)
# toolbox.register("population", tools.initRepeat, list, toolbox.individual)
# toolbox.register("mate", tools.cxTwoPoint)
# toolbox.register("mutate", perturbation_mutation, pert_rate=0.1, indpb=0.2)
# toolbox.register("select", tools.selTournament, tournsize=3)
# toolbox.register("evaluate", evaluate)

# # Optimization settings
# POP_SIZE = 200
# MAX_EVALUATIONS = 40000
# CXPB, MUTPB = 0.7, 0.2
# NUM_RUNS = 10

# def eaSimpleWithEvalCount(population, toolbox, cxpb, mutpb, max_evals, stats=None, verbose=False):
#     logbook = tools.Logbook()
#     logbook.header = ['evals', 'gen'] + (stats.fields if stats else [])
    
#     fitnesses = list(map(toolbox.evaluate, population))
#     for ind, fit in zip(population, fitnesses):
#         ind.fitness.values = fit
    
#     eval_count = len(population)
#     gen = 0
#     best_history = []
#     convergence_data = []
    
#     best = tools.selBest(population, 1)[0]
#     best_history.append((eval_count, best.fitness.values[0], best[:]))
    
#     while eval_count < max_evals:
#         gen += 1
#         offspring = toolbox.select(population, len(population))
#         offspring = list(map(toolbox.clone, offspring))
        
#         for child1, child2 in zip(offspring[::2], offspring[1::2]):
#             if random.random() < cxpb:
#                 toolbox.mate(child1, child2)
#                 del child1.fitness.values
#                 del child2.fitness.values
        
#         for mutant in offspring:
#             if random.random() < mutpb:
#                 toolbox.mutate(mutant)
#                 del mutant.fitness.values
        
#         invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
#         fitnesses = map(toolbox.evaluate, invalid_ind)
#         for ind, fit in zip(invalid_ind, fitnesses):
#             ind.fitness.values = fit
        
#         eval_count += len(invalid_ind)
#         population[:] = offspring
        
#         if stats:
#             record = stats.compile(population)
#             logbook.record(evals=eval_count, gen=gen, **record)
#             if verbose:
#                 print(logbook.stream)
        
#         current_best = tools.selBest(population, 1)[0]
#         if current_best.fitness.values[0] < best.fitness.values[0]:
#             best = current_best
#             best_history.append((eval_count, best.fitness.values[0], best[:]))
#         convergence_data.append((eval_count, best.fitness.values[0]))
    
#     return population, logbook, best_history, convergence_data

# # Track best overall solution
# best_overall = None
# best_fitness = float('inf')
# best_run_history = None
# best_convergence = None

# for run in range(NUM_RUNS):
#     random.seed(42 + run)
#     np.random.seed(42 + run)
    
#     pop = toolbox.population(n=POP_SIZE)
#     stats = tools.Statistics(lambda ind: ind.fitness.values)
#     stats.register("avg", np.mean)
#     stats.register("min", np.min)
    
#     pop, logbook, history, convergence = eaSimpleWithEvalCount(
#         pop, toolbox, cxpb=CXPB, mutpb=MUTPB, max_evals=MAX_EVALUATIONS, 
#         stats=stats, verbose=False
#     )
    
#     best_ind_run = tools.selBest(pop, 1)[0]
#     print(f"Run {run+1} best fitness: {best_ind_run.fitness.values[0]}")
    
#     if best_ind_run.fitness.values[0] < best_fitness:
#         best_fitness = best_ind_run.fitness.values[0]
#         best_overall = best_ind_run.copy()
#         best_run_history = history
#         best_convergence = convergence

# # Display results
# print("\nBest Overall Solution:")
# print("MSE:", best_fitness)
# print("Weights:", [round(w, 4) for w in best_overall[:5]])  # 5 weights
# print("Operators:", [operator_map[int(i)%4].__name__ if hasattr(operator_map[int(i)%4], '__name__') else "div" 
#                      for i in best_overall[5:]])

# # Calculate predictions
# def calculate_predictions(individual, data):
#     weights = individual[:5]  # 5 weights
#     op_indices = [int(x) % 4 for x in individual[5:]]
#     predictions = []
#     for stock in data:
#         try:
#             result = stock[0] * weights[0]
#             for i in range(1, 5):  # 5 factors
#                 op = operator_map[op_indices[i-1]]
#                 result = op(result, stock[i] * weights[i])
#             predictions.append(result)
#         except ZeroDivisionError:
#             predictions.append(0)
#     return predictions

# predictions = calculate_predictions(best_overall, ESG_scores)

# # Show model performance
# print("\nModel Performance:")
# print("MSE:", mean_squared_error(ROA_actual, predictions))
# print("\nSample predictions vs actual:")
# for i in range(min(10, len(ROA_actual))):
#     print(f"Company: {software_data['Company'][i]}, Actual: {ROA_actual[i]:.4f}, Predicted: {predictions[i]:.4f}")

# # Plot convergence
# evals, fitnesses = zip(*best_convergence)
# plt.figure(figsize=(10, 6))
# plt.plot(evals, fitnesses, 'b-', label='Best MSE')
# plt.xlabel('Function Evaluations')
# plt.ylabel('MSE')
# plt.title('Convergence of Genetic Algorithm (Best Run)')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Plot predicted vs actual
# plt.figure(figsize=(10, 6))
# plt.scatter(ROA_actual, predictions, c='blue', label='Predicted vs Actual')
# plt.plot([min(ROA_actual), max(ROA_actual)], [min(ROA_actual), max(ROA_actual)], 'r--', label='Perfect Fit')
# plt.xlabel('Actual ROA')
# plt.ylabel('Predicted ROA')
# plt.title('Predicted vs Actual ROA')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Display best solution history
# print("\nBest Solution History (Evaluations, Fitness, Weights):")
# for evals, fit, ind in best_run_history[:5]:
#     print(f"Evals: {evals}, Fitness: {fit:.6f}, Weights: {[round(w, 4) for w in ind[:5]]}")