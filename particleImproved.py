import random
import numpy as np
import operator
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

# # PSO Parameters
# POP_SIZE = 200
# MAX_EVALUATIONS = 40000
# W = 0.8  # Inertia weight
# C1 = 0.5  # Cognitive coefficient (personal best)
# C2 = 0.5  # Social coefficient (global best)
# NUM_RUNS = 10

# # PSO Implementation
# class PSO:
#     def __init__(self, dim, bounds_weights, bounds_operators):
#         self.dim = dim  # 11 dimensions (6 weights + 5 operators)
#         self.bounds_weights = bounds_weights  # [0, 1] for weights (will be normalized)
#         self.bounds_operators = bounds_operators  # [0, 3] for operators
#         self.pop_size = POP_SIZE
#         self.max_evals = MAX_EVALUATIONS
#         self.w = W
#         self.c1 = C1
#         self.c2 = C2
        
#         # Initialize particles, velocities, personal bests
#         self.X = np.zeros((self.pop_size, self.dim))
#         self.V = np.zeros((self.pop_size, self.dim))
#         self.pbest_X = np.zeros((self.pop_size, self.dim))
#         self.pbest_fitness = np.full(self.pop_size, float('inf'))
#         self.gbest_X = np.zeros(self.dim)
#         self.gbest_fitness = float('inf')
        
#         # Tracking
#         self.eval_count = 0
#         self.convergence_data = []
#         self.best_history = []

#     def normalize_weights(self, weights):
#         """Normalize weights to sum to 1, ensuring non-negative values."""
#         weights = np.maximum(weights, 0)  # Ensure non-negative
#         weight_sum = np.sum(weights)
#         if weight_sum == 0:  # Avoid division by zero
#             return np.ones_like(weights) / len(weights)  # Default to uniform if all zero
#         return weights / weight_sum

#     def initialize(self):
#         # Initialize weights [0, 1] and operators [0, 3]
#         for i in range(self.pop_size):
#             weights = np.random.uniform(self.bounds_weights[0], self.bounds_weights[1], 6)
#             self.X[i, :6] = self.normalize_weights(weights)
#             self.X[i, 6:] = np.random.uniform(self.bounds_operators[0], self.bounds_operators[1], 5)
#             self.V[i, :6] = np.random.uniform(-0.1, 0.1, 6)  # Small initial velocity for weights
#             self.V[i, 6:] = np.random.uniform(-1, 1, 5)  # Velocity for operators
        
#         # Evaluate initial population
#         for i in range(self.pop_size):
#             fitness = self.evaluate(self.X[i])
#             self.pbest_X[i] = self.X[i].copy()
#             self.pbest_fitness[i] = fitness
#             if fitness < self.gbest_fitness:
#                 self.gbest_fitness = fitness
#                 self.gbest_X = self.X[i].copy()
#             self.eval_count += 1
#             self.convergence_data.append((self.eval_count, self.gbest_fitness))
#             if i == 0 or fitness < min([h[1] for h in self.best_history]):
#                 self.best_history.append((self.eval_count, fitness, self.X[i].copy()))

#     def evaluate(self, particle):
#         weights = particle[:6]
#         op_indices = [int(x) % 4 for x in particle[6:]]
#         ROA_predicted = []
#         for stock in ESG_scores:
#             try:
#                 result = stock[0] * weights[0]
#                 for i in range(1, 6):
#                     op = operator_map[op_indices[i-1]]
#                     result = op(result, stock[i] * weights[i])
#                 ROA_predicted.append(result)
#             except ZeroDivisionError:
#                 ROA_predicted.append(0)
#         return mean_squared_error(ROA_actual, ROA_predicted)

#     def run(self):
#         self.initialize()
        
#         while self.eval_count < self.max_evals:
#             for i in range(self.pop_size):
#                 # Update velocity
#                 r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
#                 self.V[i] = (self.w * self.V[i] + 
#                             self.c1 * r1 * (self.pbest_X[i] - self.X[i]) + 
#                             self.c2 * r2 * (self.gbest_X - self.X[i]))
                
#                 # Update position
#                 self.X[i] += self.V[i]
                
#                 # Normalize weights to sum to 1 and clamp operators
#                 self.X[i, :6] = self.normalize_weights(self.X[i, :6])
#                 self.X[i, 6:] = np.clip(self.X[i, 6:], self.bounds_operators[0], self.bounds_operators[1])
                
#                 # Evaluate fitness
#                 fitness = self.evaluate(self.X[i])
#                 self.eval_count += 1
                
#                 # Update personal best
#                 if fitness < self.pbest_fitness[i]:
#                     self.pbest_fitness[i] = fitness
#                     self.pbest_X[i] = self.X[i].copy()
                
#                 # Update global best
#                 if fitness < self.gbest_fitness:
#                     self.gbest_fitness = fitness
#                     self.gbest_X = self.X[i].copy()
#                     self.best_history.append((self.eval_count, fitness, self.X[i].copy()))
                
#                 self.convergence_data.append((self.eval_count, self.gbest_fitness))
                
#                 if self.eval_count >= self.max_evals:
#                     break
        
#         return self.gbest_X, self.gbest_fitness, self.convergence_data, self.best_history

# # Run PSO multiple times
# best_overall = None
# best_fitness = float('inf')
# best_convergence = None
# best_run_history = None

# for run in range(NUM_RUNS):
#     random.seed(42 + run)
#     np.random.seed(42 + run)
    
#     pso = PSO(dim=11, bounds_weights=[0, 1], bounds_operators=[0, 3])
#     gbest_X, gbest_fitness, convergence, history = pso.run()
    
#     print(f"Run {run+1} best fitness: {gbest_fitness:.6f}")
    
#     if gbest_fitness < best_fitness:
#         best_fitness = gbest_fitness
#         best_overall = gbest_X.copy()
#         best_convergence = convergence
#         best_run_history = history

# # Display results
# print("\nBest Overall Solution:")
# print(f"MSE: {best_fitness:.6f}")
# print("Weights:", [f"{w:.4f}" for w in best_overall[:6]])
# print("Operators:", [operator_map[int(i)%4].__name__ if hasattr(operator_map[int(i)%4], '__name__') else "div" 
#                      for i in best_overall[6:]])
# print(f"Sum of weights: {np.sum(best_overall[:6]):.6f}")  # Verify sum is 1

# # Calculate predictions
# def calculate_predictions(particle, data):
#     weights = particle[:6]
#     op_indices = [int(x) % 4 for x in particle[6:]]
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
# plt.title('Convergence of PSO (Best Run)')
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
#     print(f"Evals: {evals}, Fitness: {fit:.6f}, Weights: {[f'{w:.4f}' for w in ind[:6]]}")
    
    
################################################################################################################################

software_data = {
    'Company': ['MSFT', 'CRM', 'ORCL', 'ADBE', 'ACN', 'NOW', 'IBM', 'INTU', 'PLTR', 'PANW', 
                'S2HO34', 'APP', 'CRWD', 'FTNT', 'MSTR', 'CSU', 'SNPS', 'WDAY', 'CDNS', 'TEAM', 
                'ADSK', 'DSY', 'SNOW', 'FICO', 'NET', 'HUBS', 'ZS', 'ANSS', 'ZM', 'TYL'],
    'E1 (Opportun)': [1, 1, 1, 0.5, 1, 1, 1, 0.5, 0.5, 0, 0.5, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 0, 
                      1, 1, 1, 0, 1, 0, 1, 1, 0],
    'E2 (Carbon)': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0.5, 0.5, 
                    0.5, 0.5, 0, 0.5, 0.5, 0.5, 0, 0.5, 0.5, 0],
    'S1 (Human)': [0.5, 1, 0, 1, 0.5, 1, 0, 1, 1, 1, 0.5, 0, 0.5, 0.5, 0.5, 0, 0, 0.5, 1, 0.5, 
                   1, 0.5, 0.5, 1, 0.5, 1, 1, 1, 0.5, 0.5],
    'S2 (Privacy)': [1, 1, 1, 1, 0.5, 1, 1, 0.5, 0, 0.5, 0.5, 0, 0, 0, 1, 0, 1, 1, 1, 1, 
                     1, 0.5, 1, 1, 1, 1, 1, 1, 0.5, 0.5],
    'G1': [0.5, 0, 0, 1, 0.5, 0, 0.5, 1, 0.5, 0.5, 0.5, 0, 0.5, 0.5, 0, 0.5, 0.5, 0.5, 1, 1, 0.5, 
           1, 1, 0.5, 0.5, 0.5, 0.5, 1, 0, 0.5],
    'ROA': [0.181, 0.0602, 0.0825, 0.184, 0.135, 0.0699, 0.0439, 
            0.0908, 0.0729, 0.0628, 0.145, 0.269, 0.0191, 0.179, 
            -0.0451, 0.0539, 0.161, 0.0985, 0.118, -0.0664, 
            0.103, 0.0714, -0.137, 0.317, -0.0239, 0.00122, 
            -0.00771, 0.0715, 0.0919, 0.0508]
}

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



# ESG_scores = np.array([list(v) for v in zip(*[pharma_data[k] for k in list(pharma_data.keys())[1:-1]])])
# ROA_actual = np.array(pharma_data['ROA'])

# operator_map = {0: operator.add, 1: operator.sub, 2: operator.mul, 3: lambda x, y: x / (y + 1e-6)}

# # PSO Parameters
# POP_SIZE = 200
# MAX_EVALUATIONS = 40000
# NUM_RUNS = 10

# # PSO Class
# class PSO:
#     def __init__(self, func, dim, bound_weights, bound_operators, max_iters=1000, pop=50, func_eval=40000):
#         self.func = func
#         self.dim = dim  # 10 (5 weights + 5 operators for 5 ESG factors)
#         self.bound_weights = bound_weights
#         self.bound_operators = bound_operators
#         self.pop_size = pop
#         self.iterations = max_iters
#         self.fe_th = func_eval
        
#         self.fe_count = 0
#         self.history = []
#         self.gbest = {'fitness': float('inf'), 'solution': None}
#         self.convergence_iter = 0
#         self.convergence_fe = 1
        
#         self.w = 0.8
#         self.cp = 0.5
#         self.cg = 0.5
        
#         v_range_weights = bound_weights[1] - bound_weights[0]
#         v_range_operators = bound_operators[1] - bound_operators[0]
#         self.V = np.zeros((self.pop_size, self.dim))
#         self.V[:, :5] = np.random.uniform(-v_range_weights, v_range_weights, (self.pop_size, 5))
#         self.V[:, 5:] = np.random.uniform(-v_range_operators, v_range_operators, (self.pop_size, 5))

#     def normalize_weights(self, weights):
#         weights = np.maximum(weights, 0)
#         weight_sum = np.sum(weights)
#         if weight_sum == 0:
#             return np.ones_like(weights) / len(weights)
#         return weights / weight_sum

#     def run(self):
#         self.history = []
        
#         # Initialization
#         self.X = np.zeros((self.pop_size, self.dim))
#         self.X[:, :5] = np.random.uniform(self.bound_weights[0], self.bound_weights[1], (self.pop_size, 5))
#         self.X[:, :5] = np.apply_along_axis(self.normalize_weights, 1, self.X[:, :5])
#         self.X[:, 5:] = np.random.uniform(self.bound_operators[0], self.bound_operators[1], (self.pop_size, 5))
        
#         self.Y = np.array([self.func(ind) for ind in self.X])
#         self.fe_count += self.pop_size
        
#         self.pbest_X = self.X.copy()
#         self.pbest_Y = self.Y.copy()
        
#         gbest_idx = np.argmin(self.Y)
#         self.gbest['fitness'] = self.Y[gbest_idx]
#         self.gbest['solution'] = self.X[gbest_idx].copy()
#         self.history.append(self.gbest['fitness'])
        
#         iteration = 0
#         while self.fe_count < self.fe_th:
#             iteration += 1
            
#             r1 = np.random.rand(self.pop_size, self.dim)
#             r2 = np.random.rand(self.pop_size, self.dim)
#             self.V = (self.w * self.V + 
#                       self.cp * r1 * (self.pbest_X - self.X) + 
#                       self.cg * r2 * (self.gbest['solution'] - self.X))
            
#             self.X += self.V
#             self.X[:, :5] = np.apply_along_axis(self.normalize_weights, 1, self.X[:, :5])
#             self.X[:, 5:] = np.clip(self.X[:, 5:], self.bound_operators[0], self.bound_operators[1])
            
#             for idx in range(self.pop_size):
#                 y_u = self.func(self.X[idx])
#                 self.Y[idx] = y_u
#                 self.fe_count += 1
                
#                 if y_u < self.pbest_Y[idx]:
#                     self.pbest_Y[idx] = y_u
#                     self.pbest_X[idx] = self.X[idx].copy()
                
#                 if y_u < self.gbest['fitness']:
#                     self.gbest['fitness'] = y_u
#                     self.gbest['solution'] = self.X[idx].copy()
#                     self.convergence_iter = iteration
#                     self.convergence_fe = self.fe_count
            
#             self.history.append(self.gbest['fitness'])
#             if self.fe_count >= self.fe_th:
#                 break
        
#         return self.gbest['solution'], self.gbest['fitness'], self.history

# # Define fitness function for software_data (5 ESG factors)
# def evaluate_particle(particle):
#     weights = particle[:5]  # 5 weights for 5 ESG factors
#     op_indices = [int(x) % 4 for x in particle[5:]]  # 5 operators
#     ROA_predicted = []
#     for stock in ESG_scores:
#         try:
#             result = stock[0] * weights[0]
#             for i in range(1, 5):  # Only 5 factors in software_data
#                 op = operator_map[op_indices[i-1]]
#                 result = op(result, stock[i] * weights[i])
#             ROA_predicted.append(result)
#         except ZeroDivisionError:
#             ROA_predicted.append(0)
#     return mean_squared_error(ROA_actual, ROA_predicted)

# # Calculate predictions function
# def calculate_predictions(particle, data):
#     weights = particle[:5]
#     op_indices = [int(x) % 4 for x in particle[5:]]
#     predictions = []
#     for stock in data:
#         try:
#             result = stock[0] * weights[0]
#             for i in range(1, 5):
#                 op = operator_map[op_indices[i-1]]
#                 result = op(result, stock[i] * weights[i])
#             predictions.append(result)
#         except ZeroDivisionError:
#             predictions.append(0)
#     return predictions

# # Run PSO
# best_overall = None
# best_fitness = float('inf')
# best_history = None

# for run in range(NUM_RUNS):
#     random.seed(42 + run)
#     np.random.seed(42 + run)
    
#     pso = PSO(func=evaluate_particle, dim=10, bound_weights=[0, 1], bound_operators=[0, 3], 
#               pop=POP_SIZE, func_eval=MAX_EVALUATIONS)
#     solution, fitness, history = pso.run()
    
#     print(f"Run {run+1} best fitness: {fitness:.6f}")
    
#     if fitness < best_fitness:
#         best_fitness = fitness
#         best_overall = solution.copy()
#         best_history = history

# # Display results
# print("\nBest Overall Solution:")
# print(f"MSE: {best_fitness:.6f}")
# print("Weights:", [f"{w:.4f}" for w in best_overall[:5]])
# print("Operators:", [operator_map[int(i)%4].__name__ if hasattr(operator_map[int(i)%4], '__name__') else "div" 
#                      for i in best_overall[5:]])
# print(f"Sum of weights: {np.sum(best_overall[:5]):.6f}")

# # Calculate predictions
# predictions = calculate_predictions(best_overall, ESG_scores)

# # Show model performance
# print("\nModel Performance:")
# print(f"MSE: {mean_squared_error(ROA_actual, predictions):.6f}")
# print("\nSample predictions vs actual:")
# for i in range(min(10, len(ROA_actual))):
#     print(f"Company: {pharma_data['Company'][i]}, Actual: {ROA_actual[i]:.4f}, Predicted: {predictions[i]:.4f}")

# # Plot convergence
# plt.figure(figsize=(10, 6))
# plt.plot(best_history, 'b-', label='Best MSE')
# plt.xlabel('Iterations')
# plt.ylabel('MSE')
# plt.title('Convergence of PSO (Best Run)')
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


ESG_scores = np.array([list(v) for v in zip(*[pharma_data[k] for k in list(pharma_data.keys())[1:-1]])])
ROA_actual = np.array(pharma_data['ROA'])

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
    def __init__(self, dim, bounds_weights, bounds_operators):
        self.dim = dim  # 11 dimensions (6 weights + 5 operators)
        self.bounds_weights = bounds_weights  # [0, 1] for weights (will be normalized)
        self.bounds_operators = bounds_operators  # [0, 3] for operators
        self.pop_size = POP_SIZE
        self.max_evals = MAX_EVALUATIONS
        self.w = W
        self.c1 = C1
        self.c2 = C2
        
        # Initialize particles, velocities, personal bests
        self.X = np.zeros((self.pop_size, self.dim))
        self.V = np.zeros((self.pop_size, self.dim))
        self.pbest_X = np.zeros((self.pop_size, self.dim))
        self.pbest_fitness = np.full(self.pop_size, float('inf'))
        self.gbest_X = np.zeros(self.dim)
        self.gbest_fitness = float('inf')
        
        # Tracking
        self.eval_count = 0
        self.convergence_data = []
        self.best_history = []

    def normalize_weights(self, weights):
        """Normalize weights to sum to 1, ensuring non-negative values."""
        weights = np.maximum(weights, 0)  # Ensure non-negative
        weight_sum = np.sum(weights)
        if weight_sum == 0:  # Avoid division by zero
            return np.ones_like(weights) / len(weights)  # Default to uniform if all zero
        return weights / weight_sum

    def initialize(self):
        # Initialize weights [0, 1] and operators [0, 3]
        for i in range(self.pop_size):
            weights = np.random.uniform(self.bounds_weights[0], self.bounds_weights[1], 5)
            self.X[i, :5] = self.normalize_weights(weights)
            self.X[i, 5:] = np.random.uniform(self.bounds_operators[0], self.bounds_operators[1], 5)
            self.V[i, :5] = np.random.uniform(-0.1, 0.1, 5)  # Small initial velocity for weights
            self.V[i, 5:] = np.random.uniform(-1, 1, 4)  # Velocity for operators
        
        # Evaluate initial population
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
        weights = particle[:5]
        op_indices = [int(x) % 4 for x in particle[5:]]
        ROA_predicted = []
        for stock in ESG_scores:
            try:
                result = stock[0] * weights[0]
                for i in range(1, 5):
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
                # Update velocity
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                self.V[i] = (self.w * self.V[i] + 
                            self.c1 * r1 * (self.pbest_X[i] - self.X[i]) + 
                            self.c2 * r2 * (self.gbest_X - self.X[i]))
                
                # Update position
                self.X[i] += self.V[i]
                
                # Normalize weights to sum to 1 and clamp operators
                self.X[i, :5] = self.normalize_weights(self.X[i, :5])
                self.X[i, 5:] = np.clip(self.X[i, 5:], self.bounds_operators[0], self.bounds_operators[1])
                
                # Evaluate fitness
                fitness = self.evaluate(self.X[i])
                self.eval_count += 1
                
                # Update personal best
                if fitness < self.pbest_fitness[i]:
                    self.pbest_fitness[i] = fitness
                    self.pbest_X[i] = self.X[i].copy()
                
                # Update global best
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
    
    pso = PSO(dim=9, bounds_weights=[0, 1], bounds_operators=[0, 3])
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
print("Weights:", [f"{w:.4f}" for w in best_overall[:5]])
print("Operators:", [operator_map[int(i)%4].__name__ if hasattr(operator_map[int(i)%4], '__name__') else "div" 
                     for i in best_overall[5:]])
print(f"Sum of weights: {np.sum(best_overall[:5]):.6f}")  # Verify sum is 1

# Calculate predictions
def calculate_predictions(particle, data):
    weights = particle[:5]
    op_indices = [int(x) % 4 for x in particle[5:]]
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
    print(f"Company: {software_data['Company'][i]}, Actual: {ROA_actual[i]:.4f}, Predicted: {predictions[i]:.4f}")

# Plot convergence
evals, fitnesses = zip(*best_convergence)
plt.figure(figsize=(10, 6))
plt.plot(evals, fitnesses, 'b-', label='Best MSE')
plt.xlabel('Function Evaluations')
plt.ylabel('MSE')
plt.title('Convergence of PSO (Best Run)')
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

# Display best solution history
print("\nBest Solution History (Evaluations, Fitness, Weights):")
for evals, fit, ind in best_run_history[:5]:
    print(f"Evals: {evals}, Fitness: {fit:.6f}, Weights: {[f'{w:.4f}' for w in ind[:5]]}")
    