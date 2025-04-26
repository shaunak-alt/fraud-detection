import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from deap import base, creator, tools, algorithms

# Load your dataset (replace 'fraud_data.csv' with your actual file)
try:
    data = pd.read_csv(r"C:\Users\shaun\Downloads\fraud.csv")
except FileNotFoundError:
    print("Error: 'fraud_balanced.csv' not found. Please provide the correct file path.")
    exit()

# Preprocessing
# 1. Feature Engineering
# Handle potential negative or zero values in transaction_amount
data['transaction_amount_log'] = np.log1p(data['transaction_amount'].clip(lower=1e-5))

# 2. Handle missing values
data.fillna(0, inplace=True)  # Example: Filling with 0s, adjust as needed

# 3. Select features (X) and target (y)
features = ['transaction_amount', 'transaction_amount_log', 'customer_age', 'payment_method']
target = 'is_fraud'

X = data[features]
y = data[target]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Genetic Algorithm for Feature Selection
# Define the fitness function
def evaluate_model(individual):
    selected_features = [features[i] for i, val in enumerate(individual) if val == 1]
    if len(selected_features) == 0:
        return 0.0,  # Handle the case where no features are selected

    X_train_subset = X_train[:, [i for i, val in enumerate(individual) if val == 1]]
    X_test_subset = X_test[:, [i for i, val in enumerate(individual) if val == 1]]

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_subset, y_train)
    y_pred = model.predict(X_test_subset)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy,

# Set up DEAP for Genetic Algorithm
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximize accuracy
creator.create("Individual", list, fitness=creator.FitnessMax)

# Define the toolbox
toolbox = base.Toolbox()
toolbox.register("attr_bool", lambda: np.random.randint(0, 2))  # Binary features (0 or 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(features))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register genetic operators
toolbox.register("mate", tools.cxTwoPoint)  # Two-point crossover
toolbox.register("mutate", tools.mutFlipBit, indpb=0.2)  # Flip bits with probability 0.2
toolbox.register("select", tools.selTournament, tournsize=3)  # Tournament selection
toolbox.register("evaluate", evaluate_model)  # Fitness evaluation

# Create the population
population = toolbox.population(n=50)  # Population of 50 individuals

# Run the Genetic Algorithm
num_generations = 10
for generation in range(num_generations):
    # Evaluate all individuals
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Select, mate, and mutate
    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if np.random.rand() < 0.5:  # Crossover probability
            toolbox.mate(child1, child2)
            del child1.fitness.values, child2.fitness.values

    for mutant in offspring:
        if np.random.rand() < 0.2:  # Mutation probability
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Replace population with offspring
    population[:] = offspring

# Get the best individual
best_individual = tools.selBest(population, k=1)[0]
selected_features = [features[i] for i, val in enumerate(best_individual) if val == 1]

print(f"Selected features: {selected_features}")

# Train a model with the selected features
if selected_features:
    X_train_selected = X_train[:, [i for i, val in enumerate(best_individual) if val == 1]]
    X_test_selected = X_test[:, [i for i, val in enumerate(best_individual) if val == 1]]

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_test_selected)

    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
else:
    print("No features were selected by the Genetic Algorithm!")

# Example prediction for a new transaction
new_transaction = pd.DataFrame([[100, np.log1p(100), 30, 2]], columns=features)
new_transaction_scaled = scaler.transform(new_transaction)
new_transaction_selected = new_transaction_scaled[:, [i for i, val in enumerate(best_individual) if val == 1]]
prediction = model.predict(new_transaction_selected)
print(f"Fraud prediction for new transaction: {prediction[0]}")