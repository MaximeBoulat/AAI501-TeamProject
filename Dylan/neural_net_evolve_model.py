import random
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import set_random_seed
import tensorflow as tf

# Load and preprocess data
df = pd.read_csv("robot_training_data.csv")
sensor_cols = [f"sensor_{i}" for i in range(8)]
X = df[sensor_cols]
y = df["action"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reproducibility
random.seed(42)
set_random_seed(42)

# Define genetic search space
layer_range = [1, 2, 3]
units_range = [16, 32, 64, 128]
activation_options = ['relu', 'tanh', 'sigmoid']

# Individual = [num_layers, (units_1, act_1), ..., (units_n, act_n)]
def generate_individual():
    num_layers = random.choice(layer_range)
    architecture = [num_layers]
    for _ in range(num_layers):
        units = random.choice(units_range)
        act = random.choice(activation_options)
        architecture.append((units, act))
    return architecture

# Build model from genotype
def build_model(architecture):
    model = Sequential()
    for i, (units, act) in enumerate(architecture[1:]):
        if i == 0:
            model.add(Dense(units, activation=act, input_shape=(8,)))
        else:
            model.add(Dense(units, activation=act))
    model.add(Dense(8, activation='softmax'))  # 8 classes
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Fitness: use validation accuracy
def evaluate(individual):
    model = build_model(individual)
    history = model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test),
                        epochs=10, batch_size=32, verbose=0)
    val_acc = history.history['val_accuracy'][-1]
    return (val_acc,)

# Genetic Algorithm setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, generate_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", lambda ind: mutate_arch(ind))
toolbox.register("select", tools.selTournament, tournsize=3)

# Mutation function
def mutate_arch(individual):
    if len(individual) > 1:
        layer_to_mutate = random.randint(1, len(individual)-1)
        if random.random() < 0.5:
            individual[layer_to_mutate] = (
                random.choice(units_range),
                random.choice(activation_options)
            )
        else:
            # Add or remove a layer
            if len(individual) - 1 < max(layer_range):
                individual.append((random.choice(units_range), random.choice(activation_options)))
                individual[0] += 1
            elif len(individual) - 1 > min(layer_range):
                individual.pop()
                individual[0] -= 1
    return individual,

# Run evolution
def evolve(pop_size=10, generations=5):
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.3, ngen=generations,
                                   stats=stats, halloffame=hof, verbose=True)

    return hof[0], log

# Execute
best_arch, logs = evolve()

# Final model
print(f"\n🏆 Best Architecture: {best_arch}")
final_model = build_model(best_arch)
final_model.fit(X_train_scaled, y_train, epochs=30, batch_size=32, verbose=0,
                validation_data=(X_test_scaled, y_test))
final_acc = final_model.evaluate(X_test_scaled, y_test, verbose=0)[1]

# Save
os.makedirs("neural_net_outputs", exist_ok=True)
with open("neural_net_outputs/best_architecture.txt", "w") as f:
    f.write(str(best_arch))
with open("neural_net_outputs/final_accuracy.txt", "w") as f:
    f.write(f"{final_acc:.4f}")
print(f"✅ Done. Best validation accuracy: {final_acc:.4f}")

