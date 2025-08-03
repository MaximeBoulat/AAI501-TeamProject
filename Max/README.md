# Readme

A modular pathfinding system that demonstrates different AI approaches for 2D grid navigation.

## Architecture

The entry point is script.py. This is the file that should be run by the Python interpreter. Some libraries are needed:
- PyGame
- scikit-learn
- pandas
- xgboost

The system has pluggable components:

**Renderer Options:**
- `ConsoleRenderer` - Text output in terminal
- `PygameRenderer` - GUI visualization window

**Logic Options:**
- `AStarLogic` - Optimal pathfinding algorithm
- `RandomForestLogic` - Random Forest classifier
- `XGBoostLogic` - XGBoost classifier  
- `NeuralNetworkLogic` - Neural network classifier

Components are injected into `Simulator(logic, renderer)`.

## Usage

**Generate training data for ML models:**
```python
logic = AStarLogic()
simulator = Simulator(logic, renderer)
simulator.run_multiple(num_runs=3000, collect_data=True, slow=False)
```

**Use ML models (after generating training data):**
```python
logic = RandomForestLogic()  # Trains automatically on training_data.csv
simulator = Simulator(logic, renderer)
simulator.run_multiple(num_runs=100, collect_data=False, slow=True)
``` 