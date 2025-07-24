from world import World
from agent import Agent
from logic import AStarLogic
from logic import RandomForestLogic
from logic import XGBoostLogic
from logic import NeuralNetworkLogic

from renderer import PygameRenderer
from renderer import ConsoleRenderer
from simulator import Simulator


def main():

    # Step 1: Pick a renderer

    # renderer = ConsoleRenderer()
    renderer = PygameRenderer(cell_size=30, show_path_history=True, window_title="Robot Navigation")


    # Step 2: Generate training data

    logic = AStarLogic()
    simulator = Simulator(logic, renderer)
    simulator.generate_training_data(num_runs=100)

    # Step 3: Train a model and run it

    # logic = RandomForestLogic()
    # logic = XGBoostLogic()           
    # logic = NeuralNetworkLogic()     
    # logic = NeuralNetworkLogic(hidden_layer_sizes=(200, 100, 50))  # Larger network

    # simulator = Simulator(logic, renderer)
    # simulator.run_simulation()

    
    return 0

if __name__ == "__main__":
    exit(main()) 