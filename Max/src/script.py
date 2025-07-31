from world import World
from agent import Agent

from logic import AStarLogic
from logic import LogisticRegressionLogic
from logic import SVMLogic
from logic import NaiveBayesLogic
from logic import KNNLogic
from logic import RandomForestLogic
from logic import XGBoostLogic
from logic import NeuralNetworkLogic


from renderer import PygameRenderer
from renderer import ConsoleRenderer
from simulator import Simulator


def main():

    renderer = PygameRenderer(cell_size=30, show_path_history=True, window_title="Robot Navigation")

    # logic = RandomForestLogic()
    # logic = XGBoostLogic()   
    # logic = LogisticRegressionLogic()
    # logic = SVMLogic()
    # logic = NaiveBayesLogic()
    # logic = KNNLogic()
    logic = NeuralNetworkLogic()  
    # logic = NeuralNetworkLogic(hidden_layer_sizes=(200, 100, 50))  # Larger network

    # logic = AStarLogic()

    simulator = Simulator(logic, renderer)

    # simulator.generate_training_data(num_runs=3000)
    simulator.run_simulation()
    

    
    return 0

if __name__ == "__main__":
    exit(main()) 