from agent import Agent
from world import World
from renderer import PygameRenderer, Renderer
from config import *

from typing import Dict, Any, List
import time

from models import BaseModel, NeuralNetworkModel

    
def run_simulation(num_runs: int, model: BaseModel, renderer: Renderer, slow: bool = True) -> List[Dict[str, Any]]:
       
    for run_id in range(num_runs):
        print(f"\n=== Run {run_id + 1}/{num_runs} ===")
            
        # Generate new world
        try:
            world = World.from_random(WORLD_SIZE, OBSTACLE_PROB, WALL_COUNT, WALL_MAX_LEN, MIN_START_GOAL_DISTANCE)
            agent = Agent(world, model)
        except RuntimeError as e:
                print(f"Failed to generate world for run {run_id}: {e}")
                continue
        
            # Reset components
        agent.reset()
        
        timestamp = 0
        step = 0
        
        
        try:
            # Single game loop that ticks every second
            while True:
                # Check if already at goal before trying to move
                if agent.is_at_goal():
                    break

                # Check if already at goal before trying to move
                if agent.is_at_goal():
                    break

                # Update agent (get action from logic and move)
                update_result = agent.update()
                
                # Check if map is unsolvable
                if not update_result['success']:
                    print(f"Simulation failed at step {step}: {update_result['reason']}")
                    break
                
                # Update results
                step += 1
                
                
                # Render current state
                renderer.render_world(world, agent)
                renderer.render_step_info(step,     update_result['action'], True)
                
                # Wait for next tick (1 second)
                if slow:
                    time.sleep(0.1)   
            

                # Final render and status

                renderer.render_world(world, agent)
            
            
        except KeyboardInterrupt:
            print("\nSimulation interrupted by user.")



model = NeuralNetworkModel()

model.from_file("NeuralNetwork")

renderer = PygameRenderer(cell_size=30, show_path_history=True, window_title="Robot Navigation")

run_simulation(10, model, renderer, True)
