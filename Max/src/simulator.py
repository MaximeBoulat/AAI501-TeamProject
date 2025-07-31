import csv
import os
import time
from typing import List, Dict, Optional, Any
from world import World
from agent import Agent
from logic import Logic, AStarLogic
from logic import RandomForestLogic
from renderer import Renderer, ConsoleRenderer, PygameRenderer

class Simulator:
    """Orchestrates simulation of agent navigation with different logic strategies."""
    
    def __init__(self, logic: Logic, renderer: Renderer):

        world = World.from_random(size=30, obstacle_prob=0.1)

        agent = Agent(world, logic)

        self.world = world
        self.agent = agent
        self.renderer = renderer
        self.training_data = []
        
    def run_simulation(self, collect_data: bool = False, run_id: int = 0, slow: bool = True) -> Dict[str, Any]:

        # Reset components
        self.agent.reset()
        
        # Initialize results
        results = {
            'success': False,
            'steps_taken': 0,
            'final_distance': self.agent.get_distance_to_goal(),
            'path': [self.agent.position],
            'training_data': []
        }
        
        timestamp = 0
        step = 0
        
        
        try:
            # Single game loop that ticks every second
            while True:

                # Check if already at goal before trying to move
                if self.agent.is_at_goal():
                    results['success'] = True
                    print(f"\nGoal reached in {step} steps!")
                    break

                # Update agent (get action from logic and move)
                update_result = self.agent.update(collect_data, timestamp, run_id)
                
                # Check if map is unsolvable
                if not update_result['success']:
                    print(f"Simulation failed at step {step}: {update_result['reason']}")
                    break
                
                # Update results
                step += 1
                results['steps_taken'] = step
                results['path'].append(self.agent.position)
                results['final_distance'] = self.agent.get_distance_to_goal()
                
                # Collect training data
                if update_result['training_data']:
                    results['training_data'].append(update_result['training_data'])
                    timestamp += 1
                
                # Render current state
                self.renderer.render_world(self.world, self.agent)
                self.renderer.render_step_info(step, update_result['action'], True)
                
                # Wait for next tick (1 second)
                if slow:
                    time.sleep(1.0)   
            
            # Check final state
            results['success'] = self.agent.is_at_goal()
            
            # Final render and status

            self.renderer.render_world(self.world, self.agent)
            
            if results['success']:
                print(f"\nGoal reached in {step} steps!")
            else:
                print(f"\nSimulation failed.")
            
        except KeyboardInterrupt:
            print("\nSimulation interrupted by user.")
            results['success'] = False
        
        return results
    
    def generate_training_data(self, num_runs: int) -> List[Dict[str, Any]]:
        """
        Run multiple simulations with different random worlds.
        
        Args:
            num_runs: Number of simulations to run
            **kwargs: Arguments passed to run_simulation
            
        Returns:
            List of simulation results
        """
        all_results = []
        all_training_data = []
        
        for run_id in range(num_runs):
            print(f"\n=== Run {run_id + 1}/{num_runs} ===")
            
            # Generate new world
            try:
                world = World.from_random(size=30, obstacle_prob=0.1)
                self.world = world
                self.agent.world = world
            except RuntimeError as e:
                print(f"Failed to generate world for run {run_id}: {e}")
                continue
            
            # Run simulation
            results = self.run_simulation(run_id=run_id, collect_data=True, slow=False)
            all_results.append(results)
            
            # Collect training data
            all_training_data.extend(results.get('training_data', []))
            
            # Print results
            status = "SUCCESS" if results['success'] else "FAILED"
            print(f"Run {run_id}: {status} - {results['steps_taken']} steps, "
                  f"final distance: {results['final_distance']:.2f}")
        
        # Save training data if any was collected
        if all_training_data:
            self.save_training_data(all_training_data)
        
        return all_results
    
    def save_training_data(self, training_data: List[Dict], filename: str = "training_data.csv"):
        """Save collected training data to CSV file."""
        if not training_data:
            return
        
        fieldnames = ['timestamp', 'run_id', 'position_x', 'position_y'] + \
                    [f'sensor_{i}' for i in range(8)] + \
                    ['action', 'distance_to_goal', 'path_length', 'goal_direction']
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for data in training_data:
                # Flatten position tuple
                row = {
                    'timestamp': data['timestamp'],
                    'run_id': data['run_id'],
                    'position_x': data['position'][0],
                    'position_y': data['position'][1],
                    'action': data['action'],
                    'distance_to_goal': data['distance_to_goal'],
                    'path_length': data['path_length'],
                    'goal_direction': data['goal_direction']
                }
                
                # Add sensor readings
                for i, sensor_val in enumerate(data['sensors']):
                    row[f'sensor_{i}'] = sensor_val
                
                writer.writerow(row)
        
        print(f"Training data saved to {filename} ({len(training_data)} records)")
    
    def save_world(self, run_id: int, output_dir: str = "worlds"):
        """Save current world to disk."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as numpy array
        world_file = os.path.join(output_dir, f"world_{run_id}.npy")
        import numpy as np
        np.save(world_file, self.world.grid)
        
        # Save as text
        text_file = os.path.join(output_dir, f"world_{run_id}.txt")
        with open(text_file, 'w') as f:
            for row in self.world.grid:
                f.write(" ".join(str(cell) for cell in row) + "\n")

    
    