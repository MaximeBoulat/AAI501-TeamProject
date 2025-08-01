from typing import Tuple, List
from world import World
from logic import Logic

class Agent:
    
    def __init__(self, world: World, logic: Logic):
        self.world = world
        self.position = world.start
        self.path_history = [self.position]
        self.logic = logic


    def reset(self):

        self.position = self.world.start
        self.path_history = [self.position]
        self.logic.reset()
        
    def update(self, collect_data: bool = True, timestamp: int = 0, run_id: int = 0) -> dict:

        # Get next action from logic
        action = self.logic.get_next_action(self.position, self.world)
        
        if action is None:
            return {
                'success': False,
                'action': None,
                'training_data': None,
                'reason': 'No valid action available'
            }
        
        # Collect training data before move
        training_data = None
        if collect_data:
            training_data = self.get_state_data(timestamp, run_id)
            training_data['action'] = action
        
        # Attempt to move agent
        move_success = self.move(action)
        
        return {
            'success': move_success,
            'action': action,
            'training_data': training_data,
            'reason': 'Invalid move' if not move_success else None
        }
    
    def move(self, action: int) -> bool:

        if action < 0 or action >= len(World.DIRECTIONS):
            return False
            
        dx, dy = World.DIRECTIONS[action]
        new_x = self.position[0] + dx
        new_y = self.position[1] + dy
        
        if self.world.is_valid_position(new_x, new_y):
            self.position = (new_x, new_y)
            self.path_history.append(self.position)
            return True
        
        return False
    
    def get_current_sensors(self) -> List[int]:

        return self.world.get_sensor_readings(self.position)
    
    def get_distance_to_goal(self) -> float:

        return self.world.get_distance_to_goal(self.position)
    
    def is_at_goal(self) -> bool:

        return self.world.is_goal_reached(self.position)
    
    def get_state_data(self, timestamp: int, run_id: int) -> dict:

        return {
            'timestamp': timestamp,
            'run_id': run_id,
            'position': self.position,
            'sensors': self.get_current_sensors(),
            'distance_to_goal': self.get_distance_to_goal(),
            'path_length': len(self.path_history) - 1,  # Exclude starting position
            'goal_direction': self.world.get_goal_direction_radians(self.position)
        } 