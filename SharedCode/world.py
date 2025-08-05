import numpy as np
import random
from typing import Tuple, List
import math

class World:
    
    # 8-directional movement: left, left+down, down, right+down, right, right+up, up, left+up
    DIRECTIONS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    
    def __init__(self, grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]):

        self.grid = grid
        self.start = start
        self.goal = goal
        self.size = grid.shape[0]  # Assuming square grid
        
    @classmethod
    def from_random(cls, size: int = 20, 
                    obstacle_prob: float = 0.3, 
                    wall_count: int = 5, 
                    max_wall_length: int = 10, 
                    min_start_goal_distance: int = 8, 
                    seed: int = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        attempts = 0
        max_attempts = 100
        
        while attempts < max_attempts:
            # Create grid with random obstacles
            grid = np.zeros((size, size), dtype=int)
            
            # Add random single-tile obstacles
            for y in range(size):
                for x in range(size):
                    if random.random() < obstacle_prob:
                        grid[y][x] = 1
            
            # Add horizontal or vertical walls
            for _ in range(wall_count):
                is_horizontal = random.choice([True, False])
                wall_length = random.randint(3, max_wall_length)
                
                if is_horizontal:
                    y = random.randint(0, size - 1)
                    x_start = random.randint(0, size - wall_length)
                    for i in range(wall_length):
                        if x_start + i < size:
                            grid[y][x_start + i] = 1
                else:
                    x = random.randint(0, size - 1)
                    y_start = random.randint(0, size - wall_length)
                    for i in range(wall_length):
                        if y_start + i < size:
                            grid[y_start + i][x] = 1
            
            # Generate start and goal positions
            start = (random.randint(0, size - 1), random.randint(0, size - 1))
            goal = (random.randint(0, size - 1), random.randint(0, size - 1))
            
            # Ensure start and goal are not blocked
            if grid[start[1]][start[0]] == 1 or grid[goal[1]][goal[0]] == 1:
                attempts += 1
                continue
                
            # Ensure minimum distance between start and goal
            dist = np.linalg.norm(np.subtract(start, goal))
            if dist >= min_start_goal_distance:
                return cls(grid, start, goal)
                
            attempts += 1
        
        raise RuntimeError(f"Could not generate valid world after {max_attempts} attempts")
    
    def is_valid_position(self, x: int, y: int) -> bool:
        return (0 <= x < self.size and 
                0 <= y < self.size and 
                self.grid[y][x] == 0)
    
    def get_sensor_readings(self, position: Tuple[int, int]) -> List[int]:
        readings = []
        x, y = position
        
        for dx, dy in self.DIRECTIONS:
            distance = 0
            curr_x, curr_y = x, y
            
            while True:
                curr_x += dx
                curr_y += dy
                distance += 1
                
                # Hit boundary or obstacle
                if not (0 <= curr_x < self.size and 0 <= curr_y < self.size) or self.grid[curr_y][curr_x] == 1:
                    break
                    
            readings.append(distance)
        
        return readings
    
    def is_goal_reached(self, position: Tuple[int, int]) -> bool:
        return position == self.goal
    
    def get_distance_to_goal(self, position: Tuple[int, int]) -> float:
        return np.linalg.norm(np.subtract(self.goal, position))
    
    def get_goal_direction_radians(self, position: Tuple[int, int]) -> float:
        dx = self.goal[0] - position[0]
        dy = position[1] - self.goal[1]
        angle = math.atan2(dx, dy)  # dx first because 0° = north
        angle = angle % (2 * math.pi) / (2 * math.pi) # Normalize to [0, 1)
        return angle

    
    def get_neighbors(self, position: Tuple[int, int]) -> List[Tuple[int, int]]:
        neighbors = []
        x, y = position
        
        for dx, dy in self.DIRECTIONS:
            new_x, new_y = x + dx, y + dy
            if self.is_valid_position(new_x, new_y):
                neighbors.append((new_x, new_y))
        
        return neighbors 