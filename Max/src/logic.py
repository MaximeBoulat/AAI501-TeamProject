from abc import ABC, abstractmethod
from typing import Tuple, List, Optional
import heapq
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from world import World

# XGBoost import handled in XGBoostLogic class due to potential library issues

# Data schema version for tracking

# 1.0: sample size 100
# 1.1: sample size 3000
# 2.0: sample size 3000, goal direction added
# 2.1: sample size 10000
# 2.2: match Dylan config
# 2.3: use Dylan's dataset

DATA_SCHEMA_VERSION = "2.0"

class Logic(ABC):
    """Abstract base class for agent logic/strategy implementations."""
    
    def __init__(self):
        """Initialize logic with loop detection."""
        self.position_history = []
        self.max_history_length = 10  # Track last 10 positions
        self.loop_detection_threshold = 3  # Consider loop if position visited 3+ times recently
        
    @abstractmethod
    def reset(self):
        """Reset any internal state for a new run."""
        # Clear position history when resetting
        self.position_history = []
        
    def _get_action_for_move(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> int:
        """Convert position move to action number (0-7)."""
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        
        for i, (ddx, ddy) in enumerate(World.DIRECTIONS):
            if (dx, dy) == (ddx, ddy):
                return i
        
        return -1  # Invalid move
    
    def _detect_loop(self, current_position: Tuple[int, int]) -> bool:
        """
        Detect if the agent is stuck in a loop by checking recent position history.
        
        Args:
            current_position: The current position to check
            
        Returns:
            True if a loop is detected, False otherwise
        """
        # Count how many times current position appears in recent history
        recent_visits = self.position_history[-8:] if len(self.position_history) >= 8 else self.position_history
        position_count = recent_visits.count(current_position)
        
        # If we've been to this position multiple times recently, it's likely a loop
        return position_count >= self.loop_detection_threshold
    
    def _update_position_history(self, position: Tuple[int, int]):
        """Update the position history for loop detection."""
        self.position_history.append(position)
        
        # Keep only the most recent positions
        if len(self.position_history) > self.max_history_length:
            self.position_history = self.position_history[-self.max_history_length:]

    def get_next_action(self, current_position: Tuple[int, int], world: World) -> Optional[int]:
        
        # Check for loop before making any action decision
        if self._detect_loop(current_position):
            print(f"Loop detected at position {current_position}. Returning invalid action.")
            return None
        
        # Update position history for loop detection
        self._update_position_history(current_position)
        
        if self.model is None:
            return None
            
        # Get current sensor readings and distance
        sensors = world.get_sensor_readings(current_position)
        distance_to_goal = world.get_distance_to_goal(current_position)
        goal_direction = world.get_goal_direction_radians(current_position)
        
        # Prepare input for model
        features = np.array([sensors + [distance_to_goal, goal_direction]])
        
        # Get model prediction
        try:
            action = self.model.predict(features)[0]

            print(f"action: {action}")
            
            # Validate action is in valid range
            if 0 <= action <= 7:
                return int(action)
        except Exception as e:
            print(f"prediction error: {e}")
        
        return None


class AStarLogic(Logic):
    """A* pathfinding logic implementation."""
    
    def __init__(self):
        """Initialize A* logic."""
        super().__init__()  # Initialize loop detection
        self.full_path = []
        self.current_step = 0
        
    def reset(self):
        """Reset internal state for new run."""
        super().reset()  # Reset loop detection
        self.full_path = []
        self.current_step = 0
    

    def _compute_astar_path(self, world: World, start: Tuple[int, int], 
                           goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Compute A* path from start to goal."""
        open_set = []
        heapq.heappush(open_set, (0 + self._heuristic(start, goal), 0, start, []))
        visited = set()
        
        while open_set:
            _, cost, current, path = heapq.heappop(open_set)
            
            if current in visited:
                continue
            visited.add(current)
            
            if current == goal:
                return path + [current]
            
            # Explore neighbors
            for neighbor in world.get_neighbors(current):
                if neighbor not in visited:
                    move_cost = self._get_move_cost(current, neighbor)
                    new_cost = cost + move_cost
                    heuristic = self._heuristic(neighbor, goal)
                    total_cost = new_cost + heuristic
                    
                    heapq.heappush(open_set, (total_cost, new_cost, neighbor, path + [current]))
        
        return []  # No path found
    
    def _heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Heuristic function for A* (Euclidean distance)."""
        return np.linalg.norm(np.subtract(pos1, pos2))
    
    def _get_move_cost(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> float:
        """Get cost of moving from one position to another."""
        return np.linalg.norm(np.subtract(to_pos, from_pos))
    
    def get_full_path(self, world: World, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get complete A* path for visualization or analysis."""
        return self._compute_astar_path(world, start, goal)
    
    def get_next_action(self, current_position: Tuple[int, int], world: World) -> Optional[int]:
        print(f"current_position: {current_position}")
        print(f"sensors: {world.get_sensor_readings(current_position)}")
        print(f"distance_to_goal: {world.get_distance_to_goal(current_position)}")
        print(f"goal_direction: {world.get_goal_direction_radians(current_position)}")
        """Get next action using A* pathfinding with loop detection."""
        # Check for loop before making any action decision
        if self._detect_loop(current_position):
            print(f"Loop detected at position {current_position}. Returning invalid action.")
            return None
        
        # Update position history for loop detection
        self._update_position_history(current_position)
        
        # Compute or use cached path
        if not self.full_path or self.current_step >= len(self.full_path):
            # Compute new path from current position to goal
            self.full_path = self._compute_astar_path(world, current_position, world.goal)
            self.current_step = 0
            
            # If no path found, return invalid action
            if not self.full_path:
                return None
        
        # Get next position in path
        if self.current_step + 1 < len(self.full_path):
            next_position = self.full_path[self.current_step + 1]
            action = self._get_action_for_move(current_position, next_position)
            self.current_step += 1
            
            if action != -1:
                print(f"action: {action}")
                return action
        
        return None

class RandomForestLogic(Logic):
    """Random Forest classifier logic implementation."""
    
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        """
        Initialize Random Forest logic with training data.
        
        Args:
            csv_file: Path to training data CSV file
            n_estimators: Number of trees in the forest
            random_state: Random seed for reproducibility
        """
        super().__init__()  # Initialize loop detection
        self.model = joblib.load("/models/ RandomForest.pkl")

    def reset(self):
        """Reset any internal state."""
        super().reset()  # Reset loop detection
    
    

class LogisticRegressionLogic(Logic):
    """Multinomial Logistic Regression classifier logic implementation."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize Logistic Regression logic with training data.
        
        Args:
            csv_file: Path to training data CSV file
            random_state: Random seed for reproducibility
        """
        super().__init__()  # Initialize loop detection
        self.model = joblib.load("/models/LogisticRegression.pkl")
   
    def reset(self):
        """Reset any internal state."""
        super().reset()  # Reset loop detection
    

class SVMLogic(Logic):
    """Support Vector Machine classifier logic implementation."""
    
    def __init__(self, kernel: str = 'rbf', random_state: int = 42):
        """
        Initialize SVM logic with training data.
        
        Args:
            csv_file: Path to training data CSV file
            kernel: SVM kernel type
            random_state: Random seed for reproducibility
        """
        super().__init__()  # Initialize loop detection
        self.model = joblib.load("/models/SVM.pkl")
 
    def reset(self):
        """Reset any internal state."""
        super().reset()  # Reset loop detection

class NaiveBayesLogic(Logic):
    """Naive Bayes classifier logic implementation."""
    
    def __init__(self):
        """
        Initialize Naive Bayes logic with training data.
        
        Args:
            csv_file: Path to training data CSV file
        """
        super().__init__()  # Initialize loop detection
        self.model = joblib.load("/models/NaiveBayes.pkl")
   
    def reset(self):
        """Reset any internal state."""
        super().reset()  # Reset loop detection
    
class KNNLogic(Logic):
    """K-Nearest Neighbors classifier logic implementation."""
    
    def __init__(self):
        """
        Initialize KNN logic with training data.
        
        Args:
            csv_file: Path to training data CSV file
            n_neighbors: Number of neighbors to consider
        """
        super().__init__()  # Initialize loop detection
        self.model = joblib.load("/models/KNN.pkl")
   
    def reset(self):
        """Reset any internal state."""
        super().reset()  # Reset loop detection

class XGBoostLogic(Logic):
    """XGBoost classifier logic implementation."""
    
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        """
        Initialize XGBoost logic with training data.
        
        Args:
            csv_file: Path to training data CSV file
            n_estimators: Number of boosting rounds
            random_state: Random seed for reproducibility
        """
        super().__init__()  # Initialize loop detection
        try:
            self.model = joblib.load("/models/XGBoost.pkl")
        except ImportError:
            print("XGBoost is not available. Install with: pip install xgboost")
            self.model = None
        except Exception as e:
            print(f"XGBoost initialization failed: {e}")
            print("For macOS users: Run 'brew install libomp' to install OpenMP runtime")
            self.model = None

    def reset(self):
        """Reset any internal state."""
        super().reset()  # Reset loop detection

class NeuralNetworkLogic(Logic):
    """Neural Network (MLP) classifier logic implementation."""
    
    def __init__(self, hidden_layer_sizes: tuple = (100, 50), 
                 random_state: int = 42, max_iter: int = 500):
        """
        Initialize Neural Network logic with training data.
        
        Args:
            csv_file: Path to training data CSV file
            hidden_layer_sizes: Tuple of hidden layer sizes
            random_state: Random seed for reproducibility
            max_iter: Maximum number of iterations
        """
        super().__init__()  # Initialize loop detection
        self.model = joblib.load("/models/NeuralNetwork.pkl")

    def reset(self):
        """Reset any internal state."""
        super().reset()  # Reset loop detection
    
