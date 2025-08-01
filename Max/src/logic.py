from abc import ABC, abstractmethod
from typing import Tuple, List, Optional
import heapq
import numpy as np
import pandas as pd
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

class Logic(ABC):
    """Abstract base class for agent logic/strategy implementations."""
    
    @abstractmethod
    def get_next_action(self, current_position: Tuple[int, int], world: World) -> Optional[int]:
        """
        Get the next action for the agent to take.
        Args:
            current_position: Agent's current position
            world: World containing goal and obstacles
        Returns:
            action number (0-7) or None if no valid action.
        """
        pass
        
    @abstractmethod
    def reset(self):
        """Reset any internal state for a new run."""
        pass
        
    def _get_action_for_move(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> int:
        """Convert position move to action number (0-7)."""
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        
        for i, (ddx, ddy) in enumerate(World.DIRECTIONS):
            if (dx, dy) == (ddx, ddy):
                return i
        
        return -1  # Invalid move

class AStarLogic(Logic):
    """A* pathfinding logic implementation."""
    
    def __init__(self):
        """Initialize A* logic."""
        self.full_path = []
        self.current_step = 0
        
    def reset(self):
        """Reset internal state for new run."""
        self.full_path = []
        self.current_step = 0
    
    def get_next_action(self, current_position: Tuple[int, int], world: World) -> Optional[int]:
        """Get next action using A* pathfinding."""
        path = self._compute_astar_path(world, current_position, world.goal)
        
        if len(path) >= 2:
            from_pos = path[0]
            to_pos = path[1]
            return self._get_action_for_move(from_pos, to_pos)
        
        return None
    
    
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

class RandomForestLogic(Logic):
    """Random Forest classifier logic implementation."""
    
    def __init__(self, csv_file: str = "training_data.csv", n_estimators: int = 100, random_state: int = 42):
        """
        Initialize Random Forest logic with training data.
        
        Args:
            csv_file: Path to training data CSV file
            n_estimators: Number of trees in the forest
            random_state: Random seed for reproducibility
        """
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        self._train_model(csv_file)
    
    def _train_model(self, csv_file: str):
        """Train the Random Forest model on the provided data."""
        try:
            # Load training data
            df = pd.read_csv(csv_file)
            print(f"Loaded {len(df)} training samples from {csv_file}")
 
            # count rows
            print(f"Number of rows: {len(df)}")
            
            # Prepare features: 8 sensor readings + distance_to_goal
            feature_columns = [f'sensor_{i}' for i in range(8)] + ['distance_to_goal']
            X = df[feature_columns].values
            y = df['action'].values
            
            # Split into train/test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train the model
            self.model.fit(X_train, y_train)
            
            # Evaluate on test set
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Random Forest training completed. Test accuracy: {accuracy:.3f}")
            
        except FileNotFoundError:
            print(f"Training data file {csv_file} not found. Model will not be available.")
            self.model = None
        except Exception as e:
            print(f"Error training Random Forest model: {e}")
            self.model = None
    
    def reset(self):
        """Reset any internal state."""
        pass
    
    def get_next_action(self, current_position: Tuple[int, int], world: World) -> Optional[int]:
        """Get next action using Random Forest prediction."""
        if self.model is None:
            return None
            
        # Get current sensor readings and distance
        sensors = world.get_sensor_readings(current_position)
        distance_to_goal = world.get_distance_to_goal(current_position)
        
        # Prepare input for model
        features = np.array([sensors + [distance_to_goal]])
        
        # Get model prediction
        try:
            action = self.model.predict(features)[0]
            
            # Validate action is in valid range
            if 0 <= action <= 7:
                return int(action)
        except Exception as e:
            print(f"Random Forest prediction error: {e}")
        
        return None

class LogisticRegressionLogic(Logic):
    """Multinomial Logistic Regression classifier logic implementation."""
    
    def __init__(self, csv_file: str = "training_data.csv", random_state: int = 42):
        """
        Initialize Logistic Regression logic with training data.
        
        Args:
            csv_file: Path to training data CSV file
            random_state: Random seed for reproducibility
        """
        self.model = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            random_state=random_state,
            max_iter=1000
        )
        self._train_model(csv_file)
    
    def _train_model(self, csv_file: str):
        """Train the Logistic Regression model on the provided data."""
        try:
            # Load training data
            df = pd.read_csv(csv_file)
            print(f"Loaded {len(df)} training samples from {csv_file}")
            
            # Prepare features: 8 sensor readings + distance_to_goal
            feature_columns = [f'sensor_{i}' for i in range(8)] + ['distance_to_goal']
            X = df[feature_columns].values
            y = df['action'].values
            
            # Split into train/test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train the model
            self.model.fit(X_train, y_train)
            
            # Evaluate on test set
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Logistic Regression training completed. Test accuracy: {accuracy:.3f}")
            
        except FileNotFoundError:
            print(f"Training data file {csv_file} not found. Model will not be available.")
            self.model = None
        except Exception as e:
            print(f"Error training Logistic Regression model: {e}")
            self.model = None
    
    def reset(self):
        """Reset any internal state."""
        pass
    
    def get_next_action(self, current_position: Tuple[int, int], world: World) -> Optional[int]:
        """Get next action using Logistic Regression prediction."""
        if self.model is None:
            return None
            
        # Get current sensor readings and distance
        sensors = world.get_sensor_readings(current_position)
        distance_to_goal = world.get_distance_to_goal(current_position)
        
        # Prepare input for model
        features = np.array([sensors + [distance_to_goal]])
        
        # Get model prediction
        try:
            action = self.model.predict(features)[0]
            
            # Validate action is in valid range
            if 0 <= action <= 7:
                return int(action)
        except Exception as e:
            print(f"Logistic Regression prediction error: {e}")
        
        return None

class SVMLogic(Logic):
    """Support Vector Machine classifier logic implementation."""
    
    def __init__(self, csv_file: str = "training_data.csv", kernel: str = 'rbf', random_state: int = 42):
        """
        Initialize SVM logic with training data.
        
        Args:
            csv_file: Path to training data CSV file
            kernel: SVM kernel type
            random_state: Random seed for reproducibility
        """
        self.model = SVC(
            kernel=kernel,
            random_state=random_state,
            gamma='scale'
        )
        self._train_model(csv_file)
    
    def _train_model(self, csv_file: str):
        """Train the SVM model on the provided data."""
        try:
            # Load training data
            df = pd.read_csv(csv_file)
            print(f"Loaded {len(df)} training samples from {csv_file}")
            
            # Prepare features: 8 sensor readings + distance_to_goal
            feature_columns = [f'sensor_{i}' for i in range(8)] + ['distance_to_goal']
            X = df[feature_columns].values
            y = df['action'].values
            
            # Split into train/test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train the model
            self.model.fit(X_train, y_train)
            
            # Evaluate on test set
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"SVM training completed. Test accuracy: {accuracy:.3f}")
            
        except FileNotFoundError:
            print(f"Training data file {csv_file} not found. Model will not be available.")
            self.model = None
        except Exception as e:
            print(f"Error training SVM model: {e}")
            self.model = None
    
    def reset(self):
        """Reset any internal state."""
        pass
    
    def get_next_action(self, current_position: Tuple[int, int], world: World) -> Optional[int]:
        """Get next action using SVM prediction."""
        if self.model is None:
            return None
            
        # Get current sensor readings and distance
        sensors = world.get_sensor_readings(current_position)
        distance_to_goal = world.get_distance_to_goal(current_position)
        
        # Prepare input for model
        features = np.array([sensors + [distance_to_goal]])
        
        # Get model prediction
        try:
            action = self.model.predict(features)[0]
            
            # Validate action is in valid range
            if 0 <= action <= 7:
                return int(action)
        except Exception as e:
            print(f"SVM prediction error: {e}")
        
        return None

class NaiveBayesLogic(Logic):
    """Naive Bayes classifier logic implementation."""
    
    def __init__(self, csv_file: str = "training_data.csv"):
        """
        Initialize Naive Bayes logic with training data.
        
        Args:
            csv_file: Path to training data CSV file
        """
        self.model = GaussianNB()
        self._train_model(csv_file)
    
    def _train_model(self, csv_file: str):
        """Train the Naive Bayes model on the provided data."""
        try:
            # Load training data
            df = pd.read_csv(csv_file)
            print(f"Loaded {len(df)} training samples from {csv_file}")
            
            # Prepare features: 8 sensor readings + distance_to_goal
            feature_columns = [f'sensor_{i}' for i in range(8)] + ['distance_to_goal']
            X = df[feature_columns].values
            y = df['action'].values
            
            # Split into train/test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train the model
            self.model.fit(X_train, y_train)
            
            # Evaluate on test set
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Naive Bayes training completed. Test accuracy: {accuracy:.3f}")
            
        except FileNotFoundError:
            print(f"Training data file {csv_file} not found. Model will not be available.")
            self.model = None
        except Exception as e:
            print(f"Error training Naive Bayes model: {e}")
            self.model = None
    
    def reset(self):
        """Reset any internal state."""
        pass
    
    def get_next_action(self, current_position: Tuple[int, int], world: World) -> Optional[int]:
        """Get next action using Naive Bayes prediction."""
        if self.model is None:
            return None
            
        # Get current sensor readings and distance
        sensors = world.get_sensor_readings(current_position)
        distance_to_goal = world.get_distance_to_goal(current_position)
        
        # Prepare input for model
        features = np.array([sensors + [distance_to_goal]])
        
        # Get model prediction
        try:
            action = self.model.predict(features)[0]
            
            # Validate action is in valid range
            if 0 <= action <= 7:
                return int(action)
        except Exception as e:
            print(f"Naive Bayes prediction error: {e}")
        
        return None

class KNNLogic(Logic):
    """K-Nearest Neighbors classifier logic implementation."""
    
    def __init__(self, csv_file: str = "training_data.csv", n_neighbors: int = 5):
        """
        Initialize KNN logic with training data.
        
        Args:
            csv_file: Path to training data CSV file
            n_neighbors: Number of neighbors to consider
        """
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self._train_model(csv_file)
    
    def _train_model(self, csv_file: str):
        """Train the KNN model on the provided data."""
        try:
            # Load training data
            df = pd.read_csv(csv_file)
            print(f"Loaded {len(df)} training samples from {csv_file}")
            
            # Prepare features: 8 sensor readings + distance_to_goal
            feature_columns = [f'sensor_{i}' for i in range(8)] + ['distance_to_goal']
            X = df[feature_columns].values
            y = df['action'].values
            
            # Split into train/test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train the model
            self.model.fit(X_train, y_train)
            
            # Evaluate on test set
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"KNN training completed. Test accuracy: {accuracy:.3f}")
            
        except FileNotFoundError:
            print(f"Training data file {csv_file} not found. Model will not be available.")
            self.model = None
        except Exception as e:
            print(f"Error training KNN model: {e}")
            self.model = None
    
    def reset(self):
        """Reset any internal state."""
        pass
    
    def get_next_action(self, current_position: Tuple[int, int], world: World) -> Optional[int]:
        """Get next action using KNN prediction."""
        if self.model is None:
            return None
            
        # Get current sensor readings and distance
        sensors = world.get_sensor_readings(current_position)
        distance_to_goal = world.get_distance_to_goal(current_position)
        
        # Prepare input for model
        features = np.array([sensors + [distance_to_goal]])
        
        # Get model prediction
        try:
            action = self.model.predict(features)[0]
            
            # Validate action is in valid range
            if 0 <= action <= 7:
                return int(action)
        except Exception as e:
            print(f"KNN prediction error: {e}")
        
        return None

class XGBoostLogic(Logic):
    """XGBoost classifier logic implementation."""
    
    def __init__(self, csv_file: str = "training_data.csv", n_estimators: int = 100, random_state: int = 42):
        """
        Initialize XGBoost logic with training data.
        
        Args:
            csv_file: Path to training data CSV file
            n_estimators: Number of boosting rounds
            random_state: Random seed for reproducibility
        """
        try:
            import xgboost as xgb
            self.model = xgb.XGBClassifier(
                n_estimators=n_estimators, 
                random_state=random_state,
                eval_metric='mlogloss'
            )
            self._train_model(csv_file)
        except ImportError:
            print("XGBoost is not available. Install with: pip install xgboost")
            self.model = None
        except Exception as e:
            print(f"XGBoost initialization failed: {e}")
            print("For macOS users: Run 'brew install libomp' to install OpenMP runtime")
            self.model = None
    
    def _train_model(self, csv_file: str):
        """Train the XGBoost model on the provided data."""
        if self.model is None:
            return
            
        try:
            # Load training data
            df = pd.read_csv(csv_file)
            print(f"Loaded {len(df)} training samples from {csv_file}")
            
            # Prepare features: 8 sensor readings + distance_to_goal
            feature_columns = [f'sensor_{i}' for i in range(8)] + ['distance_to_goal']
            X = df[feature_columns].values
            y = df['action'].values
            
            # Split into train/test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train the model
            self.model.fit(X_train, y_train)
            
            # Evaluate on test set
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"XGBoost training completed. Test accuracy: {accuracy:.3f}")
            
        except FileNotFoundError:
            print(f"Training data file {csv_file} not found. Model will not be available.")
            self.model = None
        except Exception as e:
            print(f"Error training XGBoost model: {e}")
            self.model = None
    
    def reset(self):
        """Reset any internal state."""
        pass
    
    def get_next_action(self, current_position: Tuple[int, int], world: World) -> Optional[int]:
        """Get next action using XGBoost prediction."""
        if self.model is None:
            return None
            
        # Get current sensor readings and distance
        sensors = world.get_sensor_readings(current_position)
        distance_to_goal = world.get_distance_to_goal(current_position)
        
        # Prepare input for model
        features = np.array([sensors + [distance_to_goal]])
        
        # Get model prediction
        try:
            action = self.model.predict(features)[0]
            
            # Validate action is in valid range
            if 0 <= action <= 7:
                return int(action)
        except Exception as e:
            print(f"XGBoost prediction error: {e}")
        
        return None

class NeuralNetworkLogic(Logic):
    """Neural Network (MLP) classifier logic implementation."""
    
    def __init__(self, csv_file: str = "training_data.csv", hidden_layer_sizes: tuple = (100, 50), 
                 random_state: int = 42, max_iter: int = 500):
        """
        Initialize Neural Network logic with training data.
        
        Args:
            csv_file: Path to training data CSV file
            hidden_layer_sizes: Tuple of hidden layer sizes
            random_state: Random seed for reproducibility
            max_iter: Maximum number of iterations
        """
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            random_state=random_state,
            max_iter=max_iter,
            solver='adam'
        )
        self._train_model(csv_file)
    
    def _train_model(self, csv_file: str):
        """Train the Neural Network model on the provided data."""
        try:
            # Load training data
            df = pd.read_csv(csv_file)
            print(f"Loaded {len(df)} training samples from {csv_file}")
            
            # Prepare features: 8 sensor readings + distance_to_goal
            feature_columns = [f'sensor_{i}' for i in range(8)] + ['distance_to_goal']
            X = df[feature_columns].values
            y = df['action'].values
            
            # Split into train/test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train the model
            self.model.fit(X_train, y_train)
            
            # Evaluate on test set
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Neural Network training completed. Test accuracy: {accuracy:.3f}")
            
        except FileNotFoundError:
            print(f"Training data file {csv_file} not found. Model will not be available.")
            self.model = None
        except Exception as e:
            print(f"Error training Neural Network model: {e}")
            self.model = None
    
    def reset(self):
        """Reset any internal state."""
        pass
    
    def get_next_action(self, current_position: Tuple[int, int], world: World) -> Optional[int]:
        """Get next action using Neural Network prediction."""
        if self.model is None:
            return None
            
        # Get current sensor readings and distance
        sensors = world.get_sensor_readings(current_position)
        distance_to_goal = world.get_distance_to_goal(current_position)
        
        # Prepare input for model
        features = np.array([sensors + [distance_to_goal]])
        
        # Get model prediction
        try:
            action = self.model.predict(features)[0]
            
            # Validate action is in valid range
            if 0 <= action <= 7:
                return int(action)
        except Exception as e:
            print(f"Neural Network prediction error: {e}")
        
        return None

class ModelLogic(Logic):
    """Machine learning model logic implementation."""
    
    def __init__(self, model):
        """Initialize with trained model."""
        self.model = model
    
    def reset(self):
        """Reset any internal state."""
        pass
    
    def get_next_action(self, current_position: Tuple[int, int], world: World) -> Optional[int]:
        """Get next action using trained model prediction."""
        # Get current sensor readings and distance
        sensors = world.get_sensor_readings(current_position)
        distance_to_goal = world.get_distance_to_goal(current_position)
        
        # Prepare input for model
        features = sensors + [distance_to_goal]
        
        # Get model prediction
        try:
            action = self.model.predict([features])[0]
            
            # Validate action is in valid range
            if 0 <= action <= 7:
                return int(action)
        except Exception as e:
            print(f"Model prediction error: {e}")
        
        return None 