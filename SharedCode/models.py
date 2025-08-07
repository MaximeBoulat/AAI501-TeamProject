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
from sklearn.preprocessing import StandardScaler

from world import World
from config import *
import joblib



USE_SCALER = True

def log_model_results(model_type: str, accuracy: float, results_file: str = "model_results.csv"):
    try:
        
        # Check if results file exists
        if os.path.exists(results_file):
            df = pd.read_csv(results_file)
        else:
            df = pd.DataFrame(columns=['model_type', 'experiment_name', 'accuracy'])
        
        # Remove any existing rows with the same model_type and experiment_name
        # Convert columns to string to ensure type consistency
        if not df.empty:
            df['experiment_name'] = df['experiment_name'].astype(str)
            mask = (df['model_type'] == model_type) & (df['experiment_name'] == EXPERIMENT_NAME)
            df = df[~mask]
        
        # Add new row with enhanced tracking
        new_row = pd.DataFrame({
            'model_type': [model_type],
            'experiment_name': [EXPERIMENT_NAME],
            'accuracy': [round(accuracy, 3)]
        })
        
        df = pd.concat([df, new_row], ignore_index=True)
        
        # Sort by model_type and experiment_name for better organization
        df = df.sort_values(['model_type', 'experiment_name']).reset_index(drop=True)
        
        # Save to CSV
        df.to_csv(results_file, index=False)
        print(f"Logged results for {model_type} ({EXPERIMENT_NAME}): {accuracy:.3f} accuracy")
        
    except Exception as e:
        print(f"Error logging results for {model_type}: {e}")
        import traceback
        traceback.print_exc()

class BaseModel(ABC):
    
    def __init__(self):
        
        self.model = None
        self.scaler = None
        # Loop detection attributes - using hash table for O(1) lookups
        self.position_counts = {}  # Hash table to track position visit counts
        self.position_queue = []   # Queue to maintain order for cleanup
        self.max_history_length = 5  # Track last 5 positions (reduced for efficiency)
        self.loop_detection_threshold = 2  # Consider loop if position visited 2+ times recently

    def from_file(self, model_type: str):
        model_dir = f"models/{model_type}"
        model_filename = f"{model_dir}/model_{EXPERIMENT_NAME}.pkl"
        scaler_filename = f"{model_dir}/scaler_{EXPERIMENT_NAME}.pkl"
        try:
            self.model = joblib.load(model_filename)
            print(f"Loaded model from {model_filename}")
            
            if USE_SCALER:
                self.scaler = joblib.load(scaler_filename)
                print(f"Loaded scaler from {scaler_filename}")
           
            
        except Exception as e:
            print(f"Error loading model {model_type}: {e}")
        
    def reset(self):
        
        # Clear position tracking when resetting
        self.position_counts = {}
        self.position_queue = []
    
    def _detect_loop(self, current_position: Tuple[int, int]) -> bool:

        # O(1) lookup in hash table - much faster than iterating through list
        position_count = self.position_counts.get(current_position, 0)
        
        # If we've been to this position multiple times recently, it's likely a loop
        return position_count >= self.loop_detection_threshold
    
    def _update_position_history(self, position: Tuple[int, int]):
        """Update the position tracking using hash table and queue for efficient cleanup."""
        # Add position to queue for order tracking
        self.position_queue.append(position)
        
        # Increment count in hash table - O(1) operation
        self.position_counts[position] = self.position_counts.get(position, 0) + 1
        
        # Clean up old positions if we exceed max history length
        if len(self.position_queue) > self.max_history_length:
            # Remove oldest position from queue and decrement its count
            old_position = self.position_queue.pop(0)
            self.position_counts[old_position] -= 1
            
            # Remove from hash table if count reaches 0 to save memory
            if self.position_counts[old_position] == 0:
                del self.position_counts[old_position]

    def _train_model(self, model_type: str, csv_file: str = TRAINING_DATA_FILE):
   
        try:
            # Load training data
            df = pd.read_csv(csv_file)
            print(f"Loaded {len(df)} training samples from {csv_file}")

            X = df[X_COLS].values
            y = df[Y_HAT_COL[0]].values

            # Split into train/test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if USE_SCALER:
                # Scale the data
                self.scaler = StandardScaler()
                X_train = self.scaler.fit_transform(X_train)
                X_test = self.scaler.transform(X_test)

            # Train the model
            self.model.fit(X_train, y_train)

            # Evaluate on test set
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"{model_type} training completed. Test accuracy: {accuracy:.3f}")

            # Log results to CSV with enhanced naming convention
            log_model_results(model_type, accuracy)

            # save the model and scaler to the file system
            model_dir = f"models/{model_type}"
            os.makedirs(model_dir, exist_ok=True)
            model_filename = f"{model_dir}/model_{EXPERIMENT_NAME}.pkl"
            scaler_filename = f"{model_dir}/scaler_{EXPERIMENT_NAME}.pkl"
            try:
                joblib.dump(self.model, model_filename)
                print(f"Saved trained model to {model_filename}")
                
                if USE_SCALER:
                    joblib.dump(self.scaler, scaler_filename)
                    print(f"Saved scaler to {scaler_filename}")

                
            except Exception as e:
                print(f"Error saving model {model_type}: {e}")
            
            
        except FileNotFoundError:
            print(f"Training data file {csv_file} not found. Model will not be available.")
            self.model = None
        except Exception as e:
            print(f"Error training SVM model: {e}")
            self.model = None
    
    def get_next_action(self, current_position: Tuple[int, int], world: World) -> Optional[int]:
        
        # Check for loop before making any action decision
        if self._detect_loop(current_position):
            print(f"Loop detected at position {current_position}. Returning invalid action.")
            return None
        
        # Update position history for loop detection
        self._update_position_history(current_position)
        
        if self.model is None or (USE_SCALER and self.scaler is None):
            return None
            
        # Get current sensor readings and distance
        sensors = world.get_sensor_readings(current_position)
        distance_to_goal = world.get_distance_to_goal(current_position)
        goal_direction = world.get_goal_direction_radians(current_position)
        
        # Prepare input for model
        features = np.array([sensors + [distance_to_goal, goal_direction]])
        
        if USE_SCALER:
            # Scale the features using the saved scaler
            features = self.scaler.transform(features)
        
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

        

class RandomForestModel(BaseModel):
    
    def train_model(self, n_estimators: int = 100, random_state: int = 42):

        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        self._train_model("RandomForest")
    

class LogisticRegressionModel(BaseModel):
    
    def train_model(self, random_state: int = 42):
    
        self.model = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            random_state=random_state,
            max_iter=1000
        )
        self._train_model("LogisticRegression")
    

class SVMModel(BaseModel):

    def train_model(self, kernel: str = 'rbf', random_state: int = 42):

        self.model = SVC(
            kernel=kernel,
            random_state=random_state,
            gamma='scale'
        )
        self._train_model("SVM")
 


class NaiveBayesModel(BaseModel):

    def train_model(self):

        self.model = GaussianNB()
        self._train_model("NaiveBayes")
   

class KNNModel(BaseModel):

    def train_model(self, n_neighbors: int = 5):

        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self._train_model("KNN")
   

class XGBoostModel(BaseModel):

    def train_model(self, n_estimators: int = 100, random_state: int = 42):

        try:
            import xgboost as xgb
            self.model = xgb.XGBClassifier(
                n_estimators=n_estimators, 
                random_state=random_state,
                eval_metric='mlogloss'
            )
            self._train_model("XGBoost")
        except ImportError:
            print("XGBoost is not available. Install with: pip install xgboost")
            self.model = None
        except Exception as e:
            print(f"XGBoost initialization failed: {e}")
            print("For macOS users: Run 'brew install libomp' to install OpenMP runtime")
            self.model = None


class NeuralNetworkModel(BaseModel):

    def train_model(self, hidden_layer_sizes: tuple = NN_HIDDEN_LAYERS, 
                 random_state: int = 42, max_iter: int = 500):

        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            random_state=random_state,
            max_iter=max_iter,
            solver='adam'
        )
        self._train_model("NeuralNetwork")
        

