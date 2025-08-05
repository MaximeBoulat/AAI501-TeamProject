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


# Data schema version for tracking

# 1.0: sample size 100
# 1.1: sample size 3000
# 2.0: sample size 3000, goal direction added
# 2.1: sample size 10000
# 2.2: match Dylan config
# 2.3: use Dylan's dataset

DATA_SCHEMA_VERSION = "2.0"


class BaseModel(ABC):

    def _train_model(self, model_type: str, csv_file: str = "training_data_2.csv"):
   
        try:
            # Load training data
            df = pd.read_csv(csv_file)
            print(f"Loaded {len(df)} training samples from {csv_file}")
            
            # Prepare features: 8 sensor readings + distance_to_goal
            feature_columns = [f'sensor_{i}' for i in range(8)] + ['distance_to_goal', 'goal_direction']
            X = df[feature_columns].values
            y = df['action'].values
            
            # Split into train/test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Scale the data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train the model
            self.model.fit(X_train_scaled, y_train)

            # Evaluate on test set
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"{model_type} training completed. Test accuracy: {accuracy:.3f}")

            log_model_results(model_type, accuracy)
            
        except FileNotFoundError:
            print(f"Training data file {csv_file} not found. Model will not be available.")
            self.model = None
        except Exception as e:
            print(f"Error training SVM model: {e}")
            self.model = None
    
        
        

class RandomForestModel(BaseModel):
    
    def __init__(self, n_estimators: int = 100, random_state: int = 42):

        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        self._train_model("RandomForest")
    

class LogisticRegressionModel(BaseModel):
    
    def __init__(self, random_state: int = 42):
    
        self.model = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            random_state=random_state,
            max_iter=1000
        )
        self._train_model("LogisticRegression")
    

class SVMModel(BaseModel):

    def __init__(self, kernel: str = 'rbf', random_state: int = 42):

        self.model = SVC(
            kernel=kernel,
            random_state=random_state,
            gamma='scale'
        )
        self._train_model("SVM")
 


class NaiveBayesModel(BaseModel):

    def __init__(self):

        self.model = GaussianNB()
        self._train_model("NaiveBayes")
   

class KNNModel(BaseModel):

    def __init__(self, n_neighbors: int = 5):

        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self._train_model("KNN")
   

class XGBoostModel(BaseModel):

    def __init__(self, n_estimators: int = 100, random_state: int = 42):

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

    def __init__(self, hidden_layer_sizes: tuple = (100, 50), 
                 random_state: int = 42, max_iter: int = 500):

        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            random_state=random_state,
            max_iter=max_iter,
            solver='adam'
        )
        self._train_model("NeuralNetwork")
        

def log_model_results(model_type: str, accuracy: float, results_file: str = "model_results.csv"):

    try:
        
        # Check if results file exists
        if os.path.exists(results_file):
            df = pd.read_csv(results_file)
        else:
            df = pd.DataFrame(columns=['model_type', 'data_schema_version', 'accuracy'])
        
        # Remove any existing rows with the same model_type and schema version
        # Convert data_schema_version to string to ensure type consistency
        df['data_schema_version'] = df['data_schema_version'].astype(str)
        mask = (df['model_type'] == model_type) & (df['data_schema_version'] == DATA_SCHEMA_VERSION)
       
        df = df[~mask]
        
        # Add new row
        new_row = pd.DataFrame({
            'model_type': [model_type],
            'data_schema_version': [DATA_SCHEMA_VERSION],
            'accuracy': [round(accuracy, 3)]
        })
        
        df = pd.concat([df, new_row], ignore_index=True)
        
        # Save to CSV
        df.to_csv(results_file, index=False)
        
    except Exception as e:
        print(f"Error logging results for {model_type}: {e}")
        import traceback
        traceback.print_exc()





RandomForestModel()
LogisticRegressionModel()
SVMModel()
NaiveBayesModel()
KNNModel()
XGBoostModel()
NeuralNetworkModel()