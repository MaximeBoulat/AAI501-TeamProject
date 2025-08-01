
# AI501 Team Project - Model Training, Evaluation, and Comparison

My commits include Exploratory Data Analysis (EDA), multiple model implementations (Random Forest, Naive Bayes, Neural Network), and a performance comparison using accuracy, F1-score, and confusion matrices.

---

## 📁 Project Structure

```
├── eda_outputs/                # Plots and summary from Exploratory Data Analysis
├── classifier_outputs/         # Outputs from Random Forest model
├── bayesian_outputs/           # Outputs from Naive Bayes model
├── neural_net_outputs/         # Outputs from Neural Network model
├── model_comparison_outputs/   # Accuracy and F1-score comparisons
├── robot_training_data.csv     # Main dataset
├── eda_analysis.py             # EDA script
├── classifier_model.py         # Random Forest implementation
├── bayesian_model.py           # Naive Bayes implementation
├── neural_net_model.py         # Neural Network implementation
├── model_comparison.py         # Model comparison with metrics
└── README.md                   # Project overview
```

---

## ✅ Steps Completed

### 1. Exploratory Data Analysis (EDA)
- Performed in `eda_analysis.py`
- Generated plots:
  - Action distribution
  - Correlation matrix
  - Distance metrics
  - Sensor boxplots

### 2. Model Training
- **Random Forest:** in `classifier_model.py`
- **Naive Bayes:** in `bayesian_model.py`
- **Neural Network:** in `neural_net_model.py`
  - Used `tensorflow.keras` for architecture
  - Applied `StandardScaler` for feature normalization

### 3. Model Evaluation
Each model evaluated on:
- Accuracy
- F1-score
- Confusion matrix

### 4. Model Comparison
- Conducted in `model_comparison.py`
- Outputs saved in `model_comparison_outputs/`:
  - CSV summary of scores
  - Bar charts for accuracy and F1-score

---

## ⚙️ How to Run

```bash
# Install dependencies
pip install pandas matplotlib seaborn scikit-learn tensorflow

# Run scripts
python eda_analysis.py
python classifier_model.py
python bayesian_model.py
python neural_net_model.py
python model_comparison.py
```

---

## 🧠 Key Findings

- All models trained on sensor data predicting robot actions.
- Model comparison shows differences in accuracy and F1-score.
- Confusion matrices give insight into classification strengths and weaknesses.

---

