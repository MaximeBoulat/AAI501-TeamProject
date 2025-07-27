# 🤖 Robot Navigation AI Project

This project explores multiple approaches to robot navigation using both classical pathfinding (A*) and supervised machine learning techniques. It evaluates how well a robot trained on sensor data can predict optimal movement and compares this against the A* benchmark path.

---

## 📂 Project Structure

```
├── generate_labels_with_astar.py       # Generates A* labels for supervised learning
├── simulate_robot_policy.py            # Simulates robot using trained ML model
├── visualize_astar_path.py             # Visualizes A* path from one configuration
├── visualize_robot_path.py             # Visualizes ML-predicted path
├── compare_paths_and_confusion.py      # Compares ML vs A* paths and performs analysis
├── robot_training_data.csv             # Original dataset
├── robot_data_with_astar_labels.csv    # Dataset with added 'astar_action' column
├── simulated_robot_path.csv            # Predicted path by ML model
├── disagreement_log.csv                # Logged mismatches between A* and true labels
```

---

## ✅ Project Steps

### 1. `generate_labels_with_astar.py`
- Builds local 3×3 obstacle maps from sensor data
- Uses A* to compute the optimal direction
- Outputs new column `astar_action`

📄 Output: `robot_data_with_astar_labels.csv`

---

### 2. `simulate_robot_policy.py`
- Loads a trained ML model (e.g., Random Forest)
- Simulates robot path based on predicted directions

📄 Output: `simulated_robot_path.csv`

---

### 3. `visualize_astar_path.py`
- Visualizes the full A* path for a sample state

🖼️ Output: `astar_path_plot.png`

---

### 4. `visualize_robot_path.py`
- Plots ML-predicted robot path using simulated output

🖼️ Output: `robot_path_plot.png`

---

### 5. `compare_paths_and_confusion.py`
- Side-by-side visualization of A* vs ML path
- Confusion matrix of human label vs A* label
- Logs rows where A* and human labels differ
- Visualizes disagreement patterns:
  - By human action
  - Sensor heatmap
  - Sorted by distance to goal
  - Clustered by run ID

🖼️ Outputs:
- `compare_paths.png`
- `confusion_matrix_astar_vs_action.png`
- `disagreements_by_action.png`
- `sensor_heatmap_mismatches.png`
- `disagreements_by_runid.png`

📄 Logs:
- `disagreement_log.csv`

---

## 🧠 Technologies Used
- Python 3.10+
- scikit-learn
- matplotlib
- seaborn
- pandas

---

## 📌 Summary
This project demonstrates how sensor-based learning can approximate A* optimal behavior, where it diverges, and how feature importance, spatial path planning, and model interpretability all contribute to explainable AI.

This structure is designed for evaluation, comparison, and presentation of real-world robotics simulation using both algorithmic and learning-based approaches.

---


