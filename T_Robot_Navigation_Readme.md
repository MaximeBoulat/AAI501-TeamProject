
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
