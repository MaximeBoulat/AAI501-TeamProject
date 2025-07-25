---

# Q-Learning for Robot Action Optimization

This script demonstrates a simple Q-learning approach to help a simulated robot learn the optimal action to take based on its sensor inputs, using real-world-like timestamped data. The model is trained using tabular Q-learning, which is effective for environments with a discrete and limited state-action space.

---

## 📂 Dataset Format

The dataset is expected to be a CSV file named `robot_training_data.csv` and structured as follows:

| timestamp | run\_id | sensor\_0 | ... | sensor\_7 | action | distance\_to\_goal |
| --------- | ------- | --------- | --- | --------- | ------ | ------------------ |
| 0         | 0       | 10        | ... | 3         | 2      | 11.7               |

* Each row represents a step in time for a training run.
* `sensor_0` to `sensor_7`: sensor readings at that time.
* `action`: the action taken (e.g., 0, 1, 2, 3).
* `distance_to_goal`: a float representing how far the robot is from the goal. Lower values are better.

---

## 🧠 Learning Setup

### Q-Learning

The algorithm learns a Q-table `Q[state][action]`, which estimates the value of taking a given action in a given state.

* **States**: Discretized vector of 8 sensor values (binned).
* **Actions**: Unique actions in the dataset (mapped to indices).
* **Reward**: Defined as the negative `distance_to_goal` — closer means higher reward.
* **Transitions**: Inferred from the order of rows in the dataset.

---

## ⚙️ Key Hyperparameters

| Name         | Value | Description                                          |
| ------------ | ----- | ---------------------------------------------------- |
| `alpha`      | 0.1   | Learning rate — how much new info overrides old      |
| `gamma`      | 0.9   | Discount factor — how much future rewards are valued |
| `epsilon`    | 0.1   | Exploration rate — chance to choose a random action  |
| `n_bins`     | 5     | Number of discrete bins per sensor                   |
| `n_episodes` | 50    | Number of training epochs over the data              |

---

## 🏋️‍♂️ Training

Each episode simulates reading the dataset sequentially:

* Convert sensor values to discrete state bins.
* Use the Q-learning update rule:

  ```python
  Q[state][action] += alpha * (reward + gamma * max(Q[next_state]) - Q[state][action])
  ```

---

## 🤖 Inference

The function `choose_action(sensor_input)` takes raw sensor values and returns the best learned action:

```python
sample_input = [5, 2, 3, 4, 1, 3, 4, 2]
print(choose_action(sample_input))  # => Best action based on Q-table
```

It uses the same discretizer (`KBinsDiscretizer`) trained on the dataset to match Q-table keys.

---

## 🧪 Evaluation and Next Steps

This implementation assumes:

* A known dataset where actions lead to measurable outcomes.
* Discrete state and action space.
* A relatively stationary environment (no concept of time decay, context shifts).

To improve:

* Add validation by simulating environment transitions.
* Try different reward shaping methods.
* Experiment with Double Q-Learning or DQN for continuous states.

---

## 📚 References

* Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
* KBinsDiscretizer: [https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html)
* Q-Learning: [https://en.wikipedia.org/wiki/Q-learning](https://en.wikipedia.org/wiki/Q-learning)

---
