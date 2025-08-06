Learning Based Navigation for Robots in Random 2D Worlds
Max Boulat, Tanya Neustice, and Dylan Scott-Dawkins
Department of Engineering, San Diego University
AAI-501-01: Introduction to Artificial Intelligence Andrew Van Benschoten
August 11, 2025

# Abstract

Autonomous navigation through complex environments is a fundamental challenge in robotics and artificial intelligence.  This project explores whether a robot can learn to navigate in a randomly generated two-dimensional world with obstacles using supervised learning.  A world generator creates 30×30 grids populated with single-tile obstacles and random linear walls; an agent equipped with eight radial sensors receives the distance to the nearest obstacle in each direction together with its Euclidean distance to the goal.  An A* path-planner produces optimal actions, which serve as the supervisory signal for training.  We perform exploratory data analysis to understand the distribution of sensor readings and evaluate multiple classification algorithms—including Random Forest, XGBoost, logistic regression, support vector machines, Naïve Bayes, K-nearest neighbors and a neural network—on the resulting dataset.  We find that all models struggle to exceed 0.37 accuracy due to severe information asymmetry and shifting-signals problems inherent in the data.  Our analysis draws on related work in partially observable Markov decision processes, imitation learning and differentiable planning to contextualize the limitations and recommend future directions.

Keywords: random forest, logistic regression, Naïve Bayes, neural networks, A*, logistic regression

# Robots in Random 2D Worlds

Navigation in environments with uncertainty and partial observability is an important problem in artificial intelligence and robotics.  Early research such as ALVINN demonstrated that a neural network could map raw sensory input to steering commands for an autonomous vehicle:contentReference[oaicite:0]{index=0}.  The formal framework for decision making with incomplete state information is the **partially observable Markov decision process** (POMDP), which models an agent that cannot directly observe the underlying state and must maintain a belief state:contentReference[oaicite:1]{index=1}.  Solving POMDPs exactly is computationally hard, spurring the development of approximate planning and learning methods.  A* search, a heuristic graph-search algorithm that combines the actual cost of a path with an admissible heuristic estimate, remains the standard for optimal path finding in fully observable domains:contentReference[oaicite:2]{index=2}.

The final team project in our applied artificial intelligence course asks us to identify an AI-driven problem, conduct a hands-on project and produce a report and presentation.  We chose to investigate whether a robot can learn to navigate random two-dimensional environments with obstacles using supervised learning.  Using A* as an expert, we generate trajectories and collect sensor readings and actions.  We then train a variety of classification models to predict the next move from sensor input, compare their performance and discuss the inherent limitations.  Our objectives are to:

1. Define an artificial world and sensor model suitable for machine-learning experiments.
2. Generate a labelled dataset by following optimal paths computed by A*.
3. Analyze the dataset to understand feature distributions, correlations and potential issues such as label contradictions.
4. Train and evaluate different machine-learning algorithms on the navigation problem.
5. Relate our findings to the broader literature on learning-based navigation and imitation learning.

## 2 Methodology

### 2.1 World generation and sensors

The environment is a square grid of size 20×20.  A world generator populates the grid with single-tile obstacles based on a probability (`obstacle_prob = 0.2`) and adds several horizontal or vertical walls of random lengths.  The start and goal locations are randomly selected so that the Euclidean distance between them is at least eight cells and neither lies on an obstacle.  Each simulation generates a new world, ensuring a diverse set of layouts.  The agent can move in eight directions corresponding to the Moore neighborhood (left, left+down, down, right+down, right, right+up, up, left+up).

To navigate, the agent initially received local information solely from eight radial distance sensors. Each sensor reports the number of unobstructed tiles from the agent’s position in one of eight directions, up to the nearest wall or obstacle. As part of our methodology to improve learning performance, we incrementally introduced additional features.

First, we added the Euclidean distance to the goal, giving the agent a global sense of how far it remained from its objective. This feature provided useful context that wasn't captured by the local sensors alone, helping guide movement decisions more effectively. 

We then added goal direction as computed angle between the agent and the goal to give more spatial context.

We also explored normalizing the direction to the goal as a unit vector, which offered a modest but consistent performance boost by encoding directional intent in a format that complements the sensor layout.

#### Dylan

With these enhancements, the final raw state representation consisted of sensor_0 to sensor_7, the distance_to_goal, and optionally, a normalized direction vector to the goal. A tuple of the agent’s coordinates was logged for reference but not used during training.

#### Max

With these enhancements, the final raw state representation consisted of sensor_0 to sensor_7, the distance_to_goal, and the goal_direction.

The target variable initially remained the action chosen by A*—an integer from 0 to 7 representing one of the eight possible moves. To investigate whether this discrete classification setup limited learning performance, we also experimented with a continuous formulation of the target variable, such as using the unit direction vector of the optimal move (ŷ as a 2D vector). This change aimed to provide a smoother learning signal and encourage better generalization, particularly in ambiguous or edge-case states.

### 2.1 Choosing Appropriate Algorithm Structures (Can move to model section?)

Selecting the right structure for the various AI selected models, for example, the neural network architecture is a critical part of our methodology. The structure of the network—including the number of layers, the number of units per layer, activation functions, and regularization strategies—can significantly affect the model's ability to learn from the input features and generalize to new environments.

We approached architecture selection empirically, starting with simple fully connected (feedforward) networks and adjusting based on performance. Shallower networks tended to underfit the problem, especially once we introduced more nuanced features like distance_to_goal and normalized direction vectors. Deeper architectures provided the capacity to model more complex relationships between inputs and the optimal actions, but came with increased risk of overfitting. We mitigated this using techniques such as dropout, early stopping, and batch normalization. The choice of output representation (categorical vs. continuous ŷ) also influenced architecture decisions. For classification targets, a softmax output layer paired with cross-entropy loss was appropriate. For continuous direction vectors, we used a linear output layer and optimized with mean squared error (MSE). In both cases, the architecture had to align with the nature of the prediction target to ensure stable and effective learning.

This iterative tuning of network structure was essential to achieving reliable performance and forms a core part of our methodology.

### 2.2 Data generation using A*

A* search uses a priority queue to explore nodes with the lowest estimated total cost (the cost so far plus a heuristic).  We use the Euclidean distance to the goal as the heuristic.  When constructing the dataset, we run A* on each randomly generated world to compute an optimal path from the start to the goal.  For every step along the path, we record the timestamp, run identifier, current position, the eight sensor readings, the Euclidean distance to the goal, the current path length and the action taken.  

| **Column**                 | **Description**                                   |
| -------------------------- | ------------------------------------------------- |
| `timestamp`                | Global index across all runs                      |
| `run_id`                   | Simulation run identifier                         |
| `position_x`, `position_y` | Agent’s coordinates (not used as features)        |
| `sensor_0…sensor_7`        | Distances to nearest obstacle in eight directions |
| `distance_to_goal`         | Euclidean distance to goal                        |
| `path_length`              | Steps taken so far                                |
| `action`                   | Optimal move (0–7) as determined by A*            |

We generated multiple batches of data:

- Version 1.1: Sample size 3000 runs (~40,000 labelled instances), no goal direction  
- Version 2.2: Sample size 3000 runs (~40,000 labelled instances), goal direction as a feature  
- Version 2.3: Sample size 10000 runs (~140,000 labelled instances), goal direction as a feature  

### 2.3 Exploratory data analysis

To gain deeper insight into the updated dataset, we performed a comprehensive exploratory data analysis (EDA). We begin by examining pairwise relationships among the eight sensor readings, the Euclidean distance to the goal and the normalized goal direction. The correlation matrix in Figure 1 shows that most sensors are only weakly correlated with each other; sensors on opposite axes exhibit slight negative correlation, while the distance_to_goal and goal_direction features correlate modestly with a few sensors. This confirms that the sensors provide largely independent views of the local environment and that the global features introduce complementary information.

Next, we assessed how informative each feature is for predicting the optimal action by fitting a Random Forest classifier to the training set and extracting feature importances (Figure 2). The two global features—distance_to_goal and goal_direction—emerge as the most important predictors, followed by sensors aligned with the primary axes. This indicates that augmenting local sensor readings with even a coarse notion of goal direction dramatically improves the agent’s ability to choose appropriate actions.

The shifting‑signals problem arises because identical sensor readings can correspond to different optimal actions in different worlds. To quantify this, we discretized sensor values into three bins (low, medium, high) and counted the number of unique actions associated with each binned sensor pattern. Figure 3 shows a histogram of the number of distinct actions per pattern. A substantial fraction of patterns correspond to two or more actions, illustrating that the label distribution conditioned on local observations is highly multi‑modal. This empirical evidence underscores the information asymmetry between the A* planner and the agent’s local perception.

Finally, we analysed how the agent’s optimal action depends on the relative orientation to the goal. We divided the continuous `goal_direction` variable into eight equal angular sectors (e.g., north, north‑east, etc.) and computed the conditional distribution of actions for each sector. The heatmap in Figure 4 reveals a clear relationship: when the goal lies in a particular sector, the agent overwhelmingly selects actions that point toward that direction, but it still occasionally chooses neighbouring moves to avoid obstacles. This visualisation demonstrates how goal_direction acts as a strong prior while still allowing for detours.

 :agentCitation{citationIndex='0' label='Correlation matrix of sensors, distance_to_goal and goal_direction'}


**Figure 1.** Correlation matrix for the eight sensor values, distance_to_goal and goal_direction in the updated dataset. Most pairwise correlations are near zero, indicating that sensors capture distinct local information. Distance_to_goal and goal_direction correlate moderately with a few sensors, suggesting they provide complementary global context.

 :agentCitation{citationIndex='1' label='Random Forest feature importances for navigation features'}


**Figure 2.** Random Forest feature importances. The global features (distance_to_goal and goal_direction) are the most informative, followed by sensors along the cardinal directions. This ranking highlights the value of including goal-direction information in the state representation.

 :agentCitation{citationIndex='2' label='Histogram of unique actions per binned sensor pattern'}


**Figure 3.** Histogram of the number of distinct optimal actions per binned sensor pattern. Many local sensor configurations map to multiple actions, evidencing the shifting‑signals problem: identical inputs can correspond to different expert actions depending on the global environment.

 :agentCitation{citationIndex='3' label='Heatmap of action probabilities by goal direction bin'}


**Figure 4.** Conditional probability of each action given the binned goal direction. Rows correspond to angular sectors (e.g., north, north‑east, etc.), and columns correspond to the eight actions. Actions aligned with the goal sector dominate, but adjacent actions still occur, reflecting obstacle avoidance.

### 3 Results

(*Results text would go here, summarising model performance and comparing versions; see the report for details.*)

### 3.2 Dylan's results + interpretation

(*Placeholder for Dylan’s analysis.*)

## 4 Related work

The difficulty of learning navigation policies from supervised data is well recognized in the literature. Pomerleau’s ALVINN system imitated human steering by training on camera and laser readings:contentReference[oaicite:3]{index=3}; while promising, it worked only on simple road scenes and suffered when the environment changed.  Kaelbling, Littman and Cassandra formalized POMDPs, highlighting the challenge of acting under partial observability:contentReference[oaicite:4]{index=4}. Later, Ross et al. introduced DAgger, an imitation-learning algorithm that addresses distribution shift by iteratively collecting expert feedback along the learner’s own trajectories:contentReference[oaicite:5]{index=5}. Our shifting-signals problem is a concrete manifestation of the same issue: data generated by following an expert does not cover states that the learner might encounter.

Hausknecht and Stone proposed the Deep Recurrent Q-Network (DRQN) to handle partially observable environments by maintaining a hidden state over time. Tamar et al. introduced **Value Iteration Networks (VIN)**, neural networks that embed a differentiable planning module and learn to perform approximate value iteration:contentReference[oaicite:6]{index=6}. VINs have been applied to grid-world navigation and could provide a more principled way to combine local observations with implicit planning. More recent work such as Neural Map incorporates external memory to build an internal map, which is critical when the task requires exploration and recall of previously visited locations.

Other research on autonomous driving emphasizes the gap between high step-wise action accuracy and actual performance. Codevilla et al. show that behavior-cloned policies with high action agreement can still crash because they lack planning and fail to recover from mistakes. Therefore, evaluation metrics should include path efficiency, collision rates and goal success rather than solely action prediction accuracy.

## 5 Discussion

Our experiments reveal several insights about using supervised learning to imitate a path planner in a partially observable environment.

1. **Global context is critical.** The feature importance analysis (Figure 2) shows that the distance_to_goal and goal_direction features contribute more to the prediction than any single sensor.  Models that incorporate even a coarse notion of where the goal lies achieve dramatically higher accuracy than those relying solely on local sensors.

2. **Information asymmetry causes shifting labels.** The contradiction analysis (Figure 3) quantifies how many sensor patterns map to multiple optimal actions.  A substantial fraction of local observations correspond to two or more different labels, confirming that the agent’s partial view leads to an ill‑posed supervised problem.

3. **Class imbalance persists.** Actions pointing toward the goal dominate the dataset, while backtracking or side steps are rare.  Models trained on such data may underperform in scenarios requiring detours. Addressing imbalance through resampling or weighting could help, but ultimately richer training data or alternative learning paradigms are needed.

4. **Temporal dependencies are ignored.** Each training instance is treated as independent, yet effective navigation depends on the sequence of prior moves.  Recurrent architectures (e.g., LSTM or GRU) could allow the agent to integrate information over time and may reduce contradictions by remembering previously observed obstacles.

5. **Evaluation must reflect navigation success.** Our evaluation focuses on step‑wise classification accuracy and F1 score.  However, high action agreement does not guarantee that an agent will reach the goal efficiently or avoid collisions.  Future work should report path length ratios, success rates and collision counts.

### 5.1 Potential improvements

* **Reduce sensor range** to two or three tiles to encourage repeated patterns and limit the feature space.
* **Augment features** with the relative angle to the goal rather than just the Euclidean distance.
* **Incorporate memory** using recurrent networks (e.g., LSTM) to aggregate information across multiple steps.
* **Use imitation-learning algorithms such as DAgger**, which query the expert for additional labels when the learner deviates, reducing distribution shift.
* **Explore reinforcement learning** with intrinsic exploration rewards.
* **Address class imbalance** by generating more trajectories requiring detours or applying class-weighting and resampling strategies.

## 6 Conclusion

This project investigated whether an agent could learn to navigate randomly generated two‑dimensional worlds with obstacles using supervised learning. We generated training data by following optimal A* paths and trained a variety of classifiers to imitate the expert. The updated exploratory analysis showed that local sensors alone are insufficient: many sensor patterns map to multiple optimal actions and the most informative features are the global cues of distance_to_goal and goal_direction. Including these global features increases test accuracy from roughly 0.4 to around 0.8 for tree‑based models and neural networks, yet contradictions remain and models struggle on rare detour actions. These findings reinforce that pure behaviour cloning is ill‑posed under partial observability and that performance saturates once the agent’s local field of view is exhausted. Future work should combine imitation learning with algorithms that reduce distribution shift (e.g., DAgger), incorporate memory (e.g., recurrent networks), embed differentiable planning modules (e.g., Value Iteration Networks) or leverage reinforcement learning to learn policies directly from experience.

# References

- Barreto, A., Dabney, W., Munos, R., Hunt, J. J., Schaul, T., van Hasselt, H., & Silver, D. (2018). *Successor features for transfer in reinforcement learning*. arXiv. https://arxiv.org/abs/1606.05312  
- Codevilla, F., Santana, E., López, A. M., & Gaidon, A. (2019). *Exploring the limitations of behavior cloning for autonomous driving*. arXiv. https://arxiv.org/abs/1904.08980  
- Delgado, K. V., de Barros, L. N., Dias, D. B., & Sanner, S. (2016). *Real-time dynamic programming for Markov decision processes with imprecise probabilities*. *Artificial Intelligence, 230*, 192-223. https://doi.org/10.1016/j.artint.2015.09.005  
- Hausknecht, M. J., & Stone, P. (2015). *Deep recurrent Q-learning for partially observable MDPs*. arXiv. http://arxiv.org/abs/1507.06527  
- Mathieu, M., Ozair, S., Srinivasan, S., Gulcehre, C., Zhang, S., Jiang, R., Le Paine, T., Powell, R., Żołna, K., Schrittwieser, J., Choi, D., Georgiev, P., Toyama, D., Huang, A., Ring, R., Babuschkin, I., Ewalds, T., Bordbar, M., Henderson, S., … Vinyals, O. (2023). *AlphaStar Unplugged: Large-scale offline reinforcement learning*. arXiv. https://arxiv.org/abs/2308.03526  
- Parisotto, E., & Salakhutdinov, R. (2017). *Neural Map: Structured memory for deep reinforcement learning*. arXiv. https://arxiv.org/abs/1702.08360  
- Petrović, L. (2018). *Motion planning in high-dimensional spaces*. arXiv. https://arxiv.org/abs/1806.07457  
- Pomerleau, D. A. (1989). *ALVINN: An autonomous land vehicle in a neural network*. In D. Touretzky (Ed.), *Advances in Neural Information Processing Systems* (Vol. 1). Morgan Kaufmann.  
- Ross, S., Gordon, G. J., & Bagnell, J. A. (2011). *A reduction of imitation learning and structured prediction to no-regret online learning*. arXiv. https://arxiv.org/abs/1011.0686  
- Tamar, A., Wu, Y., Thomas, G., Levine, S., & Abbeel, P. (2017). *Value iteration networks*. arXiv. https://arxiv.org/abs/1602.02867  
- Xia, F., Li, C., Chen, K., Shen, W. B., Martín-Martín, R., Hirose, N., Zamir, A. R., Fei-Fei, L., & Savarese, S. (2019). *Gibson Env V2: Embodied simulation environments for interactive navigation* (Tech. Rep.). Stanford University. http://svl.stanford.edu/gibson2  
