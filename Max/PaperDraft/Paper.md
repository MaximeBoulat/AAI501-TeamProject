# Max's PathFinding Research Paper

## Overview

In this paper we discuss methods to make a robot navigate a randomly generated 2D world with obstacles using artificial intelligence.

### What kind of real-world problems does that research applies to?

Any problems that involve pathfinding in a dynamic environment such as:

- The Roomba scenario
- characters in a video games
- Search and rescue, bomb defusal, attack drones etc.

## Methodology

### Tile based map

In order to simplify the problem space and make it easier to digest, we used a 2D tile based map where tiles can either be empty or obstructed to represent the world.

The agent can move in 8 directions and captures sensor readings in the 8 directions. Sensor readings contain the distance to the nearest obstacle in the direction of the sensor.

### Expert system (A*) as the source of truth

We had A* find the optimal path across multiple runs to generate the dataset.

## Feature selection

### Proposed schema

| Column | Description |
|--------|-------------|
| `timestamp` | Global step index across all runs |
| `run_id` | ID of the simulation run |
| `sensor_0...7` | Distance to obstacle in 8 directions |
| `action` | Direction taken (0-7) |
| `distance_to_goal` | Euclidean distance to goal from current cell |
| `goal_direction` | Direction measured as a decimal number between 0 and 1 where 0 is north. |

### Topics for consideration:

#### Shifting signals in the training data

The shifting signals problem represents more than a data quality issue - it reveals a fundamental information asymmetry between A* and the ML model that makes the learning task ill-defined.

##### Information Asymmetry
A* operates with complete information: full obstacle layout, optimal path planning, global connectivity knowledge, and exact goal position. The ML model receives only local sensor distances and scalar goal distance, creating an impossible learning scenario where the model must make globally optimal decisions with locally insufficient information.

##### Contradictory Labeling Examples
Consider identical sensor readings [3, 5, 2, 8, 6, 1, 4, 7] appearing in different scenarios:
- Scenario A (goal northeast): A* labels optimal action as "move northeast"
- Scenario B (goal southwest): A* labels optimal action as "move southwest"

The model encounters identical input features with opposite target labels. This violates the fundamental ML assumption of deterministic input-output mapping.

##### Measuring Contradiction Scale
The problem severity can be quantified by: (1) grouping training samples by sensor reading patterns, (2) counting unique actions labeled as optimal for each pattern, and (3) calculating label entropy per sensor configuration. High entropy patterns indicate severe contradictory labeling, with many sensor configurations likely having 3-4 different "optimal" actions depending on goal position.

##### Why This Breaks Learning
Standard ML requires consistent labeling - identical inputs should produce identical outputs. The shifting signals create systematic contradictions where equally valid examples have opposite labels. The model cannot converge to consistent decision boundaries because the training data contains contradictory examples that are mathematically impossible to resolve without additional information.

##### Performance Impact
The observed 30-40% accuracy ceiling likely reflects this contradiction. The model learns to predict the most frequent action for each sensor configuration but cannot exceed this threshold because conflicting examples prevent higher accuracy. This suggests the data generation approach requires fundamental revision rather than feature engineering solutions.

#### Density and shape of obstacles

The current obstacle generation creates two distinct types of barriers: randomly distributed single tiles (20% probability per cell) and linear walls (5 walls of 3-10 tiles each). This mixed approach generates environments with varying obstacle density and connectivity patterns.

Single-tile obstacles create scattered navigation challenges that require frequent minor course corrections. Linear walls create major barriers that force significant detours and path planning. The combination results in maps where optimal strategies vary dramatically - some require careful maneuvering through scattered obstacles, others require finding gaps in wall systems.

This heterogeneity may impede learning because the model must handle fundamentally different navigation scenarios within the same training framework. The optimal response to dense scattered obstacles (careful local navigation) differs from the optimal response to large walls (strategic detour planning).

#### Specificity of the sensor readings

Current sensor readings create an enormous sparse feature space. With 8 sensors each reading 1-20 tiles, you get combinations that rarely repeat across different maps. This creates millions of possible sensor combinations, with most patterns appearing only once or a few times in the dataset.

The high dimensionality and sparsity mean the model rarely sees repeated patterns during training. Each sensor configuration is essentially unique to specific map layouts and positions, making it difficult for the model to learn generalizable navigation rules rather than memorizing map-specific responses.

This suggests limiting sensor range (e.g., to 2 tiles) could create more repeated patterns. This would reduce the feature space to manageable dimensions where the model encounters the same sensor configurations frequently enough to learn consistent behavioral patterns.

#### Multiple behaviors to learn

Navigation requires dynamic integration of two competing objectives: goal-seeking (moving toward the target) and obstacle avoidance (preventing collisions). These behaviors often conflict - the direct path to the goal may be blocked, requiring detours that temporarily move away from the target.

The model must learn when to prioritize safety over progress and when to accept risk for efficiency. This requires contextual decision-making that depends on factors like obstacle density, goal proximity, and available alternative paths. A single classifier cannot effectively encode these competing priorities without explicit guidance about when each behavior should dominate.

The current approach assumes these behaviors can be implicitly learned from A* examples, but A* solves this through explicit path planning algorithms. The ML model lacks the architectural components to replicate this planning process.

#### Action space imbalance

The 8-directional action space may exhibit significant class imbalance. Certain actions (like moving toward the goal) are more frequent than others (like backtracking around obstacles). This creates a biased training signal where the model over-learns common actions and under-learns critical but rare maneuvers.

Additionally, the discrete 8-direction action space may be too coarse for effective navigation. Optimal paths often require precise movements that don't align with the fixed directional grid, forcing A* to approximate smooth trajectories with angular segments.

#### Temporal dependencies ignored

Current feature representation treats each navigation decision as independent, but effective navigation requires sequential reasoning. Whether a particular action is optimal depends on recent movement history, current trajectory, and planned future moves.

For example, the decision to move around an obstacle depends on which side was chosen previously and whether the robot is committed to a particular detour path. The single-step classification approach cannot capture these temporal dependencies that are essential for coherent navigation behavior.

#### Evaluation metric limitations

Accuracy measures how often the model predicts the same action as A*, but this doesn't measure navigation effectiveness. A model could achieve high accuracy while producing navigation paths that are significantly longer, get stuck in loops, or fail to reach the goal.

The evaluation should measure path quality metrics (path length, goal reach rate, navigation efficiency) rather than action prediction accuracy. High accuracy on action prediction may not translate to effective navigation performance.

## Literature review

The field of AI that this research applies to is called spatial reasoning under uncertainty.

The timeline of the milestones in this field is as follows:

- 1989: ALVINN: neural imitation of driving from visual input
- 1998: Formalization of POMDPs (Markov chains) (Kaelbling et al.)
- 2011: DAgger addresses covariate shift in imitation learning
- 2015: DRQN handles memory-based control in POMDPs
- 2016: Value Iteration Networks add differentiable planning
- 2018: Neural Map and learned SLAM-style agents emerge
- 2020: End-to-end agents combine mapping, memory, and learning (Habitat, Gibson, etc.)

### Early Foundations of Learning-Based Navigation

The idea of using sensory data to drive autonomous behavior dates back to Pomerleau (1989), who introduced ALVINN, one of the first successful applications of neural networks to navigation. ALVINN trained a model to imitate human steering behavior using forward-facing camera input, effectively laying the groundwork for behavior cloning (BC).

While effective in constrained settings, this approach assumed the data distribution during training matched that during execution—a flaw later formalized as the distributional shift problem.

### Formalizing Decision-Making Under Uncertainty

A crucial theoretical milestone came with Kaelbling, Littman, and Cassandra (1998), who introduced a formal treatment of Partially Observable Markov Decision Processes (POMDPs). This provided the mathematical foundation for decision-making in environments where the agent cannot observe the full state—directly relevant to robot navigation using only local sensor inputs.

The critique of "information asymmetry" between A* and the ML model parallels this framework: A* operates under full observability, while the learned model relies solely on partial sensor data, making the learning task ill-posed under standard supervised assumptions.

### Addressing Behavior Cloning Failures

To address the distribution mismatch inherent in BC, Ross et al. (2011) introduced DAgger (Dataset Aggregation), an iterative imitation learning method that collects expert labels along the learner's own trajectories. DAgger effectively reduces compounding errors in sequential decision tasks.

The shifting signals and contradictory labeling caused by state aliasing mentioned earlier are well-aligned with the rationale behind DAgger, which assumes the learner needs additional corrective supervision when encountering states not seen during expert demonstrations.

### Learning Sequential Decision Policies

Single-step classifiers are fundamentally limited in navigation tasks that involve temporal dependencies. This mirrors the motivation behind Hausknecht and Stone (2015), who introduced the Deep Recurrent Q-Network (DRQN)—an extension of Deep Q-Networks (DQN) using LSTMs to model hidden state over time. DRQN was explicitly designed for POMDP settings and is foundational for learning policies where observation history informs action.

### Embedding Planning Structures in Neural Networks

Models trained solely on local data struggle to emulate global planners like A*. Tamar et al. (2016) addressed this with Value Iteration Networks (VINs), which incorporate differentiable planning modules into neural networks. VINs are capable of learning approximate planning behavior using only local information.

More recent work like Neural Map (Parisotto & Salakhutdinov, 2018) introduced external spatial memory structures to allow agents to build internal maps of unseen environments, again targeting the same problem you're facing: reconciling local observations with long-term planning objectives.

### Empirical Failures and Metric Misalignment

Agents with high step-wise agreement can still perform poorly overall. This mirrors empirical findings in autonomous driving research—e.g., Codevilla et al. (2018, 2021)—which showed that BC-trained policies with high action agreement still crashed or deviated significantly due to cumulative errors and lack of strategic reasoning.

These studies advocate for task-level evaluation metrics like goal success rate, path efficiency, and collision count—exactly the metrics you propose adopting.

### Practical Simulation Frameworks and Scaling

Recent large-scale environments like Habitat, Gibson, and DeepMind Lab have become benchmarks for training and evaluating policies in partially observable 3D environments. These frameworks standardize sensor modeling, goal-directed navigation, and task completion metrics—areas you touch on with your emphasis on sensor range and entropy of sensor configurations.

## Model selection

### Decision tree

#### Support vector machines

#### Random forest

#### Naive Bayes

#### Logistic multinomial regression

#### KNN

#### XGBoost

#### Neural network

## Results

| Algorithm | Accuracy |
|-----------|----------|
| XGBoost | 0.879 |
| Logistic Regression | 0.521 |
| Random Forest | 0.877 |
| SVM | 0.463 |
| Naive Bayes | 0.688 |
| KNN | 0.273 |
| Neural Network | 0.854 |

## References

- Barreto, A., Dabney, W., Munos, R., Hunt, J. J., Schaul, T., van Hasselt, H., & Silver, D. (2018). *Successor features for transfer in reinforcement learning*. arXiv. https://arxiv.org/abs/1606.05312

- Codevilla, F., Santana, E., López, A. M., & Gaidon, A. (2019). *Exploring the limitations of behavior cloning for autonomous driving*. arXiv. https://arxiv.org/abs/1904.08980

- Delgado, K. V., de Barros, L. N., Dias, D. B., & Sanner, S. (2016). *Real-time dynamic programming for Markov decision processes with imprecise probabilities*. *Artificial Intelligence, 230*, 192-223. https://doi.org/10.1016/j.artint.2015.09.005

- Hausknecht, M. J., & Stone, P. (2015). *Deep recurrent Q-learning for partially observable MDPs*. arXiv. http://arxiv.org/abs/1507.06527

- Mathieu, M., Ozair, S., Srinivasan, S., Gulcehre, C., Zhang, S., Jiang, R., Le Paine, T., Powell, R., Żołna, K., Schrittwieser, J., Choi, D., Georgiev, P., Toyama, D., Huang, A., Ring, R., Babuschkin, I., Ewalds, T., Bordbar, M., Henderson, S., Gómez Colmenarejo, S., van den Oord, A., Czarnecki, W. M., de Freitas, N., & Vinyals, O. (2023). *AlphaStar Unplugged: Large-scale offline reinforcement learning*. arXiv. https://arxiv.org/abs/2308.03526

- Parisotto, E., & Salakhutdinov, R. (2017). *Neural Map: Structured memory for deep reinforcement learning*. arXiv. https://arxiv.org/abs/1702.08360

- Petrović, L. (2018). *Motion planning in high-dimensional spaces*. arXiv. https://arxiv.org/abs/1806.07457

- Pomerleau, D. A. (1988). *ALVINN: An autonomous land vehicle in a neural network*. In D. Touretzky (Ed.), *Advances in Neural Information Processing Systems* (Vol. 1). Morgan Kaufmann. https://proceedings.neurips.cc/paper_files/paper/1988/file/812b4ba287f5ee0bc9d43bbf5bbe87fb-Paper.pdf

- Ross, S., Gordon, G. J., & Bagnell, J. A. (2011). *A reduction of imitation learning and structured prediction to no-regret online learning*. arXiv. https://arxiv.org/abs/1011.0686

- Tamar, A., Wu, Y., Thomas, G., Levine, S., & Abbeel, P. (2017). *Value iteration networks*. arXiv. https://arxiv.org/abs/1602.02867

- Xia, F., Li, C., Chen, K., Shen, W. B., Martín-Martín, R., Hirose, N., Zamir, A. R., Fei-Fei, L., & Savarese, S. (2019, June 16). *Gibson Env V2: Embodied simulation environments for interactive navigation* (Tech. Rep.). Stanford University. http://svl.stanford.edu/gibson2