# On-Assessing-The-Safety-of-Reinforcement-Learning-algorithms-Using-Formal-Methods
This code contain files that we use to implement our techniques regarding the behavior of an RL agent when facing moving adversaries.

# Prerequisites
Python 3.7

numpy 1.16

[Prism model checker](https://www.prismmodelchecker.org/download.php)

# Implementation

Each .py files is ready to run and implement a scenario of our environment.

1. 3_patrolling_adversary.py implements the RL agent when facing the 3-cell patrolling adversary
2. 5_patrolling_adversary.py implements the RL agent when facing the 5-cell patrolling adversary
3. Learning_adversary_with_observation.py implements the RL agent when facing the learning adversary. In which case the latter has observations about the environment
4. Learning_adversary_zero_observation.py implements the RL agent when facing the learning adversary. In which case the latter has zero observations about the environment
5. Potential_based_reward_shaping.py implements the potential based reward shaping functions as defense mechanisms to improve the learning of the RL agent. 
6. Q_learning_modified.py implements the modified Q-learning algorithm as defense mechanism to improve the agent learning
7. dtcm.prism implements the automata of the environment

# BIBTEX REFERENCE

