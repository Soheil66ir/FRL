# Reinforcement learning using TD3 with Hindsight Experience Replay (HER) algorithm for Robotic Arm contol (The Fetch environments)
===================================================================

This repository contains the implementation of a **TD3** (Twin Delayed Deep Deterministic Policy Gradient) agent with **Hindsight Experience Replay (HER)** for training in the  Fetch environments using PyTorch and OpenAI Gym. The agent is designed to solve sparse-reward robotic tasks by learning from both normal and hindsight experiences.

# Table of Contents
-----------------

-   [Overview](#overview)
-   [Features](#features)
-   [Installation](#installation)
-   [Usage](#usage)
-   [Files](#files)
-   [Results](#results)
-   [References](#references)

# Overview
--------

The project implements the TD3 algorithm with HER, enabling the agent to train more efficiently in sparse reward environments by sampling and replaying transitions with substituted goals. This approach helps the agent learn from failed experiences by imagining what would have happened if the goal had been achieved, improving learning in environments like FetchPickAndPlace-v2, where rewards are sparse.

### Key Components:

-   **TD3 (Twin Delayed DDPG)**: A state-of-the-art reinforcement learning algorithm that addresses overestimation bias in the Q-values by using two critics and updating the actor less frequently.
-   **Hindsight Experience Replay (HER)**: A technique for sparse reward environments where failed experiences are re-evaluated by substituting the goal with the achieved goal, allowing the agent to learn even from failures.

# Features
--------

-   **TD3 Algorithm**: Implements the TD3 algorithm for stable and efficient learning.
-   **Hindsight Experience Replay (HER)**: Adds hindsight experiences to the replay buffer for improved learning in sparse reward environments.
-   **Fetch Environment**: A robotic control task from OpenAI's Gym, where a robotic arm has several task to complete.
-   **PyTorch Implementation**: Built using PyTorch for easy customisation and experimentation.
-   **Training and Evaluation**: Includes scripts for training the agent and evaluating its performance after training.

# Installation
------------

### Requirements

1.  Python 3.10+
2.  Install dependencies via `requirements.txt`:

`pip install -r requirements.txt`

### Additional Libraries

This project uses `gymnasium` for environments and PyTorch for deep learning. Ensure you have the necessary packages installed:

`pip install torch gymnasium gymnasium-robotics numpy matplotlib pandas`

For the Fetch environment, you need to install `mujoco`:

`pip install mujoco`

Ensure you have the required dependencies for MuJoCo and configure it by following the instructions [here](https://github.com/google-deepmind/mujoco).

# Usage
-----

### 1\. Training the Agent

You can start training the TD3 agent with HER using the following command in the desired environment's directory (For ):

`python Main.py`

The training process will:

-   Initialise the environment.
-   Train the agent for the specified number of epochs and episodes per epoch.
-   Save the trained models and training metrics (actor and critic losses, Q-values, success rates, etc.) in the `models` directory.

### 2\. Rendering the Agent

Once the model is trained, you can render the agent performing in the environment using:

`python Render.py`

This script loads the trained actor model and visualizes the agent's performance in real time.

### 3\. Evaluation

After training, you can view the metrics by loading the CSV files generated during training. These include episode rewards, success rates, and loss values stored in the `models/metrics.csv`.

# Files
-----

-   `Networks.py`: Contains the definitions of the Actor and Critic networks for TD3.
-   `Memory.py`: Implements the Replay Buffer and Hindsight Experience Replay (HER) buffer for experience storage and sampling.
-   `Agent.py`: Defines the TD3 agent, including action selection, training, and updating actor-critic networks.
-   `Main.py`: Main training loop for the TD3 agent with HER in the Fetch environment.
-   `Render.py`: Script for rendering the trained TD3 agent in the environment.
-   `requirements.txt`: Contains all the necessary libraries and dependencies for running the project.

# Results
-------

The training process will output:

-   **Trained models**: The final actor and critic networks are saved in the `models` directory as `.pth` files.
-   **Training metrics**: Training metrics (episode rewards, actor/critic losses, Q-values, success rates) are stored in `models/metrics.csv`.
-   **Graphs**: Plots of accumulated rewards, success rates, and loss values are saved as `.png` images in the `models` directory.

### Example Plots

-   **Accumulated Reward per Episode**
-   **Success Rate per Epoch**
-   **Actor Loss and Critic Loss during Training**
-   **Q-values during Training**

# References
----------

-   **TD3 Paper**: Fujimoto, Scott, et al. "Addressing Function Approximation Error in Actor-Critic Methods." ICML, 2018. [Link to paper](https://arxiv.org/abs/1802.09477)

-   **HER Paper**: Andrychowicz, Marcin, et al. "Hindsight Experience Replay." NIPS, 2017. [Link to paper](https://arxiv.org/abs/1707.01495)

-   **Gymnasium-Robotics**: A collection of robotics simulation environments for Reinforcement Learning. [Gymnasium-Robotics Website](https://robotics.farama.org/)

-   **Mujoco**: A general purpose physics engine. [Mujoco Website](https://mujoco.org/)
