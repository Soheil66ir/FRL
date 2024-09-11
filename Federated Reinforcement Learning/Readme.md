# Federated Reinforcement (TD3+HER) Learning for Gymnasium-Robotics Fetch environments

==========================================================================================

This repository implements a **TD3** (Twin Delayed Deep Deterministic Policy Gradient) agent with **Hindsight Experience Replay (HER)**, along with **Federated Learning (FL)** capabilities using Flower. The goal is to train a distributed RL agent on a robotic environment using sparse rewards while aggregating model updates from multiple clients in a federated manner.

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

This project integrates the TD3 algorithm, HER, and Flower-based federated learning for distributed RL training. The agent is designed to solve sparse-reward robotic control tasks by learning from hindsight experiences, and the federated approach allows multiple clients to train their models independently and periodically share updates with a central server.

## Key Components:

-   **TD3 (Twin Delayed DDPG)**: A state-of-the-art RL algorithm for continuous control tasks.

-   **Hindsight Experience Replay (HER)**: Helps the agent learn from failures by imagining what would have happened if the goal had been achieved.

-   **Flower (FL)**: A framework for federated learning that enables multiple clients to collaborate by sharing model updates instead of raw data.

# Features

--------

-   **TD3 Algorithm**: Implements the TD3 algorithm with twin critics and delayed policy updates.

-   **Hindsight Experience Replay (HER)**: Enhances learning in sparse reward environments by replaying experiences with substituted goals.

-   **Federated Learning (FL)**: Uses Flower to aggregate model updates from multiple clients and enable distributed training.

-   **Model Checkpointing**: Saves and loads models during training and evaluation phases.

-   **PyTorch Implementation**: Built with PyTorch for ease of customization and experimentation.

# Installation

------------

## Requirements

1\.  Python 3.10+

2\.  Install dependencies via `requirements.txt`:


`pip install -r requirements.txt`

## Additional Libraries

This project uses PyTorch, Flower for FL, and OpenAI's Gymnasium for the robotic control environment. Install the necessary libraries:


`pip install torch gymnasium gymnasium-robotics mujoco flwr pandas`

Ensure you have the required dependencies for MuJoCo and configure it by following the instructions [here](https://github.com/google-deepmind/mujoco).

# Usage

-----

## 1\. Training with Federated Learning

### Server

Start the federated learning server by running:


`python Server.py`

This will initialise the server that aggregates model updates from multiple clients.

### Clients

You can start a client to participate in the federated learning process by running:


`python PickClient.py`  For clients in Pick And Place environment.

`python PushClient.py`  For clients in Push environment.

`python SlideClient.py`  For clients in Slide environment. 

`python ReachClient.py`  For clients in Pick And Place environment.

For Reach environment the state_dim in server should change to 10.

Clients will train their local TD3 agents and periodically send model updates to the server for aggregation.


## 3\. Checkpoints

The server automatically saves model checkpoints after each federated round. These can be loaded later to resume training or for evaluation.

## 4\. Metrics and Logging

Each client logs its local training metrics (reward, success rates, loss values, etc.) in CSV files under the `metrics` directory.

# Files

-----

-   `Networks.py`: Defines the Actor and Critic networks used in the TD3 agent.

-   `Memory.py`: Implements the Replay Buffer and Hindsight Experience Replay (HER) buffer for storing and sampling experiences.

-   `Agent.py`: The TD3 agent, including action selection, training, and policy updates.

-   `Server.py`: The federated learning server using Flower, responsible for aggregating model updates from clients.

-   `PickClient.py`: A Flower client that trains the TD3 agent in Pick and Place environment locally and sends updates to the server.

-   `PushClient.py`: A Flower client that trains the TD3 agent in Push environment locally and sends updates to the server.

-   `SlideClient.py`: A Flower client that trains the TD3 agent in Slide environment locally and sends updates to the server.

-   `ReachClient.py`: A Flower client that trains the TD3 agent in Reach environment locally and sends updates to the server.

-   `requirements.txt`: List of dependencies required to run the project.

# Results

-------

The training process logs the following metrics:

-   **Episode Rewards**: Accumulated rewards per episode.

-   **Success Rate**: The percentage of successful episodes per epoch.

-   **Actor and Critic Losses**: Loss values for both actor and critic networks during training.

-   **Q-values**: Average Q-values for the critic networks during training.

These metrics are saved as CSV files for every agent with a unique identifier in the `metrics` directory for analysis.

## Example Metrics

-   **Accumulated Rewards per Episode**

-   **Success Rate per Epoch**

-   **Actor and Critic Losses**

-   **Q-values during Training**

## Federated Learning Metrics

-   **Success Rate**: Aggregated success rate across all clients after each federated round.

-   **Average Reward**: Aggregated reward across all clients.

## Pre-traied model

A pre-trained Global and local models with 6 agent performing 3 different tasks is in `Pre-trained model with 6 agent` directory

# References

----------

-   **TD3 Paper**: Fujimoto, Scott, et al. "Addressing Function Approximation Error in Actor-Critic Methods." ICML, 2018. [Link to paper](https://arxiv.org/abs/1802.09477)

-   **HER Paper**: Andrychowicz, Marcin, et al. "Hindsight Experience Replay." NIPS, 2017. [Link to paper](https://arxiv.org/abs/1707.01495)

-   **Flower (FL)**: A framework for federated learning. [Flower Website](https://flower.dev/)

-   **Gymnasium-Robotics**: A collection of robotics simulation environments for Reinforcement Learning. [Gymnasium-Robotics Website](https://robotics.farama.org/)

-   **Mujoco**: A general purpose physics engine. [Mujoco Website](https://mujoco.org/)
