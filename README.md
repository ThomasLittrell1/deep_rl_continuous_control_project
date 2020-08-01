# deep_rl_continuous_control_project

## Project Details

This repo contains the code to solve the Continuous Control Project, which is part of the Udacity Deep Reinforcement Learning nanodegree. The goal of the project is to teach a robotic arm to move to a target location. The details of the MDP are:

- States: The states are 33 dimensional continuous vectors representing physical properties of the arm's movement (e.g. position, velocity, etc.)
- Actions: Each action is a 4 dimensional vector bounded in [-1, 1] representing different forces to apply to differnt parts of the arm
- Rewards: The agent receives a reward of +0.1 for every time step that the arm is in the target location

The goal is to have the agent get an average score of +30 for at least 100 consecutive rounds.

## Getting Started

You will need three things to run the project:
1. The appropriate python packages
2. The `Reacher.app` Unity environment supplied by Udacity. I used the 20 arm version.
3. The additional `python` code provided by Udacity for running the Unity environment

### Installing the python packages

Create a new conda environment (or virtual environment of your choice) and activate
 it. Install the dependencies using `python -m pip install -r requirements.txt`.

### Installing Udacity dependencies

Udacity provides specific helper code to run the project and environment. I have not included their `Reacher.app` file or their `python` folder with additional code for the environment in this repository to protect their IP. Udacity provides specific instructions for installing the Unity environment. You can download the `python` folder of additional dependencies from your Udacity workspace and install it using `python -m pip -q install python` or by running `pip -q install ./python` in the Continuous_Control.ipynb notebook.

## Instructions

To train an agent, create a kernel from your conda environment and run all cells in the Continuous_Control.ipynb Jupyter notebook. If the agent completes the task successfully, the saved weights will be put into checkpoint_policy.pth and checkpoint_qnetwork.pth files.

All code for the agent, the networks, and the training code is kept in the src directory.
