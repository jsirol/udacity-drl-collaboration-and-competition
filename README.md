# udacity-drl-collaboration-and-competition
Train two agents to play tennis using Deep Deterministic Policy Gradients.

# Project details
The task in this project was to train an agent (or agents) to solve an environment where 2 agents play a tennis-like game and try to keep the game running by hitting the ball over the net. The task is episodic and considered solved when the average score received 100 consecutive episodes exceeds +0.5. The score of an episode is defined to be the maximum score over the two agents on the episode.

In the provided solution, the agent is trained for roughly 1400 episodes and it gains an average reward of 
+1.74 over 100 consecutive episodes around 1300 episodes mark. The environment is considered first solved after 1158 episodes.

## Trained agent in action
![Trained agent](www/trained_ddpg.gif)

## Rewards
The agent receives a reward of `+0.1` for each time the ball is hit over the net and `-0.01` when the agent lets the ball hit the ground or hits the ball out of bounds.

## State space
The continuous state space has 8 dimensions. The full vector observation consist of 3 stacked states, so the total size of observation space is 24.

## Actions
Each action (of a single agent) is a vector with 2 numbers, that correspond to movement (towards or away from the net) and jumping. Each entry in the vector is a float between -1 and 1.


# Getting started
This code was developed and tested on Python 3.6.4 and PyTorch 0.4.1. The included environment executable is for Windows (x64). The links for Linux/Mac environments are provided below (provided by Udacity, might not work in the future):

[Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)

[MacOS](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)

## Installing dependencies
A working Python installation is required. An easy way is to install using [Anaconda](https://www.anaconda.com/download/). To install a specific Python version using Anaconda, see [link](http://docs.anaconda.com/anaconda/faq/#how-do-i-get-the-latest-anaconda-with-python-3-5).

Additionally, PyTorch needs to be installed by first running (if you installed Python with Anaconda) 

`conda install pytorch -c pytorch` 

and then

`pip3 install torchvision` 

from the command line. See [PyTorch website](https://pytorch.org/) for more instructions and details.

Finally, Unity ML-Agents version 0.4.0b needs to be installed from [here](https://github.com/Unity-Technologies/ml-agents/releases/tag/0.4.0b). To install, download the .zip archive and unzip, navigate to the `python` directory, and run 

`pip3 install .`. 

For additional help see the [installation guide](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) (Note! the link is for new version of the ML-Agents where some of the syntax has changed from the version used in this project).

# Instructions to run the code
First clone or download this repository. 

Easiest way to replicate the results is to launch a jupyter notebook (contained with Anaconda installation of Python) by running 

`jupyter notebook` 

from the command line in the project root. Then, open the [Report.ipynb](Report.ipynb) file from the jupyter webpage UI and follow the instructions there.

If you want to explore the code further, it is organized as follows under the `ddpg/` folder:

1. `agent.py` contains code for the agent.
2. `model.py` contains the neural network code that is used by the agent.
3. `ddpg_trainer.py` contains code that is used to train the agent.
4. Saved weights for the Actor and Critic networks can be found under the `saved_models/` directory in the project root (to see how to load the trained model, refer to the [Report](Report.ipynb)). Weights are provided for the best solution (`best_[actor/critic].pth`) and when the environment was first solved (`[actor/critic]_solved.pth`).
