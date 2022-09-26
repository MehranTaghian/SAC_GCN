# Explainability of Deep Reinforcement Learning Algorithms in Robotic Domains by using Layer-wise Relevance Propagation

## Environments
Our modified versions of robotic environments are under the `CustomGymEnvs` directory.
In this directory, there is a `changed_envs` directory which contains the new `FetchReach-v1`
environment with changed action-space. The actions in the updated environment are torques (rather than
the x, y, and z velocity of the end-effector). Under the `envs` directory, there are 
original environments and environments with occluded entities. Under the `faulty_envs`, there
are environments with blocked joints. Under the `graph_envs`, there are environments with graph
representation of the robots.

## Graph Representation 

## Algorithm
Our algorithm is [Soft Actor-Critic](https://arxiv.org/abs/1812.05905). The one with graph representation is
under `Graph_SAC` and the original one with fully-connected network is under `SAC`. For using
Graph Neural Network architecture, we use the implementation of 
[torchgraph](https://github.com/baldassarreFe/torchgraphs.git) developed for the paper: 
[Explainability Techniques for Graph Convolutional Networks](https://arxiv.org/abs/1905.13686).

## Installation and Usage Guidelines 
### Setup
The python version is `3.8.10`.
The first step before running the project is to install `MuJoCo 2.1`:
```
$ wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
$ tar -xvf mujoco210-linux-x86_64.tar.gz
$ mv mujoco210 ~/.mujoco/
$ pip3 install -U 'mujoco-py<2.2,>=2.1'
```
Download the project file into the `$HOME/Documents` folder. 
Then add the following lines to the `~/.bashrc` file:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export PYTHONPATH=$PYTHONPATH:$HOME/Documents/SAC_GCN
```
Then install the requirements of the project:
```
$ pip3 install -r requirements.txt
```
## Experiments 
### First Phase 
To run the experiments with graph representation of the robot, run the following command:
```
python $RUN_FILE --env-name {ENV-NAME} --exp-type graph --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 -ef 100 --cuda --seed 0
```
where the `RUN_FILE` is the complete path to the `/Controller/graph/main.py` file. For a complete set of arguments, please 
check out the `main.py` file. The `ENV-NAME` can be the following names:
- `FetchReach-v2`
- `Walker2d-v2`
- `HalfCheetah-v2`
- `Hopper-v2`

After training the agent using graph networks, the [Layer-wise Relevance Propagation (LRP)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140)
is applied to highlight the contribution of each part of the robot to the decision making. The data
for experiments are saved under `./Data/{ENV-NAME}/graph`. 

run evaluation



