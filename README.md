# Explainability of Deep Reinforcement Learning Algorithms in Robotic Domains by using Layer-wise Relevance Propagation

## Environments
Our modified versions of robotic environments are under the `CustomGymEnvs` directory.
In this directory, there is a `changed_envs` directory which contains the new `FetchReach-v1`
environment called `FetchReach-v2` with changed action-space. The actions in the updated environment are torques (rather than
the x, y, and z velocity of the end-effector). Under the `envs` directory, there are 
original environments and environments with occluded entities. Under the `faulty_envs`, there
are environments with blocked joints. Under the `graph_envs`, there are environments with graph
representation of the robots.

## Graph Representation 
For parsing the robot's `xml` model and converting the representation into a graph, the 
`RobotGraphModel` package has been developed. Under this directory, there is a `model_parser.py`
file which parses the `xml` model of the environment. The `robot_graph.py` first parses the model
of the robot, identifying the nodes (body in the `xml`) and edges (joint in the `xml`) of the robot.
Two nested `<body>`'s are connected to each other through a `<joint>` that is defined in the inner body.
For each environment, we have developed a class specific to that environment that has inherited from 
the `RobotGraph` class withing the `robot_graph.py` file. Each of these subclasses define the
set of node and edge features for a specific environment. Each of these subclasses are used
by the OpenAI Gym wrappers under the `CustomGymEnvs/graph_envs`. 

## Algorithm
Our algorithm is [Soft Actor-Critic](https://arxiv.org/abs/1812.05905). The one with graph representation is
under `Graph_SAC` and the original one with fully-connected network is under `SAC`. For using
Graph Neural Network architecture, we use the implementation of 
[torchgraph](https://github.com/baldassarreFe/torchgraphs.git) developed for the paper: 
[Explainability Techniques for Graph Convolutional Networks](https://arxiv.org/abs/1905.13686).
For the LRP implementation, we use [this repository](https://github.com/baldassarreFe/graph-network-explainability) developed for the same paper.
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
$ python $MAIN_FILE --env-name {ENV-NAME} --exp-type graph
```
where the `MAIN_FILE` is the absolute path to the `/Controller/graph/main.py` file. For a complete set of arguments, please 
check out the `main.py` file. The `ENV-NAME` can be the following names:
- `FetchReach-v2`
- `Walker2d-v2`
- `HalfCheetah-v2`
- `Hopper-v2`

After training the agent using graph networks, the [Layer-wise Relevance Propagation (LRP)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140)
is applied to highlight the contribution of each part of the robot to the decision making. The data
for experiments are saved under `./Data/{ENV-NAME}/graph`. 

After the convergence of the policy, the LRP is applied to the learned policy to calculate the relevance scores given 
by each action to each entity across time-steps. To run LRP for the `ENV-NAME` environment, 
run the following:
```
$ python $EVALUATE --env-name {ENV-NAME} --exp-type graph
```
where `EVALUATE` is the absolute path to the `./Evaluate/evaluate.py` file. The result of 
running this file would be stored under `./Data/{ENV-NAME}/graph/edge_relevance.pkl` 
and `./Data/{ENV-NAME}/graph/global_relevance.pkl`, which contains the relevance scores given
to edge and global units of the input graph, respectively.

### Second Phase
In this phase, the results of the first phase are evaluated by either the following experiments:
1) **Occluding** the entity's features in the observation space, which validates its relevance score.
2) **Blocking** the joint which validates the importance of each joint in the action space. 

In each of the above, based on the amount of drop in their performance, their relevance scores are validated.
For more information, please refer to the paper. 
For all the following commands, `$MAIN_FILE` is the absolute path to the 
`./Controller/basic/main.py` file.
For running experiments in the standard setting, just run the following:
```
$ python $MAIN_FILE --env-name {ENV-NAME} --exp-type standard
```
To run experiments for the **occlusion** case, use the following command:
```
$ python $MAIN_FILE --env-name {ENV-NAME} --exp-type {ENTITY-NAME}
```
where `ENTITY-NAME` is the name of the entity we want to occlude. For each environment, the list
of the `ENTITY-NAME`s' are appeared in the following:
- `FetchReach-v2`
  - goal
  - shoulder_pan_joint
  - shoulder_lift_joint
  - upperarm_roll_joint
  - wrist_flex_joint
  - forearm_roll_joint
  - wrist_roll_joint
  - elbow_flex_joint
- `Walker2d-v2`
  - torso
  - foot_joint
  - leg_joint
  - thigh_joint
  - foot_left_joint
  - leg_left_joint
  - thigh_left_joint
- `HalfCheetah-v2`
  - torso
  - bfoot
  - bshin
  - bthigh
  - ffoot
  - fshin
  - fthigh
- `Hopper-v2`
  - torso
  - foot_joint
  - leg_joint
  - thigh_joint

For running experiments for the **blockage** case, use the following command:
```
$ python $MAIN_FILE --env-name {BROKEN-ENV-NAME} --exp-type {JOINT-NAME}
```
where `BROKEN-ENV-NAME` is the name of the environment with broken joint, as appeared in the
following list:
- `FetchReachBroken-v2`
- `Walker2dBroken-v2`
- `HalfCheetahBroken-v2`
- `HopperBroken-v2`

and `JOINT-NAME` is the name of the joint we want to block. For each environment, 
the list of the `JOINT-NAME`s' are appeared in the following:
- `FetchReachBroken-v2`
  - shoulder_pan_joint
  - shoulder_lift_joint
  - upperarm_roll_joint
  - wrist_flex_joint
  - forearm_roll_joint
  - wrist_roll_joint
  - elbow_flex_joint
- `Walker2dBroken-v2`
  - foot_joint
  - leg_joint
  - thigh_joint
  - foot_left_joint
  - leg_left_joint
  - thigh_left_joint
- `HalfCheetahBroken-v2`
  - bfoot
  - bshin
  - bthigh
  - ffoot
  - fshin
  - fthigh
- `HopperBroken-v2`
  - foot_joint
  - leg_joint
  - thigh_joint

Note that these experiments use the original SAC algorithm with fully-connected networks
under the `SAC` directory. For each environment, the resulting data is stored under the following
directories:
- For the occlusion case: `./Data/{ENV-NAME}/{ENTITY-NAME}`
- For the blockage case: `./Data/{BROKEN-ENV-NAME}/{JOINT-NAME}`


### Plots
To plot the results of the experiments, run the following code:
```
$ python $PLOT --env-name {ENV-NAME}
```
where `$PLOT` is the absolute path to the `./Plots/plot.py` file.
The result would be stored under `./Result/{ENV-NAME}.jpg`.
