# Css-RL-Formation

This repository contains the official implementation of our paper, which addresses the coordinated formation maintenance and obstacle avoidance problem for multi-UAV systems in unknown environments. We propose an adaptive formation control approach named Css-RL-Formation, combining deep reinforcement learning (DRL) and consensus theory. The developed framework adopts a distributed position-assignment module for dynamic target allocation, and constructs a hybrid controller integrating a DRL-based control module and a consensus control module. A threat-aware nonlinear fusion mechanism is further designed to merge control outputs according to real-time obstacle clearance measurements.

This repository provides the ROS Stage validation environment, controller
source code, pretrained checkpoint, experiment launcher, and data recorder used to reproduce experiments. The demonstration video is publicly available at https://www.bilibili.com/video/BV1Kv876MESi/

## Reproduced experiment

| Item | Configuration |
| --- | --- |
| Simulator | Stage 4.3 with ROS Noetic |
| Robots | 4 differential-drive robots |
| Initial formation | top, right, bottom, left |
| Formation-centre target | `(44, 1)` m |
| Laser | 270 degrees, 512 samples, 5 m range |
| Scene | 33 static obstacles, 0.85 x 0.85 m each |
| Robot footprint | 0.40 x 0.32 m |
| Pretrained checkpoint | `affine1222_4500` |

Initial world-frame positions are `(0, 4.242)`, `(4.242, 0)`, `(0, -4.242)`,
and `(-4.242, 0)` metres.

## Repository layout

```text
Css-RL-Formation/
├── docker/                           # Reproducible environment and scripts
├── docs/                             # Stage experiment details
└── ros_ws/src/
    ├── distributed_rl_formation/     # Control, inference, logging, and plots
    └── stage_ros-add_pose_and_crash/ # Stage ROS interface for this experiment
```

## Requirements

- Linux with Docker Engine 20.10 or newer
- At least 12 GB of free disk space while building the image
- X11 only when the Stage GUI is required
- NVIDIA Container Toolkit only for the optional CUDA configuration

The default configuration runs on CPU and does not require an NVIDIA GPU.

## Quick start with Docker

Clone the repository and build the CPU image:

```bash
git clone https://github.com/uav-learning-lab/Css-RL-Formation.git
cd Css-RL-Formation

docker build \
  -f docker/Dockerfile \
  -t css-rl-formation:noetic-cpu \
  .
```

Create a reusable container and compile the Catkin workspace:

```bash
./docker/create_stage_container.sh
./docker/build_stage.sh
```

Run the experiment without a GUI:

```bash
STAGE_GUI=false ./docker/run_stage.sh
```

Press `Ctrl+C` to stop the experiment.

## Stage GUI

The GUI container must be created with the X11 socket mounted:

```bash
xhost +local:docker

STAGE_CONTAINER=css-rl-stage-gui \
STAGE_X11=true \
./docker/create_stage_container.sh

STAGE_CONTAINER=css-rl-stage-gui \
./docker/build_stage.sh

STAGE_CONTAINER=css-rl-stage-gui \
STAGE_GUI=true \
./docker/run_stage.sh

xhost -local:docker
```

## Runtime checks

Open another terminal while the experiment is running:

```bash
docker exec -it css-rl-stage bash
source /opt/ros/noetic/setup.bash
source /workspace/devel_dense/setup.bash

rosnode list
rostopic hz /iris_0/scan
rostopic hz /iris_0/coll_avoid/cmd_vel_flu
rostopic echo -n 1 /robot_0/base_pose_ground_truth
```

A complete run starts 13 ROS nodes: Stage, the pose bridge, four assignment
nodes, the consistency controller, four fused-control nodes, and the goal
publisher.

## Experiment outputs

Results are written to:

```text
ros_ws/results_dense/Cons-RL/RL_Formation/Dense/<run_id>/
```

Each run contains four trajectory files, a formation-similarity file, and one
fusion log per robot:

```text
fusion_data/uav_0.txt
fusion_data/uav_1.txt
fusion_data/uav_2.txt
fusion_data/uav_3.txt
```

Every data row has exactly six whitespace-separated fields:

```text
time_s x y rl_velocity_(vx,wz) consistency_velocity_(vx,wz) alpha
```

## Reproducibility information

Checkpoint path:

```text
ros_ws/src/distributed_rl_formation/src/policy/affine1222_4500
```

Checkpoint SHA-256:

```text
58acabe58a4928c26cac2c06adbf507145e7f323a779fa968e4e9d42aca63560
```

The public CPU environment uses Python 3.8, PyTorch 2.0.1, NumPy 1.24.4,
SciPy 1.10.1, ROS Noetic, and Stage 4.3. 
