# Docker environment

This directory provides a reproducible CPU environment for the Stage validation.
The default image is built from the official Ubuntu 20.04 / ROS Noetic image and
installs Stage dependencies through the ROS package manifests in this repository.

## Requirements on the host

- Linux with Docker Engine 20.10 or newer
- An X11 desktop only when the Stage GUI is required
- At least 12 GB free disk space for the build cache and image
- NVIDIA Container Toolkit only for the optional CUDA configuration

The default image does not require an NVIDIA GPU.

## Build the default CPU image

Run this from the repository root:

```bash
docker build \
  -f docker/Dockerfile \
  -t css-rl-formation:noetic-cpu \
  .
```

The build downloads the official `ros:noetic-ros-base-focal` base, resolves
ROS/Stage dependencies with `rosdep`, and installs CPU PyTorch 2.0.1, NumPy
1.24.4 and SciPy 1.10.1.

## Create and build the Catkin workspace

```bash
./docker/create_stage_container.sh
./docker/build_stage.sh
```

The container mounts `ros_ws/` at `/workspace`.

## Headless validation

```bash
STAGE_GUI=false ./docker/run_stage.sh
```

## Stage GUI

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

## Optional CUDA PyTorch image

```bash
docker build \
  -f docker/Dockerfile \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118 \
  -t css-rl-formation:noetic-cu118 \
  .

STAGE_IMAGE=css-rl-formation:noetic-cu118 \
STAGE_GPUS=all \
./docker/create_stage_container.sh
```

GPU acceleration is optional. The controller automatically falls back to CPU.
