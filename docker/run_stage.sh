#!/usr/bin/env bash
set -euo pipefail

container_name="${STAGE_CONTAINER:-css-rl-stage}"
container_workspace="/workspace"
gui="${STAGE_GUI:-false}"

docker exec -it "${container_name}" bash -lc "
  set -e
  source /opt/ros/noetic/setup.bash
  if [ -f /home/uav/anaconda3/etc/profile.d/conda.sh ]; then
    source /home/uav/anaconda3/etc/profile.d/conda.sh
    conda activate formation
  fi
  source ${container_workspace}/devel_dense/setup.bash
  roslaunch distributed_rl_formation stage_dense_reproduction.launch \
    gui:=${gui} \
    results_root:=${container_workspace}/results_dense
"
