#!/usr/bin/env bash
set -euo pipefail

container_name="${STAGE_CONTAINER:-css-rl-stage}"
container_workspace="/workspace"

docker exec "${container_name}" bash -lc "
  set -e
  source /opt/ros/noetic/setup.bash
  if [ -f /home/uav/anaconda3/etc/profile.d/conda.sh ]; then
    source /home/uav/anaconda3/etc/profile.d/conda.sh
    conda activate formation
  fi
  python3 -c \"import numpy, scipy, torch; print('Python environment OK:', numpy.__version__, scipy.__version__, torch.__version__)\"
  cd ${container_workspace}
  if [ ! -e src/CMakeLists.txt ]; then
    cd src
    catkin_init_workspace
    cd ..
  fi
  python_executable=\$(command -v python3)
  export ROS_PARALLEL_JOBS='-j4 -l4'
  catkin_make \
    --build build_dense \
    --force-cmake \
    --only-pkg-with-deps distributed_rl_formation stage_ros_add_pose_and_crash \
    -DCATKIN_DEVEL_PREFIX=${container_workspace}/devel_dense \
    -DPYTHON_EXECUTABLE=\${python_executable} \
    -DCMAKE_BUILD_TYPE=Release \
    -j4 -l4
"
