ARG ROS_BASE_IMAGE=ros:noetic-ros-base-focal
FROM ${ROS_BASE_IMAGE}

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      cmake \
      git \
      libgl1-mesa-dri \
      libgl1-mesa-glx \
      pkg-config \
      python3-dev \
      python3-pip \
      python3-rosdep \
      python3-tk \
    && rm -rf /var/lib/apt/lists/*

COPY ros_ws/src /tmp/css_rl_ros_ws/src
RUN source /opt/ros/noetic/setup.bash \
    && rosdep update --rosdistro noetic \
    && apt-get update \
    && rosdep install \
         --from-paths /tmp/css_rl_ros_ws/src \
         --ignore-src \
         --rosdistro noetic \
         -r -y \
    && rm -rf /tmp/css_rl_ros_ws /var/lib/apt/lists/*

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=2.0.1
RUN python3 -m pip install \
      --index-url "${TORCH_INDEX_URL}" \
      "torch==${TORCH_VERSION}" \
    && python3 -m pip install \
      matplotlib==3.7.5 \
      numpy==1.24.4 \
      PyYAML==6.0.1 \
      rospkg==1.5.1 \
      scipy==1.10.1

WORKDIR /workspace
CMD ["bash"]
