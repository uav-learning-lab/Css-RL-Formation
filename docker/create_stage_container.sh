#!/usr/bin/env bash
set -euo pipefail

container_name="${STAGE_CONTAINER:-css-rl-stage}"
image_name="${STAGE_IMAGE:-css-rl-formation:noetic-cpu}"
cpu_limit="${STAGE_CPUS:-8}"
memory_limit="${STAGE_MEMORY:-24g}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_dir="$(cd "${script_dir}/.." && pwd)"
host_workspace="${repository_dir}/ros_ws"
container_workspace="/workspace"

if docker container inspect "${container_name}" >/dev/null 2>&1; then
  echo "Container ${container_name} already exists."
  echo "Start it with: docker start ${container_name}"
  exit 0
fi

docker_args=(
  run -dit
  --name "${container_name}"
  --network host
  --cpus "${cpu_limit}"
  --memory "${memory_limit}"
  --memory-swap "${memory_limit}"
  --shm-size 4g
  -v "${host_workspace}:${container_workspace}"
)

if [ -n "${STAGE_GPUS:-}" ]; then
  docker_args+=(--gpus "${STAGE_GPUS}")
fi

if [ "${STAGE_X11:-false}" = "true" ]; then
  docker_args+=(
    -e "DISPLAY=${DISPLAY:-:0}"
    -e QT_X11_NO_MITSHM=1
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw
  )
fi

docker "${docker_args[@]}" "${image_name}" bash

echo "Created ${container_name}; workspace mounted at ${container_workspace}."
