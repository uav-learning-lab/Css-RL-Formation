cmake_minimum_required(VERSION 3.0.2)
project(distributed_rl_formation)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  message_generation
  nav_msgs
  rospy
  sensor_msgs
  std_msgs
)

add_message_files(
  FILES
  AuctionInfo.msg
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs message_runtime nav_msgs rospy sensor_msgs std_msgs
)

catkin_install_python(PROGRAMS
  scripts/Consistent_formations.py
  scripts/dis_align.py
  scripts/plot_control_dominance.py
  scripts/publish_experiment_goal.py
  scripts/stage_pose_bridge.py
  src/dis_formation_control.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY
  config
  launch
  worlds
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY
  src/model
  src/policy
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/src
)
