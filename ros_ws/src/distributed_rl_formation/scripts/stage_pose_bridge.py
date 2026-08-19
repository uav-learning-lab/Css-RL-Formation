#!/usr/bin/env python3
"""Bridge Stage odometry into the historical MAVROS-shaped algorithm topics."""

import os

import rospy
import yaml
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry


class StagePoseBridge:
    def __init__(self):
        rospy.init_node("dense_stage_pose_bridge")
        package_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        config_file = rospy.get_param(
            "~config_file",
            os.path.join(package_dir, "config", "dense_experiment.yaml"),
        )
        with open(config_file, "r") as stream:
            config = yaml.safe_load(stream)

        self.bias = config["formation_config"]["bias"]
        self.pose_pubs = []
        self.body_velocity_pubs = []
        self.local_velocity_pubs = []

        for robot_id, _ in enumerate(self.bias):
            iris_namespace = "/iris_{}".format(robot_id)
            robot_namespace = "/robot_{}".format(robot_id)
            self.pose_pubs.append(
                rospy.Publisher(
                    iris_namespace + "/mavros/local_position/pose",
                    PoseStamped,
                    queue_size=5,
                )
            )
            self.body_velocity_pubs.append(
                rospy.Publisher(
                    iris_namespace + "/mavros/local_position/velocity_body",
                    TwistStamped,
                    queue_size=5,
                )
            )
            self.local_velocity_pubs.append(
                rospy.Publisher(
                    iris_namespace + "/mavros/local_position/velocity_local",
                    TwistStamped,
                    queue_size=5,
                )
            )
            rospy.Subscriber(
                robot_namespace + "/base_pose_ground_truth",
                Odometry,
                self.ground_truth_callback,
                callback_args=robot_id,
                queue_size=5,
            )
            rospy.Subscriber(
                robot_namespace + "/odom",
                Odometry,
                self.odometry_callback,
                callback_args=robot_id,
                queue_size=5,
            )

    def ground_truth_callback(self, msg, robot_id):
        # The old controller expects per-vehicle local coordinates, then adds
        # its configured initial bias to reconstruct world coordinates.
        pose = PoseStamped()
        pose.header = msg.header
        pose.header.frame_id = "map"
        pose.pose = msg.pose.pose
        pose.pose.position.x -= self.bias[robot_id][0]
        pose.pose.position.y -= self.bias[robot_id][1]
        self.pose_pubs[robot_id].publish(pose)

    def odometry_callback(self, msg, robot_id):
        # stage_ros odometry twist is expressed in the robot frame, matching
        # mavros/local_position/velocity_body used by the controller.
        velocity = TwistStamped()
        velocity.header = msg.header
        velocity.twist = msg.twist.twist
        self.body_velocity_pubs[robot_id].publish(velocity)
        self.local_velocity_pubs[robot_id].publish(velocity)


if __name__ == "__main__":
    StagePoseBridge()
    rospy.spin()
