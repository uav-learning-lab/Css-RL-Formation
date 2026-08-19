#!/usr/bin/env python3
"""Publish the fixed centre target used by the dense Stage scenario."""

import time

import rospy
from geometry_msgs.msg import Point, PoseStamped
from std_msgs.msg import String


def main():
    rospy.init_node("dense_experiment_goal")
    goal_x = rospy.get_param("~x", 44.0)
    goal_y = rospy.get_param("~y", 1.0)
    delay = rospy.get_param("~delay", 2.0)

    nav_goal_pub = rospy.Publisher(
        "/move_base_simple/goal", PoseStamped, queue_size=1, latch=True
    )
    leader_goal_pub = rospy.Publisher("goal", Point, queue_size=1, latch=True)
    command_pub = rospy.Publisher("/gcs_cmd", String, queue_size=1, latch=True)

    # Wall time is intentional: it also works before Gazebo starts /clock.
    time.sleep(max(0.0, delay))

    nav_goal = PoseStamped()
    nav_goal.header.stamp = rospy.Time.now()
    nav_goal.header.frame_id = "map"
    nav_goal.pose.position.x = goal_x
    nav_goal.pose.position.y = goal_y
    nav_goal.pose.orientation.w = 1.0

    leader_goal = Point(x=goal_x, y=goal_y, z=0.0)
    nav_goal_pub.publish(nav_goal)
    leader_goal_pub.publish(leader_goal)
    command_pub.publish(String(data="AUTO.CANC"))
    rospy.loginfo("Dense experiment target published at (%.3f, %.3f)", goal_x, goal_y)
    rospy.spin()


if __name__ == "__main__":
    main()
