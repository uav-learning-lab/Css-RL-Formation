#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import numpy as np
import rospy
import yaml
from geometry_msgs.msg import Twist,TwistStamped, PoseStamped, Point
import math
import threading
import os
M_PI = 3.14159265

class Formation_Control:
    def __init__(self):
        # 初始化配置
        rospy.init_node("control_node")
        package_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        config_file = rospy.get_param(
            "~config_file",
            os.path.join(package_dir, "config", "dense_experiment_virtual.yaml"),
        )
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        formation_num = config['formation_config']['nums']
        bias = config['formation_config']['bias']
        self.num_env = formation_num
        self.bias = np.array(bias)
        # 控制参数初始化
        self.dt = 0.05
        self.d_bias = np.array(self.bias, dtype=np.float32)
        self.goal = np.zeros(2)
        self.has_goal = False
        self.rotation_angle = 0.0
        self.distance = 0

        self.all_velocity = [np.zeros(2, dtype=float) for _ in range(self.num_env)]
        self.all_curr_pose = [np.zeros(2, dtype=float) for _ in range(self.num_env)]
        self.all_siny_cosp = [np.zeros(1, dtype=float) for _ in range(self.num_env)]
        self.all_cosy_cosp = [np.zeros(1, dtype=float) for _ in range(self.num_env)]
        self.all_yaw = [np.zeros(1, dtype=float) for _ in range(self.num_env)]
        self.d_bias_update = [np.zeros(2, dtype=float) for _ in range(self.num_env)]
        self.uav_goal_pose = [np.zeros(2, dtype=float) for _ in range(self.num_env)]
        self.d_bias_timer = rospy.Timer(rospy.Duration(0.1), self.update_d_bias)  # 10Hz更新频率
        self.lock = threading.Lock()  # 创建锁

        # 初始化虚拟leader状态
        self.u0 = np.array([self.r0(0), self.d0(0)])
        self.w0 = np.zeros(1)
        self.v0 = np.zeros(1)
        self.mean_pose_curr = np.zeros(2)
        self.mean_pose_goal = np.zeros(2)
        self.virtual_leader_pose = np.zeros(2)
        self.virtual_leader_yaw = np.zeros(1)
        self.virtual_leader_velocity = np.zeros(1)

        self.u1 = np.zeros(2)
        self.w1 = np.zeros(1)
        self.r1 = np.zeros(1)
        self.z1 = np.zeros(2)
        self.v1 = 0.0

        self.u2 = np.zeros(2)
        self.w2 = np.zeros(1)
        self.r2 = np.zeros(1)
        self.z2 = np.zeros(2)
        self.v2 = 0.0

        self.u3 = np.zeros(2)
        self.w3 = np.zeros(1)
        self.r3 = np.zeros(1)
        self.z3 = np.zeros(2)
        self.v3 = 0.0

        self.u4 = np.zeros(2)
        self.w4 = np.zeros(1)
        self.r4 = np.zeros(1)
        self.z4 = np.zeros(2)
        self.v4 = 0.0

        self.move_velocity_pub = []

        self.goal_sub = rospy.Subscriber('goal', Point, self.goal_callback, queue_size=5)  # 订阅目标点


        for i in range(self.num_env):
            self.local_velocity_sub = rospy.Subscriber("iris_%d/mavros/local_position/velocity_body" % (i),
                                                       TwistStamped, self.all_velocity_callback, (i), queue_size=1)
            self.position_sub = rospy.Subscriber("iris_%d/mavros/local_position/pose" % (i), PoseStamped,
                                                 self.all_pos_callback, (i), queue_size=1)
            self.move_velocity_pub.append(rospy.Publisher("/iris_%d/Consistency_speed" % (i), Twist, queue_size=1))

            self.assignments_sub = rospy.Subscriber('/uav_%d/assignment' % (i),
                             Point, self.local_goal_callback, (i), queue_size=1)

    def goal_callback(self, msg):
        """目标点回调函数"""
        if not self.has_goal:  # 首次接收目标时初始化
            self.goal = [msg.x, msg.y]
        else:  # 后续更新保持列表可变性
            self.goal[0] = msg.x
            self.goal[1] = msg.y
        self.has_goal = True
        # rospy.loginfo("New goal received: %s" % self.goal)

    def update_d_bias(self, event):
        """定时更新编队偏移量的方法"""
        with self.lock:
            for uav_id in range(self.num_env):
                center_align_x = self.mean_pose_goal[0] - self.mean_pose_curr[0]
                center_align_y = self.mean_pose_goal[1] - self.mean_pose_curr[1]
                # 计算当前目标与平均位置的偏移量
                offset_x = self.uav_goal_pose[uav_id][0] - center_align_x - self.mean_pose_curr[0]
                offset_y = self.uav_goal_pose[uav_id][1] - center_align_y - self.mean_pose_curr[1]
                # 应用低通滤波平滑过渡
                self.d_bias_update[uav_id][0] = offset_x
                self.d_bias_update[uav_id][1] = offset_y
            # 更新全局偏移量配置
            self.d_bias[-self.num_env:] = np.stack(self.d_bias_update)
            # print(self.d_bias)

    def local_goal_callback(self, msg, uav_id):
        self.uav_goal_pose[uav_id] = [msg.x, msg.y]
        sum_x = 0
        sum_y = 0
        for uav_id in range(self.num_env):
            sum_x += self.uav_goal_pose[uav_id][0]
            sum_y += self.uav_goal_pose[uav_id][1]
        self.mean_pose_goal = np.array([
            sum_x / self.num_env,
            sum_y / self.num_env])

    def all_velocity_callback(self, vel, uav_id):
        self.all_velocity[uav_id][0] = vel.twist.linear.x
        self.all_velocity[uav_id][1] = vel.twist.angular.z

    def all_pos_callback(self, loc, uav_id):
        """位置回调函数"""
        self.all_curr_pose[uav_id][0] = loc.pose.position.x + self.bias[uav_id + 1][0]
        self.all_curr_pose[uav_id][1] = loc.pose.position.y + self.bias[uav_id + 1][1]
        self.all_siny_cosp[uav_id] = 2 * (
                loc.pose.orientation.w * loc.pose.orientation.z + loc.pose.orientation.x * loc.pose.orientation.y)
        self.all_cosy_cosp[uav_id] = 1 - 2 * (
                loc.pose.orientation.y * loc.pose.orientation.y + loc.pose.orientation.z * loc.pose.orientation.z)
        self.all_yaw[uav_id][0] = math.atan2(self.all_siny_cosp[uav_id], self.all_cosy_cosp[uav_id])

        sum_x = 0
        sum_y = 0
        for uav_id in range(self.num_env):
            sum_x += self.all_curr_pose[uav_id][0]
            sum_y += self.all_curr_pose[uav_id][1]
        self.mean_pose_curr = np.array([
            sum_x / self.num_env,
            sum_y / self.num_env])

    def control_vel(self, action, ID):
        """批量更新多个UAV的控制命令"""
        v = max(min(action[0], 10.0), 0.0)
        w = max(min(action[1], 1.0), -1.0)
        move_cmd = Twist()
        move_cmd.linear.x = v
        move_cmd.linear.y = 0.
        move_cmd.linear.z = 0.
        move_cmd.angular.x = 0.
        move_cmd.angular.y = 0.
        move_cmd.angular.z = w
        self.move_velocity_pub[ID].publish(move_cmd)

    def update_virtual_leader(self):

        target_vector = self.goal - self.mean_pose_curr
        self.virtual_leader_pose = self.mean_pose_curr + \
                                0.1 * (target_vector / np.linalg.norm(target_vector) + 1e-6 )

        self.virtual_leader_yaw = np.arctan2(target_vector[1], target_vector[0])
        self.virtual_leader_velocity = 1.5

    def reach_goal(self, curr_pose, goal, goal_radius):
        # 增加目标有效性验证
        if goal is None:
            rospy.logwarn("Goal is not initialized")
            return False  # 默认未到达目标

        distance_to_goal = np.sqrt(
            (curr_pose[0] - goal[0]) ** 2 +
            (curr_pose[1] - goal[1]) ** 2
        )
        return distance_to_goal < goal_radius

    def run(self):
        rate = rospy.Rate(1 / self.dt)
        i = 1
        while not rospy.is_shutdown():
            while not self.has_goal or self.goal is None:
                rospy.logwarn_throttle(5, "Waiting for initial goal...")
            self.update_virtual_leader()
            # 获取误差项
            wx, wy = self.w_error(1, i * self.dt)

            self.z1[0] = ((self.all_curr_pose[0][0] - self.virtual_leader_pose[0] -
                           (self.d_bias[1][0] - self.d_bias[0][0])) +
                          (self.all_curr_pose[0][0] - self.all_curr_pose[3][0] -
                           (self.d_bias[1][0] - self.d_bias[4][0])) + 2 * wx) / 2
            self.z1[1] = ((self.all_curr_pose[0][1] - self.virtual_leader_pose[1] - (
                        self.d_bias[1][1] - self.d_bias[0][1])) +
                          (self.all_curr_pose[0][1] - self.all_curr_pose[3][1] - (
                                  self.d_bias[1][1] - self.d_bias[4][1])) + 2 * wy) / 2
            # 计算u1
            self.u1 = self.u0 - 2 * (np.array([self.all_velocity[0][0] * np.cos(self.all_yaw[0][0]),
                                               self.all_velocity[0][0] * np.sin(self.all_yaw[0][0])])
                                     - np.array([self.virtual_leader_velocity * np.cos(self.virtual_leader_yaw),
                                                 self.virtual_leader_velocity * np.sin(self.virtual_leader_yaw)])
                                     - self.faixy(self.z1))

            # 更新控制状态
            self.w1, self.v1 = self.Update_Control(
                self.all_yaw[0][0], self.all_velocity[0][0], self.u1, self.dt)

            self.z2[0] = ((self.all_curr_pose[1][0] - self.all_curr_pose[0][0] - (
                        self.d_bias[2][0] - self.d_bias[1][0])) +
                          (self.all_curr_pose[1][0] - self.all_curr_pose[2][0] - (
                                  self.d_bias[2][0] - self.d_bias[3][0])) + 2 * wx) / 2
            self.z2[1] = ((self.all_curr_pose[1][1] - self.all_curr_pose[0][1] - (
                        self.d_bias[2][1] - self.d_bias[1][1])) +
                          (self.all_curr_pose[1][1] - self.all_curr_pose[2][1] - (
                                  self.d_bias[2][1] - self.d_bias[3][1])) + 2 * wy) / 2
            # 计算u2
            self.u2 = self.u0 - 2 * (np.array([self.all_velocity[1][0] * np.cos(self.all_yaw[1][0]),
                                               self.all_velocity[1][0] * np.sin(self.all_yaw[1][0])])
                                     - np.array([self.virtual_leader_velocity * np.cos(self.virtual_leader_yaw),
                                                 self.virtual_leader_velocity * np.sin(self.virtual_leader_yaw)])
                                     - self.faixy(self.z2))

            # 更新控制状态
            self.w2, self.v2 = self.Update_Control(
                self.all_yaw[1][0], self.all_velocity[1][0], self.u2, self.dt
            )


            # 计算z3
            self.z3[0] = ((self.all_curr_pose[2][0] - self.all_curr_pose[1][0] - (
                        self.d_bias[3][0] - self.d_bias[2][0])) +
                          (self.all_curr_pose[2][0] - self.all_curr_pose[3][0] - (
                                      self.d_bias[3][0] - self.d_bias[4][0])) + 2 * wx) / 2
            self.z3[1] = ((self.all_curr_pose[2][1] - self.all_curr_pose[1][1] - (
                        self.d_bias[3][1] - self.d_bias[2][1])) +
                          (self.all_curr_pose[2][1] - self.all_curr_pose[3][1] - (
                                      self.d_bias[3][1] - self.d_bias[4][1])) + 2 * wy) / 2
            # 计算u3
            self.u3 = self.u0 - 2 * (np.array([self.all_velocity[2][0] * np.cos(self.all_yaw[2][0]),
                                               self.all_velocity[2][0] * np.sin(self.all_yaw[2][0])])
                                     - np.array([self.virtual_leader_velocity * np.cos(self.virtual_leader_yaw),
                                                 self.virtual_leader_velocity * np.sin(self.virtual_leader_yaw)])
                                     - self.faixy(self.z3))

            # 更新控制状态
            self.w3, self.v3 = self.Update_Control(
                self.all_yaw[2][0], self.all_velocity[2][0], self.u3, self.dt
            )

            # 计算z4
            self.z4[0] = ((self.all_curr_pose[3][0] - self.all_curr_pose[2][0] - (
                        self.d_bias[4][0] - self.d_bias[3][0])) +
                          (self.all_curr_pose[3][0] - self.virtual_leader_pose[0] - (
                                      self.d_bias[4][0] - self.d_bias[0][0])) + 2 * wx) / 2
            self.z4[1] = ((self.all_curr_pose[3][1] - self.all_curr_pose[2][1] - (
                        self.d_bias[4][1] - self.d_bias[3][1])) +
                          (self.all_curr_pose[3][1] - self.virtual_leader_pose[1] - (
                                      self.d_bias[4][1] - self.d_bias[0][1])) + 2 * wy) / 2
            # 计算u4
            self.u4 = self.u0 - 2 * (np.array([self.all_velocity[3][0] * np.cos(self.all_yaw[3][0]),
                                               self.all_velocity[3][0] * np.sin(self.all_yaw[3][0])])
                                     - np.array([self.virtual_leader_velocity * np.cos(self.virtual_leader_yaw),
                                                 self.virtual_leader_velocity * np.sin(self.virtual_leader_yaw)])
                                     - self.faixy(self.z4))

            # 更新控制状态
            self.w4, self.v4 = self.Update_Control(
                self.all_yaw[3][0], self.all_velocity[3][0], self.u4, self.dt
            )

            self.control_vel([self.v1, self.w1], 0)
            self.control_vel([self.v2, self.w2], 1)
            self.control_vel([self.v3, self.w3], 2)
            self.control_vel([self.v4, self.w4], 3)

            i = i + 1

            while self.reach_goal(self.virtual_leader_pose, self.goal, 0.5):
                self.control_vel([0.0,0.0], 0)
                self.control_vel([0.0,0.0], 1)
                self.control_vel([0.0,0.0], 2)
                self.control_vel([0.0,0.0], 3)
                rate.sleep()


    def Update_Control(self, th, v, u, dt):
        a, b, c, d = self.TAinv(th, v)
        A = np.array([[a, b], [c, d]])
        r, w = np.dot(A, u)
        v = v + r * dt
        return w, v

    def r0(self, t):
        return 0.1 * np.sin(0.4 * t / 100)

    def d0(self, t):
        return 0.1 * np.cos(0.2 * t / 100)

    def initialize_agents(self, pose, v, theta):
        # Initialize variables
        p = pose
        v = v
        th = theta
        v_ = np.array([v * np.cos(th), v * np.sin(th)])  # velocity components in x and y directions
        z = np.array([0, 0])  # Initial z vector
        u = np.array([0, 0])  # Initial u vector
        r = 0  # Initial rotation (r)
        w = 0  # Initial angular velocity (w)

        return p, v, th, v_, z, u, r, w

    def faiv(self, r):
        return -1.5 * (1 - np.exp(-0.8 * r)) / (1 + np.exp(-0.8 * r))  # 增强响应速度

    def faixy(self, r):
        return -3 * (1 - np.exp(-0.8 * r)) / (1 + np.exp(-0.8 * r))

    def TA(self, th, v):
        # 计算角度和速度的控制矩阵A
        a = np.cos(th)
        b = -v * np.sin(th)
        c = np.sin(th)
        d = v * np.cos(th)

        return a, b, c, d

    def TAinv(self, th, v):

        a = np.cos(th)
        b = np.sin(th)
        c = -np.sin(th) / (v + 1e-6)
        d = np.cos(th) / (v + 1e-6)

        return a, b, c, d

    def w_error(self, i, t):
        # 计算 wx 和 wy 的测量误差
        wx = 0.3 * (np.cos(t + i * np.pi / 6) + np.cos(t / 3 + i * np.pi / 6) +
                    np.cos(t / 5 + i * np.pi / 6) + np.cos(t / 7 + i * np.pi / 6))

        wy = 0.3 * (np.sin(t + i * np.pi / 6) + np.sin(t / 3 + i * np.pi / 6) +
                    np.sin(t / 5 + i * np.pi / 6) + np.sin(t / 7 + i * np.pi / 6))

        return wx, wy


if __name__ == "__main__":
    controller = Formation_Control()
    try:
        controller.run()
    except rospy.ROSInterruptException:
        pass
