#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from __future__ import print_function
import os
import numpy as np
import sys
import copy
import math
import torch
import torch.nn as nn
from torch.optim import Adam
from collections import deque

# Make the bundled policy modules importable both from the source tree and from
# a catkin-generated launcher script.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
try:
    import rospkg
    PACKAGE_DIR = rospkg.RosPack().get_path("distributed_rl_formation")
    SCRIPT_DIR = os.path.join(PACKAGE_DIR, "src")
except Exception:
    pass
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import rospy
from geometry_msgs.msg import Twist, Pose, TwistStamped, PoseStamped, Vector3, Point
from sensor_msgs.msg import LaserScan
from model.net import CNNPolicy
from model.ppo import generate_action_no_sampling
import yaml
import time
from std_msgs.msg import String

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 下面老是报错 shape 不一致

MAX_EPISODES = 5000
LASER_BEAM = 512
LASER_HIST = 3
HORIZON = 200
GAMMA = 0.99
LAMDA = 0.95
BATCH_SIZE = 512
EPOCH = 10
COEFF_ENTROPY = 5e-4
CLIP_VALUE = 0.1
NUM_ENV = 1
OBS_SIZE = 512
ACT_SIZE = 2
LEARNING_RATE = 5e-5
M_PI = 3.14159265


class Controller:
    def __init__(self, type, id):
        self.id = int(id)
        self.type = type
        self.uav_namespace = type + "_" + id
        rospy.init_node(self.uav_namespace + "control_node")

        config_file = rospy.get_param(
            "~config_file",
            os.path.join(PACKAGE_DIR, "config", "dense_experiment.yaml"),
        )
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        formation_num = config['formation_config']['nums']
        bias = config['formation_config']['bias']
        formation_scene = config['formation_config']['formation_scene']
        sim_type = config['formation_config']['sim_type']

        self.num_env = formation_num
        self.bias = np.array(bias)
        self.formation_scene = formation_scene
        self.desired_formation = np.array(bias)
        self.sim_type = sim_type

        results_root = rospy.get_param(
            "~results_root",
            os.path.join(
                os.environ.get("ROS_HOME", os.path.expanduser("~/.ros")),
                "distributed_rl_formation_results",
            ),
        )
        self.directory = os.path.join(
            results_root, "Cons-RL", self.sim_type, self.formation_scene
        )
        self.folder_created = False


        os.makedirs(self.directory, exist_ok=True)
        if self.id == 0 and (self.folder_created == False):
            self.existing_folders = [int(f) for f in os.listdir(self.directory) if os.path.isdir(os.path.join(self.directory, f))]
            self.index = max(self.existing_folders) + 1 if self.existing_folders else 1
            rospy.set_param('/formation_experiment/global_index', self.index)
            self.folder_created = True
        else:
            while not rospy.has_param('/formation_experiment/global_index'):
                rospy.loginfo("Waiting for the index parameter...")
                rospy.sleep(0.1)

                # 获取全局索引
            self.index = rospy.get_param('/formation_experiment/global_index', 0)

        self.trace_store_path = os.path.join(self.directory, str(self.index))

        os.makedirs(self.trace_store_path, exist_ok=True)
        self.fusion_data_directory = os.path.join(
            self.trace_store_path, "fusion_data"
        )
        os.makedirs(self.fusion_data_directory, exist_ok=True)
        self.fusion_trace_path = os.path.join(
            self.fusion_data_directory, "uav_{}.txt".format(self.id)
        )
        if not os.path.exists(self.fusion_trace_path):
            with open(self.fusion_trace_path, 'w') as f:
                f.write(
                    "# time_s x y rl_velocity_(vx,wz) "
                    "consistency_velocity_(vx,wz) alpha\n"
                )


        self.rate = rospy.Rate(30)
        self.init_time = rospy.Time.now()
        self.beam_mum = OBS_SIZE
        self.laser_cb_num = 0
        self.scan = None
        self.time_now_sec = None
        self.time_last_sec = None
        self.time_init_noise = False
        self.noise = 0.
        self.noise_std_dev = 0.0
        self.yaw = 0.
        self.pitch = 0.
        self.roll = 0.

        self.all_velocity = [np.zeros(2, dtype=float) for _ in range(self.num_env)]
        self.all_curr_pose = [np.zeros(2, dtype=float) for _ in range(self.num_env)]
        self.pose_received = [False for _ in range(self.num_env)]
        self.uav_goal_pose = [np.zeros(2, dtype=float) for _ in range(self.num_env)]
        self.consistency_control_velocity = [np.zeros(2, dtype=float) for _ in range(self.num_env)]
        self.distance_to_goal = [np.zeros(1, dtype=float) for _ in range(self.num_env)]
        self.all_siny_cosp = [np.zeros(1, dtype=float) for _ in range(self.num_env)]
        self.all_cosy_cosp = [np.zeros(1, dtype=float) for _ in range(self.num_env)]
        self.all_yaw = np.zeros(self.num_env)
        self.orientation_deviation = np.zeros(self.num_env)
        self.mean_pose = np.zeros(2)
        self.goal_radius = 1.0
        #1.0##2.5////3.5//一个应对密集障碍,一个应对大型障碍
        if self.formation_scene == 'Dense':
            self.safe_distance = 2.5
        if self.formation_scene == 'Large':
            self.safe_distance = 4.0
        if self.formation_scene == 'Dynamic':
            self.safe_distance = 3.5

        self.global_goal = np.array([44, 1])

        self.local_velocity_linear_x = 0.
        self.local_velocity_angular_z = 0.

        self.obstacle_min = 0.0
        self.obstacle_threat = 0.0
        self.formation_center = [0.0, 0.0]
        self.similarity_error = 0.0
        self.gcs_cmd = String()
        self.adj_matrix = np.array([
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0]
        ])
        self.desired_distance = self.bias[0][0] + self.bias[0][1]
        self.kv_adjust = 0.05
        self.velocity_adjust = 0.0
        self.std_dev = 0.1
        self.cur_position = PoseStamped()
        self.laser_sub = rospy.Subscriber('/%s/scan' % self.uav_namespace, LaserScan, self.laser_scan_callback)

        self.move_velocity_pub = rospy.Publisher(self.uav_namespace + "/coll_avoid/cmd_vel_flu", Twist,
                                                 queue_size=10)

        self.goal_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.global_goal_callback, queue_size=5)

        self.gcs_cmd_sub = rospy.Subscriber("/gcs_cmd", String, self.gcs_cmd_callback)

        for i in range(self.num_env):
            rospy.Subscriber('/uav_%d/assignment' % (i),
                             Point, self.local_goal_callback, (i),  queue_size=1)
            rospy.Subscriber("iris_%d/mavros/local_position/velocity_body" % (i), TwistStamped,
                             self.all_velocity_callback, (i), queue_size=1)  # 订阅UAV位置
            rospy.Subscriber("iris_%d/mavros/local_position/pose" % (i), PoseStamped,
                             self.all_pos_callback, (i), queue_size=1)  # 订阅UAV位置
            rospy.Subscriber("iris_%d/Consistency_speed" % (i), Twist,
                             self.consistency_control_callback, (i), queue_size=1)  # 订阅UAV位置


    def gcs_cmd_callback(self, msg):
        self.gcs_cmd = msg.data

    def consistency_control_callback(self, ctl, uav_id):
        #一致性编队的控制输出
        self.consistency_control_velocity[uav_id][0] = ctl.linear.x
        self.consistency_control_velocity[uav_id][1] = ctl.angular.z

    def all_pos_callback(self, loc, uav_id):
        """位置回调函数"""
        self.all_curr_pose[uav_id][0] = loc.pose.position.x + self.bias[uav_id][0]  # 更新当前UAV位置信息x坐标
        self.all_curr_pose[uav_id][1] = loc.pose.position.y + self.bias[uav_id][1]  # 更新当前UAV位置信息y坐标
        self.pose_received[uav_id] = True

        self.all_siny_cosp[uav_id] = 2 * (
                loc.pose.orientation.w * loc.pose.orientation.z + loc.pose.orientation.x * loc.pose.orientation.y)
        self.all_cosy_cosp[uav_id] = 1 - 2 * (
                loc.pose.orientation.y * loc.pose.orientation.y + loc.pose.orientation.z * loc.pose.orientation.z)
        self.all_yaw[uav_id] = math.atan2(self.all_siny_cosp[uav_id], self.all_cosy_cosp[uav_id])
        self.yaw = self.all_yaw[self.id]

        sum_x = 0
        sum_y = 0
        for uav_id in range(self.num_env):
            sum_x += self.all_curr_pose[uav_id][0]
            sum_y += self.all_curr_pose[uav_id][1]
        self.mean_pose = np.array([
            sum_x / self.num_env,
            sum_y / self.num_env])


    def all_velocity_callback(self, vel, uav_id):
        self.all_velocity[uav_id][0] = vel.twist.linear.x
        self.all_velocity[uav_id][1] = vel.twist.angular.z

    def local_goal_callback(self, msg, uav_id):
        self.uav_goal_pose[uav_id][0] = msg.x
        self.uav_goal_pose[uav_id][1] = msg.y


    def global_goal_callback(self,msg):
        self.global_goal[0] = msg.pose.position.x  # 更新目标x坐标
        self.global_goal[1] = msg.pose.position.y  # 更新目标y坐标

    def laser_scan_callback(self, scan):
        self.scan_param = [scan.angle_min, scan.angle_max, scan.angle_increment, scan.time_increment,
                           scan.scan_time, scan.range_min, scan.range_max]
        self.scan = np.array(scan.ranges)
        self.obstacle_min = np.min(self.scan)
        self.obstacle_threat = self.sigmoid(
            1.5 * (self.safe_distance - self.obstacle_min)
        )
        # if self.formation_scene == 'Large':
        #     self.obstacle_threat = 1 / np.exp(- 1.0 * (self.safe_distance - self.obstacle_min))
        #     self.obstacle_threat = np.clip(self.obstacle_threat, 0, 1)
        # else:
        #     if np.pi/2 <= min_angle or min_angle <= -np.pi/2:
        #         self.obstacle_threat = 0
        #     else:
        #         ###Dynamic
        #         if self.formation_scene == 'Dynamic':
        #             self.obstacle_threat = 1 / np.exp(- 0.1 * (self.safe_distance - self.obstacle_min))
        #             self.obstacle_threat = np.clip(self.obstacle_threat, 0, 1)
        #         # ###dense
        #         if self.formation_scene == 'Dense':
        #             self.obstacle_threat = 1 / np.exp(- 0.1 * (self.safe_distance - self.obstacle_min))
        #             self.obstacle_threat = np.clip(self.obstacle_threat, 0, 1)


        self.laser_cb_num += 1

    def get_laser_observation(self):
        scan = copy.deepcopy(self.scan)
        scan = scan
        scan[np.isnan(scan)] = 8.0
        scan[np.isinf(scan)] = 8.0
        raw_beam_num = len(scan)
        sparse_beam_num = self.beam_mum
        step = float(raw_beam_num) / sparse_beam_num
        sparse_scan_left = []
        index = 0.
        for x in range(int(sparse_beam_num / 2)):
            sparse_scan_left.append(scan[int(index)])
            index += step
        sparse_scan_right = []
        index = raw_beam_num - 1.
        for x in range(int(sparse_beam_num / 2)):
            sparse_scan_right.append(scan[int(index)])
            index -= step
        scan_sparse = np.concatenate((sparse_scan_left, sparse_scan_right[::-1]), axis=0)
        return scan_sparse / 8.0 - 0.45

    def get_local_velocity(self):
        self.local_velocity_linear_x = self.all_velocity[self.id ][0]
        self.local_velocity_angular_z = self.all_velocity[self.id ][1]
        local_velocity = [self.local_velocity_linear_x, self.local_velocity_angular_z]
        return local_velocity

    def control_vel(self, action):
        move_cmd = Twist()
        move_cmd.linear.x = action[0]
        move_cmd.linear.y = 0.
        move_cmd.linear.z = 0.
        move_cmd.angular.x = 0.
        move_cmd.angular.y = 0.
        move_cmd.angular.z = action[1]
        self.move_velocity_pub.publish(move_cmd)

    def get_local_goal(self):
        x = self.all_curr_pose[self.id][0]
        y = self.all_curr_pose[self.id][1]
        theta = self.yaw
        [goal_x, goal_y] = [self.uav_goal_pose[self.id][0],
                            self.uav_goal_pose[self.id][1]]

        local_x = (goal_x - x) * np.cos(theta) + (goal_y - y) * np.sin(theta)
        local_y = -(goal_x - x) * np.sin(theta) + (goal_y - y) * np.cos(theta)
        return [local_x, local_y]

    def reach_goal(self, curr_pose, goal, goal_radius):
        distance_to_goal = \
            np.sqrt((curr_pose[0] - goal[0])**2 +
                (curr_pose[1] - goal[1])**2)
        if distance_to_goal < goal_radius:
            return True
        return False

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def calcDist2(self, v1, v2):
        v1 = np.array(v1)
        v2 = np.array(v2)
        return np.sum(np.abs(v1 - v2) ** 2)

    def calcMatrices(self, swarm):
        self.swarm_size = len(swarm)
        self.Adj = np.zeros((self.swarm_size, self.swarm_size))
        self.Deg = np.zeros(self.swarm_size)
        self.SNL = np.zeros((self.swarm_size, self.swarm_size))

        # Adjacency and Degree
        for i in range(self.swarm_size):
            for j in range(self.swarm_size):
                self.Adj[i, j] = self.calcDist2(swarm[i], swarm[j])
                self.Deg[i] += self.Adj[i, j]

        # Symmetric Normalized Laplacian
        for i in range(self.swarm_size):
            for j in range(self.swarm_size):
                if i == j:
                    self.SNL[i, j] = 1
                else:
                    self.SNL[i, j] = -self.Adj[i, j] * (self.Deg[i] ** -0.5) * (self.Deg[j] ** -0.5)

        return self.Adj, self.Deg, self.SNL

    def formation_evaluate(self, nowForm, DesiredForm):
        A, D, Lhat = self.calcMatrices(nowForm)
        A_des, D_des, Lhat_des = self.calcMatrices(DesiredForm)
        DLhat = Lhat - Lhat_des
        cost = (np.abs(DLhat) ** 2).sum()
        return cost


    def enjoy(self, policy, action_bound):
        time_init_sec = None
        while (self.scan is None or not all(self.pose_received)) and not rospy.is_shutdown():
            rospy.loginfo_throttle(
                5.0, "%s waiting for laser and four poses", self.uav_namespace
            )
            self.rate.sleep()
        if rospy.is_shutdown():
            return
        obs = self.get_laser_observation()
        obs_stack = deque([obs, obs, obs])  # initialize the observation stack
        goal = np.asarray(self.get_local_goal())
        speed = np.asarray(self.get_local_velocity())
        should_break = False

        # formation
        while not rospy.is_shutdown():
            inference_time_start = rospy.Time.now()

            if self.reach_goal(self.mean_pose, self.global_goal, 1.0):
                self.control_vel([0.0, 0.0])
                self.rate.sleep()
                rospy.loginfo("到达目标区域")
            else:
                obs = self.get_laser_observation()
                obs_stack.append(obs)
                obs_stack = deque([obs, obs, obs])
                goal = np.asarray(self.get_local_goal())
                speed = np.asarray(self.get_local_velocity())
                state = [obs_stack, goal, speed]
                mean, scaled_action = generate_action_no_sampling(state_list=state, policy=policy,
                                                               action_bound=action_bound)
                rl_actions = scaled_action[0] # RL output
                consistency_actions = self.consistency_control_velocity[self.id]
                # Preserve the components from this exact control instant. ROS
                # callbacks may refresh the consistency array before file output.
                logged_rl_actions = np.array(rl_actions, copy=True)
                logged_consistency_actions = np.array(
                    consistency_actions, copy=True
                )

                if self.sim_type in ["无旋转无一致性","有旋转无一致性"]:
                    final_actions = rl_actions
                    fusion_alpha = 1.0
                else:
                    fusion_alpha = float(self.obstacle_threat)
                    final_actions = consistency_actions * (1 - fusion_alpha) + rl_actions * fusion_alpha
                final_actions[0] = max(min(1.0, final_actions[0]), 0.0)
                #dense
                if self.formation_scene == 'Dense':
                    final_actions[1] = final_actions[1] * (1 + self.obstacle_threat)
                    final_actions[1] = max(-1.2,min(final_actions[1],1.2))
                #large
                if self.formation_scene == 'large':
                    final_actions[1] = final_actions[1] * (1 + self.obstacle_threat)
                    final_actions[1] = max(-1.0, min(final_actions[1], 1.0))
                # Dynamic
                if self.formation_scene == 'Dynamic':
                    final_actions[1] = final_actions[1] * (1 + self.obstacle_threat)
                    final_actions[1] = max(-1.2, min(final_actions[1], 1.2))

                self.similarity_error = self.formation_evaluate(self.all_curr_pose, self.desired_formation)
                self.control_vel(final_actions)#################################
                inference_time_end = (rospy.Time.now() - inference_time_start).to_sec() * 1000

                self.rate.sleep()

                if self.gcs_cmd == 'AUTO.CANC':
                    if time_init_sec is None:
                        time_init_sec = rospy.Time.now()
                    time_now_sec = (rospy.Time.now() - time_init_sec).to_sec()
                    flight_file_path = os.path.join(self.trace_store_path, f'{self.id}_flight_trace.txt')

                    with open(flight_file_path, 'a') as f:
                        f.write(str(time_now_sec) + ' ' + str(self.all_curr_pose[self.id][0]) + ' '
                                + str(self.all_curr_pose[self.id][1]) + '\n')

                    with open(self.fusion_trace_path, 'a') as f:
                        f.write(
                            '{} {} {} ({},{}) ({},{}) {}\n'.format(
                                time_now_sec,
                                self.all_curr_pose[self.id][0],
                                self.all_curr_pose[self.id][1],
                                logged_rl_actions[0],
                                logged_rl_actions[1],
                                logged_consistency_actions[0],
                                logged_consistency_actions[1],
                                fusion_alpha,
                            )
                        )

                    if self.id == 0:
                        error_file_path = os.path.join(self.trace_store_path, f'similarity_error.txt')
                        with open(error_file_path, 'a') as f:
                            f.write(str(time_now_sec) + ' ' + str(self.similarity_error) + ' ' + str(inference_time_end) + '\n')


if __name__ == '__main__':
    controller = Controller(sys.argv[1], sys.argv[2])
    reward = None
    action_bound = [[0, -1], [1, 1]]
    policy_path = rospy.get_param(
        "~policy_path", os.path.join(SCRIPT_DIR, "policy")
    )
    policy = CNNPolicy(frames=LASER_HIST, action_space=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    policy.eval()
    opt = Adam(policy.parameters(), lr=LEARNING_RATE)
    mse = nn.MSELoss()

    if not os.path.exists(policy_path):
        os.makedirs(policy_path)

    # file = policy_path + '/formation_avo4_9820'
    file = policy_path + '/affine1222_4500'
    # file = policy_path + '/phase2_22240'
    if os.path.exists(file):
        print('[UAV ' + str(controller.id) + ']: ... Loading Model ..............')
        print(file)
        state_dict = torch.load(file, map_location=device)
        policy.load_state_dict(state_dict)
    else:
        print('[UAV ' + str(controller.id) + ']: Error: Policy File Cannot Find')
        exit()

    try:
        controller.enjoy(policy=policy, action_bound=action_bound)
    except KeyboardInterrupt:
        import traceback

        traceback.print_exc()
