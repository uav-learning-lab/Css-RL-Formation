import rospy  # ROS Python API
import numpy as np  # 用于数组和数值计算
from scipy.optimize import linear_sum_assignment  # 线性求解任务分配
from scipy.spatial.distance import cdist  # 计算集合点之间的距离
from distributed_rl_formation.msg import AuctionInfo  # 自定义消息类型，用于拍卖中的信息传递
from geometry_msgs.msg import Point, PoseStamped, TwistStamped # ROS标准消息类型, 用于坐标和位姿
from std_msgs.msg import Float32MultiArray  # ROS标准消息类型, 用于多浮点数数组
import sys  # 系统参数模块
import math  # 数学运算模块
import yaml
import os

class UAV_AUCTION:
    def __init__(self, uav_id):
        package_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        config_file = rospy.get_param(
            "~config_file",
            os.path.join(package_dir, "config", "dense_experiment.yaml"),
        )
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        formation_num = config['formation_config']['nums']
        bias = config['formation_config']['bias']
        sim_type = config['formation_config']['sim_type']

        """初始化UAV拍卖系统"""
        self.uav_id = int(uav_id)  # 当前UAV的唯一标识符
        self.num_uavs = formation_num # UAV数量
        self.desired_formation = np.array(bias)
        self.bias = np.array(bias)
        self.sim_type = sim_type

        self.error = 0.0
        self.align_pos_self = None
        self.assignment = False  # 是否已分配任务
        self.all_pos_reported = False  # 是否所有位置已报告
        self.all_auction_reported = False  # 是否所有拍卖消息已报告
        self.final_assignment = np.zeros(2)  # 初始化分配结果
        self.bid_matrix = np.zeros((self.num_uavs, self.num_uavs))  # 初始化竞标矩阵
        self.priority_indices = []  # UAV优先索引
        self.available_tasks_list = [float('inf')] * self.num_uavs  # 可用任务列表，初始化为无限大
        self.available_uavs_list = [float('inf')] * self.num_uavs  # 可用UAV列表，初始化为无限大
        self.curr_pose = [np.zeros(2, dtype=float) for _ in range(self.num_uavs)]  # 初始化当前位置信息
        self.align_pos = np.zeros((self.num_uavs, 2))  # 初始化校准位置
        self.goal = [70,0]  # 初始化目标位置
        self.auction_msg = AuctionInfo()  # 初始化拍卖信息消息
        self.setup_ros_communications()  # 设置ROS通信
        self.max_bid_index = None  # 最大竞标索引

    def setup_ros_communications(self):
        """设置ROS的发布者和订阅者"""
        self.bid_matrix_pub = rospy.Publisher(f'/uav_{self.uav_id}/bid_matrix', Float32MultiArray, queue_size=10)  # 发布竞标矩阵
        self.assign_task_pub = rospy.Publisher(f'/uav_{self.uav_id}/auction', AuctionInfo, queue_size=10)  # 发布任务拍卖信息
        self.assignment_pos_pub = rospy.Publisher(f'/uav_{self.uav_id}/assignment', Point, queue_size=10)  # 发布分配的任务位置

        self.bid_matrix_subs = [
            rospy.Subscriber(f'/uav_{i}/bid_matrix', Float32MultiArray, self.bid_matrix_callback, i, queue_size=5) for i in range(self.num_uavs) if i != self.uav_id
        ]  # 订阅其他UAV的竞标矩阵
        self.goal_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback, queue_size=5)  # 订阅目标点

        for i in range(self.num_uavs):
            rospy.Subscriber("iris_%d/mavros/local_position/pose" % (i), PoseStamped,
                             self.pos_callback, (i), queue_size=1)  # 订阅UAV位置
            rospy.Subscriber(f"/uav_{i}/auction", AuctionInfo, self.auction_callback, (i))  # 订阅拍卖信息

    def auction_callback(self, msg, uav_id):
        """拍卖信息回调函数"""
        self.available_uavs_list[uav_id] = int(msg.uav_id)  # 更新可用UAV列表
        self.available_tasks_list[msg.task_id] = float(msg.task_id)  # 更新可用任务列表

    def update_priority_indices(self):
        """更新优先级索引"""
        free_uav_indices = [i for i, status in enumerate(self.available_uavs_list) if math.isinf(status)]  # 获取空闲UAV索引
        if not free_uav_indices:
            self.all_auction_reported = True
            return
        distance_sums = np.array([[np.linalg.norm(self.curr_pose[idx] - t) for t in self.align_pos] for idx in free_uav_indices])
        distance_sums = np.sum(distance_sums, axis=1)
        sorted_indices = np.argsort(-distance_sums)  # 按距离和排序
        self.priority_indices = [free_uav_indices[idx] for idx in sorted_indices]

    def goal_callback(self, msg):
        """目标点回调函数"""
        self.goal[0] = msg.pose.position.x  # 更新目标x坐标
        self.goal[1] = msg.pose.position.y  # 更新目标y坐标
        rospy.loginfo("Updated goal from RViz: %s" % [self.goal])

    def bid_matrix_callback(self, msg, uav_id):
        """竞标矩阵回调函数"""
        self.bid_matrix[uav_id] = np.array(msg.data).reshape((self.num_uavs,))  # 重塑竞标矩阵

    def pos_callback(self, msg, uav_id):
        """位置回调函数"""
        self.curr_pose[uav_id][0] = msg.pose.position.x + self.bias[uav_id][0]  # 更新当前UAV位置信息x坐标
        self.curr_pose[uav_id][1] = msg.pose.position.y + self.bias[uav_id][1]  # 更新当前UAV位置信息y坐标

    def re_initialize(self):
        """处理当前位置和理想编队位置"""
        self.curr_pose = np.array(self.curr_pose)
        _, self.align_pos_self, self.error = self.find_optimal_assignment(self.curr_pose.T, self.desired_formation.T)
        align_pos = self.align_pos_self.T
        available_tasks_list = [float('inf')] * self.num_uavs
        available_uavs_list = [float('inf')] * self.num_uavs
        return align_pos,available_tasks_list,available_uavs_list

    def arun(self, q, p):
        """计算旋转矩阵R和变换向量t"""
        d, n = q.shape
        mu_q = q.mean(1)
        direction_vector = self.goal - mu_q  # 方向向量
        norm = np.linalg.norm(direction_vector)
        if norm == 0:
            unit_direction_vector = np.zeros_like(direction_vector)
        else:
            unit_direction_vector = 3.0 * (direction_vector / norm)  # 规范化方向向量
        mu_q = mu_q + unit_direction_vector
        mu_p = p.mean(1)  # 获取p的均值
        Q = q - np.tile(mu_q, (n, 1)).T  # 均值中心化
        P = p - np.tile(mu_p, (n, 1)).T  # 均值中心化
        H = np.matmul(Q, P.T)  # 计算协方差矩阵
        U, _, Vt = np.linalg.svd(H, full_matrices=True, compute_uv=True)  # 奇异值分解
        D = np.eye(d)
        if np.linalg.det(np.matmul(U, Vt)) < 0:
            D[d - 1, d - 1] = -1
        R = np.matmul(np.matmul(U, D), Vt)
        t = mu_q - np.matmul(R, mu_p)
        return R, t

    def align(self, q, p):
        """对齐p与q"""
        if self.sim_type in ["有旋转有一致性","有旋转无一致性",'RL_Formation']:
            R, t = self.arun(q, p)
            return np.dot(R, p) + np.tile(t, (p.shape[1], 1)).T
        else:
            _, t = self.arun(q, p)
            return p + np.tile(t, (p.shape[1], 1)).T

    def find_optimal_assignment(self, q, p, last=None):
        """找到q和p的最优分配"""
        if last is None:
            last = [i for i in range(q.shape[1])]
        qq = np.zeros_like(q)
        for vehid, formpt in enumerate(last):
            qq[:, formpt] = q[:, vehid]
        paligned = self.align(qq, p)
        S = cdist(q.T, paligned.T)  # 计算距离矩阵
        _, P = linear_sum_assignment(S)  # 获取线性求解结果
        error = np.linalg.norm(paligned - q)  # 计算对齐误差
        return P.tolist(), paligned, error

    def auction_phase(self):
        """执行拍卖阶段"""
        distances = np.zeros(self.num_uavs)
        bid_matrix = np.zeros(self.num_uavs)
        for i in range(self.num_uavs):
            distances[i] = np.linalg.norm(self.curr_pose[self.uav_id] - self.align_pos[i])  # 计算距离
            bid_matrix[i] = 1 / (distances[i] + 1e-6)  # 计算竞标值
        self.bid_matrix[self.uav_id] = bid_matrix  # 更新竞标矩阵
        bid_msg = Float32MultiArray()
        bid_msg.data = bid_matrix.flatten()
        self.bid_matrix_pub.publish(bid_msg)  # 发布竞标矩阵消息

        self.update_priority_indices()
        self.choose_task(bid_matrix)  # 选择任务

    def choose_task(self, self_bid):
        """选择当前UAV的任务"""
        free_task_indices = [index for index, value in enumerate(self.available_tasks_list) if math.isinf(value)]
        if not free_task_indices:
            self.all_auction_reported = True
            return
        self.max_bid_index = max(free_task_indices, key=lambda i: self_bid[i])  # 获取最大竞标索引
        if (self.uav_id == self.priority_indices[0]) and (not self.assignment):
            self.auction_msg.header.stamp = rospy.Time.now()  # 更新时间戳
            self.auction_msg.uav_id = self.uav_id  # 当前UAV的ID
            self.auction_msg.task_id = self.max_bid_index  # 当前UAV分配的任务ID
            self.assign_task_pub.publish(self.auction_msg)  # 发布拍卖消息
            self.final_assignment = self.align_pos[self.max_bid_index]  # 更新分配位置
            self.assignment = True  # 设置分配标志

            final_coordinates_pub = Point()
            final_coordinates_pub.x = self.final_assignment[0]
            final_coordinates_pub.y = self.final_assignment[1]
            self.assignment_pos_pub.publish(final_coordinates_pub)

    def spin(self):
        """主循环"""
        while not rospy.is_shutdown():
            self.align_pos , self.available_tasks_list, self.available_uavs_list = self.re_initialize()
            self.assignment = False
            self.all_auction_reported = False
            rospy.sleep(0.2)
            while not self.all_auction_reported:
                self.auction_phase()

if __name__ == '__main__':
    rospy.init_node('uav_auction', anonymous=True)
    uav_auction = UAV_AUCTION(sys.argv[1])
    try:
        uav_auction.spin()
    except KeyboardInterrupt:
        pass
