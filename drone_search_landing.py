# 合并后的脚本：drone_combined_ROS2.py

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Float64
from geometry_msgs.msg import Point
import sys, signal
import time
import threading
from rclpy.qos import ReliabilityPolicy, QoSProfile
from mavros_msgs.msg import Altitude, State, PositionTarget, GlobalPositionTarget
from sensor_msgs.msg import BatteryState, NavSatFix, Imu
from geometry_msgs.msg import PoseStamped, Vector3, TwistStamped, Quaternion
from geographic_msgs.msg import GeoPoseStamped
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL
from transforms3d import euler
from enum import Enum
import numpy as np
import cmath
import math
import struct
from tutorial_interfaces.srv import DroneStatus, DroneMissionPath
from tutorial_interfaces.msg import Img

# 存储任务路径点的列表
drone_point = []
# 临时检测状态变量
temp_detect_status = False

# 定义地面控制命令的枚举类
class groundControlCommand(Enum):
    DRONE_IDLE = '0'            # 无人机空闲状态
    DRONE_TAKEOFF = '1'         # 无人机起飞命令
    DRONE_MISSION_START = '2'   # 无人机任务开始命令
    DRONE_RSEARCH_START = '3'   # 无人机区域搜索开始命令

# 无人机订阅节点，订阅各类传感器和状态信息
class DroneSubscribeNode(Node):
    def __init__(self):
        super().__init__('drone_subscriber')
        # 创建各种订阅者，订阅无人机的高度、位置、速度、姿态等信息
        self.AltitudeSub = self.create_subscription(Altitude, 'mavros/altitude', self.Altcb, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.LocalPositionSub = self.create_subscription(PoseStamped, 'mavros/local_position/pose', self.LPcb, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.velocitySub = self.create_subscription(TwistStamped, 'mavros/local_position/velocity_body', self.VELcb, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.GlobalPositionSub = self.create_subscription(NavSatFix, 'mavros/global_position/global', self.GPcb, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.imuSub = self.create_subscription(Imu, 'mavros/imu/data', self.IMUcb, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.headingSub = self.create_subscription(Float64, 'mavros/global_position/compass_hdg', self.HDcb, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.StateSub = self.create_subscription(State, 'mavros/state', self.Statecb, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.InfoSub = self.create_subscription(Img, 'img', self.IMGcb, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        
        # 初始化各种变量
        self.altitude = 0.0
        self.local_x = 0.0
        self.local_y = 0.0
        self.local_z = 0.0
        self.latitude = 0.0
        self.longitude = 0.0
        self.gps_altitude = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.angu_x = 0.0
        self.angu_y = 0.0
        self.angu_z = 0.0
        self.acc_x = 0.0
        self.acc_y = 0.0
        self.acc_z = 0.0
        self.velocity = 0.0
        self.heading = 0.0
        self.state = State()

        # 图像识别相关变量
        self.first_detect = False      # 是否第一次检测到目标
        self.camera_center = False     # 目标是否在相机中心
        self.motor_pitch = 0.0         # 云台俯仰角
        self.motor_yaw = 0.0           # 云台偏航角
        self.target_latitude = 0.0     # 目标纬度
        self.target_longitude = 0.0    # 目标经度
        self.hold_status = False       # 无人机是否处于悬停状态
    
    # 状态回调函数，更新无人机状态
    def Statecb(self, msg: State):
        self.state = msg
    
    # 高度回调函数，更新无人机高度
    def Altcb(self, msg): 
        self.altitude = msg.relative

    # 本地位置回调函数，更新无人机本地坐标
    def LPcb(self, msg: PoseStamped):
        self.local_x = msg.pose.position.x
        self.local_y = msg.pose.position.y
        self.local_z = msg.pose.position.z

    # 全球定位回调函数，更新无人机经纬度
    def GPcb(self, msg):
        self.latitude = msg.latitude
        self.longitude = msg.longitude
        self.gps_altitude = msg.altitude

    # IMU回调函数，更新无人机姿态角度和加速度
    def IMUcb(self, msg: Imu):
        # 将四元数转换为欧拉角
        ned_euler_data = euler.quat2euler([msg.orientation.w,
                                        msg.orientation.x,
                                        msg.orientation.y,
                                        msg.orientation.z])
        self.pitch = radian_conv_degree(ned_euler_data[0])
        self.roll = radian_conv_degree(ned_euler_data[1])
        self.yaw = radian_conv_degree(ned_euler_data[2])
    
        # 更新角速度
        self.angu_x = msg.angular_velocity.x * 57.3
        self.angu_y = msg.angular_velocity.y * 57.3
        self.angu_z = msg.angular_velocity.z * 57.3
        
        # 更新线加速度
        self.acc_x = msg.linear_acceleration.x
        self.acc_y = msg.linear_acceleration.y
        self.acc_z = msg.linear_acceleration.z
    
    # 速度回调函数，更新无人机速度
    def VELcb(self, msg):
        self.velocity = msg.twist.linear.x 
    
    # 航向回调函数，更新无人机航向
    def HDcb(self, msg):
        self.heading = msg.data

    # 获取无人机高度
    def get_altitude(self):
        return self.altitude

    # 获取无人机本地X坐标
    def get_local_x(self):
        return self.local_x

    # 获取无人机本地Y坐标
    def get_local_y(self):
        return self.local_y

    # 获取无人机本地Z坐标
    def get_local_z(self):
        return self.local_z

    # 图像信息回调函数，更新图像识别相关变量
    def IMGcb(self, Img):
        self.first_detect = Img.first_detect
        self.camera_center = Img.camera_center
        self.motor_pitch = Img.motor_pitch
        self.motor_yaw = Img.motor_yaw
        self.target_latitude = Img.target_latitude
        self.target_longitude = Img.target_longitude
        self.hold_status = Img.hold_status

# 无人机发布节点，用于发布控制指令
class DronePublishNode(Node):
    def __init__(self, freqPositionLocal):
        super().__init__('drone_publisher')
        # 创建各种发布者，用于发送无人机的位置、模式等指令
        self.setPointPositionLocal_pub = self.create_publisher(PoseStamped, 'mavros/setpoint_position/local', 10)
        self.setPointPositionGlobal_pub = self.create_publisher(GlobalPositionTarget, 'mavros/setpoint_raw/global', 10)
        self.setCameraDetection_pub = self.create_publisher(Img, 'img', 1)
        self.setPointPositionLocal_timer = self.create_timer(1/freqPositionLocal, self.setPointPositionLocal_callback)
        self.setPointPositionGlobal_timer = self.create_timer(1/50, self.setPointPositionGlobal_callback)
        self.setCameraDetection_timer = self.create_timer(1/20, self.setCameraDetection_callback)

        # 初始化变量
        self.alwaysSend = False                 # 是否持续发送本地位置指令
        self.alwaysSendPosLocal = PoseStamped() # 持续发送的本地位置
        self.alwaysSendGlobal = False           # 是否持续发送全局位置指令
        self.alwaysSendPosGlobal = GlobalPositionTarget()  # 持续发送的全局位置
        self.vision_status = Img()              # 视觉状态信息
        self.hold_status = False                # 无人机是否处于悬停状态

    # 发送本地位置指令（单次）
    def sendPositionLocal(self, data: PoseStamped):
        self.setPointPositionLocal_pub.publish(data)

    # 定时器回调函数，持续发送本地位置指令
    def setPointPositionLocal_callback(self):
        if self.alwaysSend == True:
            self.setPointPositionLocal_pub.publish(self.alwaysSendPosLocal)

    # 发送全局位置指令（单次）
    def sendPositionGlobal(self, data: PoseStamped):
        self.setPointPositionGlobal_pub.publish(data)

    # 定时器回调函数，持续发送全局位置指令
    def setPointPositionGlobal_callback(self):
        if self.alwaysSendGlobal == True:
            self.setPointPositionGlobal_pub.publish(self.alwaysSendPosGlobal)

    # 定时器回调函数，发布视觉状态信息
    def setCameraDetection_callback(self):
        self.vision_status.hold_status = self.hold_status
        # 此处可根据需要发布视觉状态信息
        # self.setCameraDetection_pub.publish(self.vision_status)

# 无人机客户端节点，用于发送服务请求，如解锁、起飞、降落等
class DroneClientNode(Node):
    def __init__(self):
        super().__init__('drone_mavros_service')
        # 创建各种服务客户端
        self._arming = self.create_client(CommandBool, 'mavros/cmd/arming')
        self._takeoff = self.create_client(CommandTOL, 'mavros/cmd/takeoff')
        self.land = self.create_client(CommandTOL, 'mavros/cmd/land')
        self._setMode = self.create_client(SetMode, 'mavros/set_mode')

        # 等待服务可用
        while not self._arming.wait_for_service(timeout_sec=1.0):
            time.sleep(1)
        print("arming client OK")
        while not self._takeoff.wait_for_service(timeout_sec=1.0):
            time.sleep(1)
        print("takeoff client OK")
        while not self.land.wait_for_service(timeout_sec=1.0):
            time.sleep(1)
        print("land client OK")
        while not self._setMode.wait_for_service(timeout_sec=1.0):
            time.sleep(1)
        print("setMode client OK")
        
    # 发送解锁请求
    def requestCmdArm(self):
        CmdArm = CommandBool.Request()
        CmdArm.value = True
        self.future = self._arming.call_async(CmdArm)
        rclpy.spin_until_future_complete(self, self.future, timeout_sec=3.0)
        print("Drone Arming Now")
        print(self.future.result())
        return self.future.result()
    
    # 设置无人机模式，如OFFBOARD、AUTO.LAND等
    def requestSetMode(self, value):
        setModeCmdReq = SetMode.Request()
        setModeCmdReq.custom_mode = value
        self.future = self._setMode.call_async(setModeCmdReq)
        rclpy.spin_until_future_complete(self, self.future, timeout_sec=1.0)
        print(f"{value} mode now")
        return self.future.result()
    
    # 发送降落请求
    def requestLand(self):
        setModeCmdReq = SetMode.Request()
        setModeCmdReq.custom_mode = "AUTO.LAND"
        self.future = self._setMode.call_async(setModeCmdReq)
        rclpy.spin_until_future_complete(self, self.future, timeout_sec=1.0)
        print("landing------")
        return self.future.result()

    # 发送返回返航请求
    def requestRTL(self):
        setModeCmdReq = SetMode.Request()
        setModeCmdReq.custom_mode = "AUTO.RTL"
        self.future = self._setMode.call_async(setModeCmdReq)
        rclpy.spin_until_future_complete(self, self.future, timeout_sec=1.0)
        print("RTL------")
        return self.future.result()
        
# 无人机服务节点，用于处理来自地面站的任务和命令
class DroneServiceNode(Node):
    def __init__(self):
        super().__init__('drone_service')
        # 创建服务，接收任务路径和状态命令
        self.srv_missionpath = self.create_service(DroneMissionPath, 'drone_mission_path', self.drone_path_recv)
        self.srv_status = self.create_service(DroneStatus, 'drone_status', self.drone_command)
        
    # 处理任务路径的回调函数
    def drone_path_recv(self, request, response):
        global drone_point
        drone_point.clear()

        print('len(request.point_count)', len(request.point_count))
        for i in range(int(request.point_count)):
            print('point:', i, '---------------------------')
            print('latitude:', request.latitude[i])
            print('longitude:', request.longitude[i])
            print('altitude:', request.altitude[i])
            print('speed:', request.speed[i])
            print('yaw_delta:', request.yaw_delta[i])
        
            # 将任务点添加到列表
            drone_point.append([request.speed[i], request.altitude[i], request.latitude[i], request.longitude[i], request.yaw_delta[i]])
            print('len(drone_point)', len(drone_point))

        response.path_check = True
        print(request)
        print('response:', response)

        return response

    # 处理状态命令的回调函数
    def drone_command(self, request, response):
        global droneState
        # 更新无人机状态
        droneState.droneState = int(request.status)

        response.check = True
        print(request)
        print('response:', response)
        
        return response

# 无人机状态类，用于存储当前无人机的状态
class drone_state():
    def __init__(self):
        self.droneState = int(groundControlCommand.DRONE_IDLE.value)

# 无人机起飞函数，使用全局坐标
def takeoff_global(pub: DronePublishNode, sub: DroneSubscribeNode, srv: DroneClientNode, rel_alt: float):
    data = GlobalPositionTarget()
    data.coordinate_frame = 6  # FRAME_GLOBAL_REL_ALT
    data.type_mask = 0
    data.velocity.x = 0.25
    data.velocity.y = 0.25
    current_latitude = sub.latitude
    current_longitude = sub.longitude
    data.latitude = current_latitude
    data.longitude = current_longitude
    data.altitude = rel_alt
    data.yaw = degree_conv_radian(sub.yaw)
    pub.alwaysSendPosGlobal = data
    pub.alwaysSendGlobal = True

    # 检查并解锁无人机
    result = check_arm(sub, srv)
    if result == False:
        return False
    
    # 设置无人机模式为OFFBOARD
    srv.requestSetMode("OFFBOARD")

# 飞向指定全局坐标，并处理目标检测
def fly_to_global(pub: DronePublishNode, sub: DroneSubscribeNode, cli: DroneClientNode, latitude, longitude, altitude, delta_yaw, origin_lat, origin_lon):
    print('fly to latitude', latitude, " longitude:", longitude, " delta_yaw:", delta_yaw)

    pub.alwaysSendPosGlobal.latitude = latitude
    pub.alwaysSendPosGlobal.longitude = longitude
    pub.alwaysSendPosGlobal.altitude = altitude

    m_convert_lng = 1/101775.45  # 台灣經度1度約為101775.45m
    m_convert_lat = 1/110936.2   # 緯度1度約為110936.2m

    # 如果没有提供经纬度，则使用当前无人机位置
    if latitude == 0.0:
        latitude = sub.latitude
    if longitude == 0.0:
        longitude = sub.longitude

    # 飞向目标点，直到接近目标点或检测到目标
    while (((abs(sub.latitude - latitude)*110936.32 > 3) or (abs(sub.longitude - longitude)*101775.45 > 3)) and (sub.first_detect == False)):
        time.sleep(0.1)
    
    if (latitude != 0.0) and (longitude != 0.0):
        while (((abs(sub.latitude - latitude)*110936.2 > 1.5) or (abs(sub.longitude - longitude)*101775.45 > 1.5)) and (sub.first_detect == False)):
            print("lat distance:", abs(sub.latitude - latitude)*110936.2, "lng distance:", abs(sub.longitude - longitude)*101775.45, "alt distance:", abs(sub.altitude - altitude))
            time.sleep(0.1)
    
    # 如果需要旋转无人机
    if ((delta_yaw != 0)):
        print("delta_yaw", delta_yaw)

        target_yaw = sub.yaw - delta_yaw
        if target_yaw > 180:
            target_yaw = target_yaw - 360
        if target_yaw < -180:
            target_yaw = target_yaw + 360

        pub.alwaysSendPosGlobal.yaw = degree_conv_radian(sub.yaw - delta_yaw)
        while ((abs(sub.yaw - target_yaw) > 1) and (sub.first_detect == False)):
            print("Rotating to target yaw")
            time.sleep(0.1)
        print(sub.yaw)
        
    # 如果检测到目标，返回True
    if sub.first_detect == True:
        print("Target detected!")
        return True  # Return True when target is detected
    
    time.sleep(1)
    print('Reached the destination')
    return False  # Return False when no target detected

# 飞向指定全局坐标，不处理目标检测
def fly_to_global_without_detect(pub: DronePublishNode, sub: DroneSubscribeNode, latitude, longitude, altitude, delta_yaw):
    print('fly to latitude', latitude, " longitude:", longitude, " delta_yaw:", delta_yaw)

    pub.alwaysSendPosGlobal.latitude = latitude
    pub.alwaysSendPosGlobal.longitude = longitude
    pub.alwaysSendPosGlobal.altitude = altitude

    if latitude == 0.0:
        latitude = sub.latitude
    if longitude == 0.0:
        longitude = sub.longitude

    # 飞向目标点，直到接近目标点
    while (((abs(sub.latitude - latitude)*110936.32 > 3) or (abs(sub.longitude - longitude)*101775.45 > 3))):
        time.sleep(0.1)
    
    if (latitude != 0.0) and (longitude != 0.0):
        while (((abs(sub.latitude - latitude)*110936.2 > 1.5) or (abs(sub.longitude - longitude)*101775.45 > 1.5))):
            print("lat distance:", abs(sub.latitude - latitude)*110936.2, "lng distance:", abs(sub.longitude - longitude)*101775.45, "alt distance:", abs(sub.altitude - altitude))
            time.sleep(0.1)
    
    time.sleep(1)
    print('Reached the destination')

# 无人机沿着X方向移动，以使云台的Pitch角度达到要求
def drone_moving_along_the_x(pub: DronePublishNode, sub: DroneSubscribeNode, origin_heading):

    safty_distance = 10.0  # 安全距离，防止无人机移动过远

    while (90.0 - sub.motor_pitch >= 5.0):
        # 计算移动方向
        theta = sub.heading - origin_heading
        delta_y = -0.1 * math.sin(math.radians(theta))
        delta_x = 0.1 * math.cos(math.radians(theta))

        delta_lat = delta_y * (1 / 101775.45)
        delta_lon = delta_x * (1 / 110936.32)

        target_lat = sub.latitude + delta_lat
        target_lon = sub.longitude + delta_lon

        safty_distance -= math.hypot(delta_x, delta_y)
        if safty_distance <= 0.0:
            print("Over safety distance")
            return

        pub.alwaysSendPosGlobal.latitude = target_lat
        pub.alwaysSendPosGlobal.longitude = target_lon

        # 等待无人机到达目标点
        while ((abs(sub.latitude - target_lat)*110936.32 > 3) or (abs(sub.longitude - target_lon)*101775.45 > 3)):
            time.sleep(0.1)
        print(f"Pitch error: {90.0 - sub.motor_pitch} degrees")
    
    print("Drone forward movement finished")

# 将弧度转换为角度
def radian_conv_degree(Radian): 
    return ((Radian / math.pi) * 180)

# 将角度转换为弧度
def degree_conv_radian(Degree): 
    return ((Degree / 180) * math.pi)

# 无人机运行线程，处理节点的旋转
def _droneSpinThread(pub, sub, cli, srv):
    executor = MultiThreadedExecutor()
    executor.add_node(sub)
    executor.add_node(pub)
    executor.add_node(srv)
    executor.spin()

# 计算当前航向和目标航向之间的旋转角度
def rotation_angle(current_heading, target_heading):
    diff = (target_heading - current_heading) % 360

    if diff <= 180:
        return float(diff)
    else:
        return float(diff - 360)

# 检查并解锁无人机
def check_arm(sub: DroneSubscribeNode, srv: DroneClientNode):
    if sub.state.armed == False:
        i = 0
        while not srv.requestCmdArm():
            i += 1
            print('Waiting to arm')
            time.sleep(0.5)
            if i == 10:
                return False
        print("Armed successfully")
        return True
    else:
        print("Already armed")
        return True

# 计算两点之间的方位角
def calculate_bearing(lat1, lon1, lat2, lon2):
    lat_error_m = (lat2 - lat1)*110936.2  # y方向误差
    lng_error_m = (lon2 - lon1)*101775.45  # x方向误差

    bearing_deg = math.atan2(lat_error_m, lng_error_m)
    bearing_deg = math.degrees(bearing_deg)

    if bearing_deg > 180:
        bearing_deg -= 360
    elif bearing_deg < -180:
        bearing_deg += 360

    return bearing_deg * (-1)

# 信号处理函数，用于优雅地退出程序
def signal_handler(signal, frame):
    print("\nProgram exiting gracefully")
    rclpy.shutdown()
    sys.exit(0)

if __name__ == '__main__':

    # 系统中断（Ctrl + C）
    signal.signal(signal.SIGINT, signal_handler)

    global droneSub
    global dronePub
    global droneCli
    global droneSrv
    global droneState

    freq = 50  # 发布频率
    takeoffAltitude = 10.0  # 无人机起飞高度

    rclpy.init()

    # 创建节点和状态对象
    dronePub = DronePublishNode(freq)
    droneSub = DroneSubscribeNode()
    droneCli = DroneClientNode()
    droneSrv = DroneServiceNode()
    droneState = drone_state()

    # 启动无人机运行线程
    droneSpinThread = threading.Thread(target=_droneSpinThread, args=(dronePub, droneSub, droneCli, droneSrv))
    droneSpinThread.start()
    time.sleep(3)
    
    # 等待无人机获取经纬度信息
    while droneSub.latitude == 0 and droneSub.longitude == 0:
        time.sleep(0.1)

    print("_droneSpinThread started")

    while True:
        if droneState.droneState == 1:  # DRONE_TAKEOFF
            droneState.droneState = int(groundControlCommand.DRONE_IDLE.value)
            
            origin_latitude = droneSub.latitude
            origin_longitude = droneSub.longitude
            takeoff_global(dronePub, droneSub, droneCli, takeoffAltitude)
            temp_status = False
            
            while True:
                if droneState.droneState == 2:  # DRONE_MISSION_START
                    print("Mission start---------")
                    
                    print('len', len(drone_point))

                    point_len = len(drone_point)
                    lon1 = drone_point[0][3]  # 起始经度
                    lat1 = drone_point[0][2]  # 起始纬度

                    lon2 = drone_point[1][3]  # 第二个点经度
                    lat2 = drone_point[1][2]  # 第二个点纬度

                    print("Calculating initial bearing")
                    bearing = calculate_bearing(lat1, lon1, lat2, lon2)
                    print(f"Initial bearing: {bearing}")
                    fly_to_global(dronePub, droneSub, droneCli, origin_latitude, origin_longitude, takeoffAltitude, bearing, origin_latitude, origin_longitude)
                    print("Initial bearing adjustment complete")

                    # 开始区域搜索
                    for row in range(len(drone_point)):
                        print(f"Navigating to waypoint {row}")
                        fly_to_global(dronePub, droneSub, droneCli, drone_point[row][2], drone_point[row][3], 10.0, drone_point[row][4], origin_latitude, origin_longitude)
                        if droneSub.first_detect == True:
                            print("Target detected during area search")
                            # 对齐无人机的航向与云台方向
                            temp_yaw = droneSub.motor_yaw
                            fly_to_global_without_detect(dronePub, droneSub, droneSub.latitude, droneSub.longitude, droneSub.altitude, temp_yaw)
                            # 向前移动，使云台的Pitch角度垂直于目标
                            drone_moving_along_the_x(dronePub, droneSub, droneSub.heading)
                            # 降落
                            droneCli.requestLand()
                            droneState.droneState = int(groundControlCommand.DRONE_IDLE.value)
                            temp_status = True
                            break

                    if temp_status == False:  # 如果任务结束但未检测到目标
                        print("Returning to origin point")
                        fly_to_global_without_detect(dronePub, droneSub, origin_latitude, origin_longitude, 10.0, 0.0)
                        droneCli.requestLand()
                        droneState.droneState = int(groundControlCommand.DRONE_IDLE.value)
                    temp_status = False
                time.sleep(0.1)
