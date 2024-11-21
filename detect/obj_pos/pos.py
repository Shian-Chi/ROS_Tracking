from math import sin, cos, tan, radians
from statistics import mean
from transforms3d import euler
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import ReliabilityPolicy, QoSProfile
from mavros_msgs.msg import Altitude
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import NavSatFix, Imu
from tutorial_interfaces.msg import MotorInfo
class Vector:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

class VerticalTargetPositioning:
    def __init__(self):
        self.positions = [] 
        self.motors = [] 
        self.targetAngles = [] 
        self.groundTargetPos = Vector()
        self.D_xy = None

        self.add_position()
        self.add_position()
        
        self.add_motor()
        self.add_motor()

        self.add_target_angle()
        self.add_target_angle()
    
    def add_position(self, vector=None):
        self.positions.append(vector if vector else Vector())

    def add_motor(self, vector=None):
        self.motors.append(vector if vector else Vector())

    def add_target_angle(self, vector=None):
        self.targetAngles.append(vector if vector else Vector())

    # Measure distance
    def D_xy_Value(self):
        try:
            angle_diff = tan(radians(self.targetAngles[1].y)) - tan(radians(self.targetAngles[0].y))
            if angle_diff == 0:
                print("Error: Angle difference is zero in D_xy calculation.")
                self.D_xy = None
            else:
                self.D_xy = (self.positions[1].z - self.positions[0].z) / angle_diff
        except ZeroDivisionError:
            print("Error: Division by zero in D_xy calculation.")
            self.D_xy = None
        return self.D_xy
    
    # Calculate ground target position
    def groundTargetPosition(self):
        if self.targetAngles[1].y != 0.0 and self.targetAngles[0].y != 0.0:
            D_xy = self.D_xy_Value()
            if D_xy is None:
                return 0.0, 0.0
            self.groundTargetPos.x = self.positions[1].x + D_xy * cos(radians(self.targetAngles[0].z))
            self.groundTargetPos.y = self.positions[1].y + D_xy * sin(radians(self.targetAngles[0].z))
            self.groundTargetPos.z = self.positions[1].z - D_xy * tan(radians(self.targetAngles[1].y))
            return self.groundTargetPos.x, self.groundTargetPos.y
        return 0.0, 0.0
    
    def update(self, latitude, longitude, altitude,
               imuRoll=0.0, imuPitch=0.0, imuYaw=0.0,
               motorRoll=0.0, motorPitch=0.0, motorYaw=0.0):
        
        self.positions[1].x, self.positions[1].y, self.positions[1].z = self.positions[0].x, self.positions[0].y, self.positions[0].z
        self.positions[0].x, self.positions[0].y, self.positions[0].z = longitude, latitude, altitude 

        self.imuAndGimbalAngleUpdate(imuRoll, imuPitch, imuYaw, motorRoll, motorPitch, motorYaw)

    def imuAndGimbalAngleUpdate(self, imuRoll=0.0, imuPitch=0.0, imuYaw=0.0, 
                                motorRoll=0.0, motorPitch=0.0, motorYaw=0.0):
        self.targetAngles[0].x, self.targetAngles[0].y, self.targetAngles[0].z = \
            self.targetAngles[1].x, self.targetAngles[1].y, self.targetAngles[1].z
        
        self.targetAngles[1].x = imuRoll + motorRoll
        self.targetAngles[1].y = imuPitch + motorPitch
        self.targetAngles[1].z = imuYaw + motorYaw

class VerticalTargetPositioningWithAveraging(VerticalTargetPositioning):
    def __init__(self):
        super().__init__()
        self.D_xy_values = []
        self.ground_x_values = []
        self.ground_y_values = []

    def update_and_calculate(self, latitude, longitude, altitude,
                             imuRoll, imuPitch, imuYaw,
                             motorRoll, motorPitch, motorYaw):
        self.update(latitude, longitude, altitude, imuRoll, imuPitch, imuYaw, motorRoll, motorPitch, motorYaw)
        
        if self.positions[1].z != 0.0 and self.targetAngles[0].y != 0.0:
            D_xy = self.D_xy_Value()
            if D_xy is not None:
                self.D_xy_values.append(D_xy)
            
            ground_x, ground_y = self.groundTargetPosition()
            self.ground_x_values.append(ground_x)
            self.ground_y_values.append(ground_y)

            print(f"Sample Data: D_xy = {D_xy}, Ground Target Position = (x={ground_x}, y={ground_y})")
        else:
            print("Not enough data to perform calculation.")

    def get_averages(self):
        avg_D_xy = mean(self.D_xy_values) if self.D_xy_values else 0.0
        avg_ground_x = mean(self.ground_x_values) if self.ground_x_values else 0.0
        avg_ground_y = mean(self.ground_y_values) if self.ground_y_values else 0.0
        return avg_D_xy, avg_ground_x, avg_ground_y


class radianSub(Node):
    def __init__(self):
        super().__init__('drone_subscriber')
        self.AltitudeSub = self.create_subscription(Altitude, 'mavros/altitude', self.Altcb, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.GlobalPositionSuub = self.create_subscription(NavSatFix, 'mavros/global_position/global', self.GPScb, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.imuSub = self.create_subscription(Imu, 'mavros/imu/data', self.IMUcb, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.motorSub  = self.create_subscription(MotorInfo, 'motor_info', self.motorInfocb, 10)
        self.altitude = 0.0
        self.latitude = 0.0
        self.longitude = 0.0
        self.gps_altitude = 0.0
        self.UAV_pitch_radian = 0.0
        self.UAV_roll_radian = 0.0
        self.UAV_yaw_radian = 0.0
        self.M_yaw_radian = 0.0
        self.M_pitch_radian = 0.0
        self.AltitudeSub
    
    def Altcb(self, msg): 
        self.altitude = msg.relative
 
    def GPScb(self, msg):
        self.latitude = msg.latitude
        self.longitude = msg.longitude
        self.gps_altitude = msg.altitude

    def IMUcb(self, msg :Imu):
        ned_euler_data = euler.quat2euler([msg.orientation.w,
                                        msg.orientation.x,
                                        msg.orientation.y,
                                        msg.orientation.z])
        self.pitch_radian = ned_euler_data[0]
        self.roll_radian = ned_euler_data[1]
        self.yaw_radian = ned_euler_data[2]
        
    def motorcb(self, msg):
        self.M_pitch_radian = radians(msg.pitch_angle)
        self.M_yaw_radian = radians(msg.yaw_angle)
    
class TimerTask(Node):
    def __init__(self, sub):
        super().__init__("detect_timer_task")
        self.timer = self.create_timer(1/20, self.timer_callback)
        if sub is not None:
            self.sub = sub
        else:
            self.sub = radianSub()
            
        self.vtp = VerticalTargetPositioningWithAveraging()
        self.create_timer(1, self.test_postion)
        
        self.data_samples = [
        (25.019192519150497, 121.40117406115353, 100.0, 10.0, 20.0, 30.0, 5.0, 10.0, 15.0),
        (25.01929703176808, 121.40145569310286, 95.0, 12.0, 22.0, 32.0, 7.0, 12.0, 17.0),
        (25.019397898512302, 121.40170111523015, 90.0, 14.0, 24.0, 34.0, 9.0, 14.0, 19.0)
        ]
    def test_postion(self):
        for sample in self.data_samples:
            latitude, longitude, altitude, imuRoll, imuPitch, imuYaw, motorRoll, motorPitch, motorYaw = sample
            self.vtp.update_and_calculate(latitude, longitude, altitude, imuRoll, imuPitch, imuYaw, motorRoll, motorPitch, motorYaw)
            time.sleep(1)
        avg_D_xy, avg_ground_x, avg_ground_y = self.vtp.get_averages()
        print("\nAverage Values After 3 Samples:")
        print(f"Average D_xy: {avg_D_xy}")
        print(f"Average Ground Target Position: (x={avg_ground_x}, y={avg_ground_y})")
        
    def postion(self):
        for sample in self.data_samples:
            latitude, longitude, altitude, imuRoll, imuPitch, imuYaw, motorRoll, motorPitch, motorYaw = sample
            self.vtp.update_and_calculate(latitude, longitude, altitude, imuRoll, imuPitch, imuYaw, motorRoll, motorPitch, motorYaw)

        avg_D_xy, avg_ground_x, avg_ground_y = self.vtp.get_averages()
        print("\nAverage Values After 3 Samples:")
        print(f"Average D_xy: {avg_D_xy}")
        print(f"Average Ground Target Position: (x={avg_ground_x}, y={avg_ground_y})")
        
def main():
    rclpy.init()
    taskNode = TimerTask()
    rclpy.spin(taskNode)


if __name__ == "__main__":
    main()
