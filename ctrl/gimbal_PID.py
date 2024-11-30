import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

import sys, time
import signal
from ctrl.pid.PID_Calc import PID_Ctrl
from ctrl.pid.motor import motorCtrl, motorInitPositions
from ctrl.pid.parameter import Parameters
from tutorial_interfaces.msg import Bbox, MotorInfo

pid = PID_Ctrl()
para = Parameters()
yaw = motorCtrl(1, "yaw", 90.0)
pitch = motorCtrl(2, "pitch", 360.0)


class GimbalSubscriber(Node):
    def __init__(self):
        super().__init__('PID_Ctrl_subscriber')
        self.subscription = self.create_subscription(Bbox, 'bbox', self.listener_callback, 10)
        self.subscription  # prevent unused variable warning

        self.detect = False
        self.ID = -1
        self.conf = -1
        self.x0 = 0
        self.y0 = 0
        self.x1 = 0
        self.y1 = 0

    def listener_callback(self, msg):        
        self.detect = msg.detect
        self.ID = msg.class_id
        self.conf = msg.confidence
        self.x0 = msg.x0
        self.y0 = msg.y0
        self.x1 = msg.x1
        self.y1 = msg.y1

    def get_bbox(self):
        # print(f"get_bbox: {self.detect}")
        return self.x0, self.y0, self.x1, self.y1


def getGimbalEncoders():
    Y_ret, Y_Encoder= yaw.getEncoder()
    P_ret, P_Encoder= pitch.getEncoder()
    return Y_Encoder, P_Encoder


class GimbalTimerTask(Node):
    def __init__(self, sub):
        super().__init__('gimbal_timer_task')
        motorInitPositions(yaw, 0.0)
        time.sleep(1)
        motorInitPositions(pitch, 10.0)

        self.error_range = 0.15 # %
        self.width_error_range = para.video_width * self.error_range
        self.height_error_range = para.video_height * self.error_range
        if sub is not None:
            self.sub_para = sub
        else:
            try:
                self.sub_para = GimbalSubscriber()
            except:
                print("no subscriber parameter")
                sys.exit(1)
                
        self.gimbal_task = self.create_timer(1 / 21, self.gimdal_ctrl)

        # Read motor Info
        self.motorInfoPublish = self.create_publisher(MotorInfo, "motor_info", 10)
        self.motor_timer = self.create_timer(1/10, self.motor_callback)

        self.motorInfo = MotorInfo()
        
        self.center_status = False
        self.l_xyxy = [0, 0, 0, 0]
        self.pitchEncoder, self.yawEncoder = 0, 0
        self.pitchAngle, self.yawAngle = 0.0, 0.0
        
    def gimdal_ctrl(self):
        is_center = False
        err = [0, 0]
        if self.sub_para.detect:
            xyxy = list(self.sub_para.get_bbox())

            x, y = (xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2
            pid_output, err = pid.pid_run(x, y)
            # Motor rotation
            yaw.incrementTurnVal(int(pid_output[0] * 100))
            pitch.incrementTurnVal(int(pid_output[1] * 100))
        
        self.center_status = err[0] <= self.width_error_range and err[1] <= self.height_error_range
                    

    def motor_callback(self):
        _, yawData = yaw.getEncoder()
        time.sleep(0.01)
        _, pitchData = pitch.getEncoder()
        self.motorInfo.pitch_pluse = pitchData
        self.motorInfo.yaw_pluse =  yawData 
        pA, yA = pitchData / para.uintDegreeEncoder, yawData / para.uintDegreeEncoder
        self.motorInfo.pitch_angle = pA
        self.motorInfo.yaw_angle = yA
        # print(f"center: {self.center_status}\nyaw angle: {yA:.2f}, pitch angle: {pA:.2f}\n")
        self.motorInfoPublish.publish(self.motorInfo)
        
def spinThread(sub, task):
    executor = MultiThreadedExecutor()
    executor.add_node(sub)
    executor.add_node(task)
    executor.spin()

    sub.destroy_node()
    task.destroy_node()
    rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)

    global ros_sub, timer_task

    ros_sub = GimbalSubscriber()
    timer_task = GimbalTimerTask(ros_sub)

    spinThread(ros_sub, timer_task)


if __name__ == '__main__':
    main()