import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

import sys
import signal
from ctrl.pid.PID_Calc import PID_Ctrl
from ctrl.pid.motor import motorCtrl
from tutorial_interfaces.msg import Bbox

pid = PID_Ctrl()

yaw = motorCtrl(1, "yaw", 0, 90.0)
pitch = motorCtrl(2, "pitch", 0, 360.0)


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

        self.x0 = msg.x0
        self.y0 = msg.y0
        self.x1 = msg.x1
        self.y1 = msg.y1

    def get_bbox(self):
        print(f"get_bbox: {self.detect}")
        return self.x0, self.y0, self.x1, self.y1


class GimbalTimerTask(Node):
    def __init__(self, sub):
        super().__init__('gimbal_timer_task')
        
        if sub is not None:
            self.sub_para = sub
        else:
            self.sub_para = GimbalSubscriber()
            
        gimbal_period = 1 / 21  # 21 Hz
        self.gimbal_task = self.create_timer(gimbal_period, self.gimdal_ctrl)

        self.bbox_center = False
        self.l_xyxy = [0, 0, 0, 0]

    def gimdal_ctrl(self):
        m_flag1 = m_flag2 =False  # Ensure flags are initialized
        xyxy = list(self.sub_para.get_bbox())

        x, y = (xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2
        pidErr = pid.pid_run(x, y)
        print(f"PID ERR: {pidErr}")
        # Motor rotation
        if abs(pidErr[0]) != 0:
            yaw.incrementTurnVal(int(pidErr[0] * 100))
        else:
            m_flag1 = True

        if abs(pidErr[1]) != 0:
            pitch.incrementTurnVal(int(pidErr[1] * 100))
        else:
            m_flag2 = True

        self.bbox_center = m_flag1 and m_flag2
        xyxy = self.l_xyxy

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
