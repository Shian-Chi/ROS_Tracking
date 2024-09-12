import rclpy
from rclpy.node import Node
from tutorial_interfaces.msg import Lidar
# from mavros_msgs.msg imoprt Lidar
from lidar.lidar import LIDARLite_v4LED


lidar = LIDARLite_v4LED()
lidar.configure()


class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('lidar_publisher')
        # Lidar 
        self.Lidar = Lidar()
        self.disPublish = self.create_publisher(Lidar, "lidar", 10)
        lidar_timer = 1.0
        self.lidar_timer = self.create_timer(lidar_timer, self.lidar_callback)

    def lidar_callback(self):
        lidar.takeRange()
        lidar.waitForBusy()
        dis = float(lidar.readDistance() / 256)
        print(f"distance: {dis}cm")
        self.Lidar.distance_cm = dis
        self.disPublish.publish(self.Lidar)
        
def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()