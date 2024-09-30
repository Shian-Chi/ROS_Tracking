import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Range
import smbus2
import time

class LidarPublisher(Node):
    def __init__(self):
        super().__init__('lidar_publisher')
        self.publisher_ = self.create_publisher(Range, 'lidar_distance', 10)
        self.timer = self.create_timer(0.1, self.publish_distance)  # 10Hz
        self.bus = smbus2.SMBus(8)  # I2C Bus number may vary
        self.lidar_address = 0x62

    def publish_distance(self):
        distance = self.read_lidar_distance()
        if distance is not None:
            msg = Range()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.radiation_type = Range.INFRARED
            msg.field_of_view = 0.0477  # Beam divergence in radians
            msg.min_range = 0.05  # 5 cm minimum range
            msg.max_range = 10.0  # 10 meters maximum range
            msg.range = distance / 100.0  # Convert cm to meters
            self.publisher_.publish(msg)

    def read_lidar_distance(self):
        try:
            # Write to register 0x00 to trigger distance measurement
            self.bus.write_byte_data(self.lidar_address, 0x00, 0x04)
            time.sleep(0.02)
            
            # Read the distance from registers 0x10 (low) and 0x11 (high)
            low_byte = self.bus.read_byte_data(self.lidar_address, 0x10)
            high_byte = self.bus.read_byte_data(self.lidar_address, 0x11)
            distance = (high_byte << 8) + low_byte
            return distance
        except Exception as e:
            self.get_logger().error(f"Error reading from LIDAR: {e}")
            return None

def main(args=None):
    rclpy.init(args=args)
    lidar_publisher = LidarPublisher()
    rclpy.spin(lidar_publisher)
    lidar_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
