import rclpy
from rclpy.node import Node
from tutorial_interfaces.msg import Bbox
import cv2

class ImageDisplayNode(Node):
    def __init__(self):
        super().__init__('image_display_node')

        # 訂閱Bbox主題
        self.subscription = self.create_subscription(
            Bbox,
            'bbox',  # 替換為實際的主題名稱
            self.bbox_callback,
            10)
        self.subscription  # 防止未使用變量警告

        # 初始化bbox數據
        self.bbox_data = None

        # 打開RTSP流
        self.rtsp_url = 'rtsp://140.131.13.133:8080/video_feed'  # 替換為實際的RTSP URL
        self.cap = cv2.VideoCapture(self.rtsp_url)

        if not self.cap.isOpened():
            self.get_logger().error('無法打開RTSP流')
            exit(1)
        else:
            print("RTSP is opened")

    def bbox_callback(self, msg):
        # 存儲接收到的bbox數據
        self.bbox_data = msg

def main(args=None):
    rclpy.init(args=args)
    node = ImageDisplayNode()

    try:
        while rclpy.ok():
            # 從RTSP流讀取幀
            ret, frame = node.cap.read()
            if not ret:
                node.get_logger().error('無法從RTSP流讀取幀')
                break

            # 如果有bbox數據，繪制邊界框
            if node.bbox_data and node.bbox_data.detect:
                x0 = node.bbox_data.x0
                y0 = node.bbox_data.y0
                x1 = node.bbox_data.x1
                y1 = node.bbox_data.y1

                # 在幀上繪制矩形
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)

                # 可選地，在框上方顯示類別ID和置信度
                label = f'ID: {node.bbox_data.class_id}, Conf: {node.bbox_data.confidence:.2f}'
                cv2.putText(frame, label, (x0, y0 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 顯示圖像
            cv2.imshow('RTSP Stream with Bbox', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # 處理ROS2回調
            rclpy.spin_once(node, timeout_sec=0)

    except KeyboardInterrupt:
        pass
    finally:
        node.cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
