#!/bin/bash

cd /home/ubuntu/yolo/yolo_tracking_v2/
source ./install/setup.bash


echo "123456789" | sudo -S chmod 666 /dev/ttyRTK

echo "123456789" | sudo -S chmod 666 /dev/ttyXbee

echo "123456789" | sudo -S chmod 666 /dev/ttyPixhawk

echo "123456789" | sudo -S chmod 666 /dev/ttyTHS0 

source ./scripts/lidar.bash

ros2 run mavros mavros_node --ros-args --param fcu_url:=serial:///dev/ttyPixhawk &
sleep 20

# 使用 pipenv run 執行 Python 腳本
pipenv run python3 /home/ubuntu/yolo/yolo_tracking_v2/Node/lidar/drone_lidar.py &
sleep 20

pipenv run python3 /home/ubuntu/yolo/yolo_tracking_v2/Node//flight/drone_ROS2.py &
sleep 20

pipenv run python3 /home/ubuntu/yolo/yolo_tracking_v2/Node/PWCL/drone_PWCL_new.py &
sleep 20

pipenv run python3 /home/ubuntu/yolo/yolo_tracking_v2/trackDetect_v2.py 


