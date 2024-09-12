#!/bin/bash

echo "123456789" | sudo -S chmod 666 /dev/ttyRTK

echo "123456789" | sudo -S chmod 666 /dev/ttyXbee

echo "123456789" | sudo -S chmod 666 /dev/ttyPixhawk

echo "123456789" | sudo -S chmod 666 /dev/ttyTHS0 


cd /home/ubuntu/yolo/yolo_tracking/
source /home/ubuntu/yolo/yolo_tracking/install/setup.bash

# 使用 pipenv run 執行 Python 腳本
pipenv run python3 /home/ubuntu/yolo/yolo_tracking/drone_ROS2.py &
sleep 5

pipenv run python3 /home/ubuntu/yolo/yolo_tracking/drone_PWCL_new.py &
sleep 5

pipenv run python3 /home/ubuntu/yolo/yolo_tracking/trackDetect.py 


