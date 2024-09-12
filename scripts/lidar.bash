#!/bin/bash

echo "123456789" | sudo -S  i2cdetect -y 8

source /home/ubuntu/yolo/yolo_tracking_v2/install/setup.bash

# 使用 pipenv run 執行 Python 腳本
pipenv run python3 /home/ubuntu/yolo/yolo_tracking_v2/Node/lidar/drone_lidar.py &


