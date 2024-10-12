#!/bin/bash

cd /home/ubuntu/yolo/yolo_tracking_v2/
source ./install/setup.bash


echo "123456789" | sudo -S chmod 666 /dev/ttyRTK

echo "123456789" | sudo -S chmod 666 /dev/ttyXbee

echo "123456789" | sudo -S chmod 666 /dev/ttyPixhawk

echo "123456789" | sudo -S chmod 666 /dev/ttyTHS0 

ros2 run mavros mavros_node --ros-args --param fcu_url:=serial:///dev/ttyPixhawk &
sleep 20

nohup python3 /home/ubuntu/CSI_Camera_H265/CSI_H265.py &
sleep 3

nohup python3 /home/ubuntu/CSI_Camera_H265/recording_timeout.py &
sleep 3

# Flight
python3 /home/ubuntu/yolo/yolo_tracking_v2/drone_landing_ROS2.py &
sleep 20

# PWCL
python3 /home/ubuntu/yolo/yolo_tracking_v2/drone_PWCL_new.py &
sleep 20

#Lidar
#nohup python3 /home/ubuntu/yolo/yolo_tracking_v2/lidar_alt.py &
#sleep 3

# YOLO
pipenv run python3 /home/ubuntu/yolo/yolo_tracking_v2/trackDetect_2.py &

sleep 1000000000
