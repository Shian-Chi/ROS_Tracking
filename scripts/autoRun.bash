#!/bin/bash

cd /home/ubuntu/yolo/yolo_tracking_v2/
source ./install/setup.bash


echo "123456789" | sudo -S chmod 777 /dev/ttyRTK

echo "123456789" | sudo -S chmod 777 /dev/ttyXbee

echo "123456789" | sudo -S chmod 777 /dev/ttyPixhawk 

echo "123456789" | sudo -S chmod 777 "/dev/ttyTHS0"

echo "123456789" | sudo -S i2cdetect -y 8
echo "123456789" | sudo -S  chmod 777 /dev/i2c-8

echo "run mavros mavros_node"
ros2 run mavros mavros_node --ros-args --param fcu_url:=serial:///dev/ttyPixhawk &
sleep 20

nohup python3 /home/ubuntu/CSI_Camera/CSI_H265.py &

#nohup python3 /home/ubuntu/bbb.py
#sleep 3

nohup python3 /home/ubuntu/CSI_Camera_H265/recording_timeout.py &
sleep 3

# Flight
echo "run Flight"
python3 /home/ubuntu/torch_v2/yolo_tracking_v2/flight/drone_landing_ROS2.py &
sleep 10

# PWCL
echo "run PWCL"
python3 /home/ubuntu/torch_v2/yolo_tracking_v2/PWCL/drone_PWCL_new.py &
sleep 10

#Lidar
echo "run Lidar"
echo "123456789" | sudo -S chmod +x /home/ubuntu/torch_v2/yolo_tracking_v2/lidar/lidar_alt.py
nohup python3 /home/ubuntu/torch_v2/yolo_tracking_v2/lidar/lidar_alt.py &
sleep 3

# YOLO
echo "run YOLO"
pipenv run python3 /home/ubuntu/torch_v2/yolo_tracking_v2/detect/trackDetect_2.py &

sleep infinity
