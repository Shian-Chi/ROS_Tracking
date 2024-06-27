#!/bin/bash
sudo chmod 666 "/dev/ttyTHS0"
sudo i2cdetect -y 8

#if [ -e "/dev/ttyUSB0" ]; then 
#    sudo chmod 666 "/dev/ttyUSB0"
#    echo "/dev/ttyUSB0 exists."
#else
#    echo "/dev/ttyUSB0 does not exist."
#fi
