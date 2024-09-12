#!/bin/bash
echo "123456789" | sudo -S chmod 777 "/dev/ttyTHS0"
echo "123456789" | sudo -S i2cdetect -y 8

#if [ -e "/dev/ttyUSB0" ]; then 
#    sudo chmod 666 "/dev/ttyUSB0"
#    echo "/dev/ttyUSB0 exists."
#else
#    echo "/dev/ttyUSB0 does not exist."
#fi
