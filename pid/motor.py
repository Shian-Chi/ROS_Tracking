from pid.motorInit import MotorSet
from pid.parameter import Parameters, hexStr
import numpy as np
import struct
from time import sleep as delay
from typing import List

para = Parameters()
motor = MotorSet()

X_Center = 1920 / 2
Y_Center = 1080 / 2

rxBuffer = np.zeros(12, dtype="uint8")
HC = np.uint8(62)  # header Code


class motorInformation:
    def __init__(self, ID, maxAngles: float):
        self.ID = ID
        self.encoder = float(0.0)
        self.angle = float(0.0)
        self.speed = float(0.0)
        self.powers = float(0.0)
        self.current = float(0.0)
        self.temperature = 0
        self.maxAngles = maxAngles

    def update_encoder(self, encoder_value):
        if 0 <= encoder_value <= 32767:  # 确保编码器值在有效范围内
            self.encoder = encoder_value
            self.angle = (encoder_value / 32767.0)
        else:
            print("Invalid encoder value")

    def update_speed(self, speed_value):
        self.speed = speed_value

    def update_power(self, power):
        self.powers = power

    def update_voltage_current(self, current):
        self.current = current
        
    def getEncoder(self):
        return self.encoder
    
    def getAngle(self):
        if self.encoder == 32767:
            self.angle = 0.0
        else:
            self.angle = self.encoder / para.uintDegreeEncoder
        return self.angle
    
    def getSpeed(self):
        return self.speed



class motorCtrl:
    def __init__(self, motorID, postionInit: float, maxAngles:float):
        self.ID = np.uint8(motorID)
        self.info = motorInformation(motorID, maxAngles)
        self.bootPosition(postionInit)

    def stop(self):
        cmd = 129  # 0x81
        data = struct.pack("5B", HC, cmd, self.ID, 0, HC + cmd + self.ID + 0)
        motor.send(data, 5)
        return rxBuffer

    def singleTurnVal(self, dir, value: int):
        global rxBuffer
        cmd = np.uint8(165)  # 0xA5
        check_sum = Checksum(value + dir)
        value = np.uint16(value)
        buffer = struct.pack("6BH2B", HC, cmd, self.ID, 4, HC + cmd + self.ID + 4, dir, value, 0, check_sum)
        motor.send(buffer, 10)
        return buffer

    def incrementTurnVal(self, value: int):
        cmd = np.uint8(167)  # 0xA7
        check_sum = Checksum(value)
        buffer = struct.pack("<5BiB", HC, cmd, self.ID, 4, HC + cmd + self.ID + 4, value, check_sum)
        motor.send(buffer, 10)
        return buffer

    def motorZero(self, dir):
        data = struct.pack("10B", 62, 165, self.ID, 4, 62 + 165 + self.ID + 4, dir, 0, 0, 0, dir)
        motor.send(data, 10)
        delay(0.1)
    
    def readMotorStatus2(self, data=None, cmd=0x9C):
        header = 0x3E  # Frame header
        command = cmd  # Command to read motor status 2
        data_length = 0  # Data length, assumed to be 0 because the command does not require additional data
        checksum = Checksum(header + command + self.ID) # Calculate checksum
        
        motor.ser.flushOutput()
        if command == 0x9C:
            # Sending the command frame
            command_packet = struct.pack("5B", header, command, self.ID, data_length, checksum)
            motor.send(command_packet, 5)
        
        # Receiving the response
        if data is None:
            response = motor.recv(13, 5)  # Update the byte length to match your data format
        else:
            response = data
            
        rLen = len(response)
        response = search_command(response, command)
        print(f"Rx size:{rLen}, response: {hexStr(response)}")
        if response is None:
            return
        hc, c, id, dlen, hcs, tmp, _, _, speed_low, speed_high, encoder_low, encoder_high, dcs = response

        # Processing torque current and output power
        if hc == HC and hcs == Checksum(sum([hc, c, id, dlen])):
            # Processing motor speed
            speed = (speed_high << 8) | speed_low
            if speed > 32767:
                speed -= 65536  # Convert to signed integer

            # Processing encoder position
            encoder = (encoder_high << 8) | encoder_low

            # Updating motor information
            self.info.update_encoder(encoder)
            self.info.update_speed(speed)
            self.info.temperature = tmp  # Assuming temperature is stored in the voltage property

            print(f"Motor Status Updated: Temperature {tmp}, Speed {speed}, Encoder {encoder}")
        else:
            print("motor recv data error")
            
    def getAngle(self):
        self.readMotorStatus2()
        return self.info.getAngle()
    
    def bootPosition(self, pos):
        for i in range(6):  # 最多嘗試6次以達到指定的精度
            print("bootPosition count:", i)
            current_angle = self.getAngle()
            print(current_angle)

            if self.ID == 2:
                # if current_angle > 90.0:
                #     diff = (current_angle + 180) % 360 - 180
                # else:
                    diff = pos - current_angle
            elif self.ID == 1:
                if current_angle > 90.0:
                    diff = ((current_angle + 180) % 360 - 180)
                else:
                    diff = (pos - current_angle)
            move_val = int(diff * 100)
            if abs(move_val) > 0:  # 只有當有實際移動時才發送命令
                self.incrementTurnVal(move_val)
            else:
                break  # 如果沒有需要移動的距離，則提前結束循環
            delay(0.1)
            
# Calculate Checksum of received data
def calc_value_Checksum(value):
    value = value & 0xFFFFFFFF
    return value & 0xFF


# Calculate Checksum of send data
def Checksum(value):
    val = np.int32(value)
    arr = np.array([val >> 24 & 0xFF, val >> 16 & 0xFF, val >> 8 & 0xFF, val & 0xFF], dtype="uint8")
    total = np.sum(arr)
    check_sum = np.uint8(total & np.uint8(0xFF))
    return np.uint8(check_sum)


def motorSend(data, size):
    return motor.send(data, size)


def search_command(response, command):
    p = [i for i, value in enumerate(response) if value == HC]  # Find all the locations where Head Code appears
    for i in p:
        if i + 1 < len(response) and response[i + 1] == command:
            if i + 13 <= len(response):
                rep = response[i:i + 13]  # Starting from i, take 13 bytes
                return rep
    return None  