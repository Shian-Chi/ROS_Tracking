from pid.motorInit import MotorSet
from pid.parameter import Parameters, hexStr
import numpy as np
import struct
import time
from typing import List

para = Parameters()
motor = MotorSet()

X_Center = 1920 / 2
Y_Center = 1080 / 2

rxBuffer = np.zeros(12, dtype="uint8")
HC = np.uint8(62)  # header Code


class motorInformation:
    def __init__(self, ID, mode, maxAngles: float):
        self.ID = ID
        self.encoder = np.uint16(0)
        self.encoderOffset = np.uint16(0)
        self.encoderRaw = np.uint16(0)
        self.angle = float(0.0)
        self.speed = float(0.0)
        self.powers = float(0.0)
        self.current = float(0.0)
        self.temperature = 0
        self.maxAngles = maxAngles
        self.mode = mode
        self.bootEncoderVal = 0

    def update_encoder(self, encoder_value):
        if 0 <= encoder_value <= 32767:
            self.encoder = encoder_value
            self.angle = (encoder_value / 32767.0)
        else:
            print(self.mode, ": Invalid encoder value")

    def update_speed(self, speed_value):
        self.speed = speed_value

    def update_power(self, power):
        self.powers = power

    def update_encoderOffset(self, value):
        self.encoderOffset = value

    def update_encoderRaw(self, value):
        self.encoderRaw = value

    def update_voltage_current(self, current):
        self.current = current

    def getEncoder(self):
        return self.encoder

    def getAngle(self):
        self.angle = self.encoder / para.uintDegreeEncoder
        return self.angle

    def getSpeed(self):
        return self.speed

    def bootEncoder(self, val):
        self.bootEncoderVal = val


class motorCtrl:
    def __init__(self, motorID, mode, postionInit: float, maxAngles: float):
        self.ID = np.uint8(motorID)

        if mode is None:
            if self.ID == 1:
                self.mode = "yaw"
            if self.ID == 2:
                self.mode = "pitch"
            if self.ID == 3:
                self.mode = "row"
        else:
            self.mode = mode

        self.info = motorInformation(motorID, self.mode, maxAngles)
        # self.bootPosition(postionInit)
        # self.bootZero()

    def stop(self):
        cmd = 129  # 0x81
        data = struct.pack("5B", HC, cmd, self.ID, 0, HC + cmd + self.ID + 0)
        info = motor.echo(data, 5)
        return info == data

    def singleTurnVal(self, dir, value: int):
        cmd = np.uint8(165)  # 0xA5
        check_sum = Checksum(value + dir)
        value = np.uint16(value)
        buffer = struct.pack("6BH2B", HC, cmd, self.ID, 4,
                             HC + cmd + self.ID + 4, dir, value, 0, check_sum)
        info = motor.echo(buffer, 10, 13)
        return cmd, info

    def incrementTurnVal(self, value: int):
        cmd = np.uint8(167)  # 0xA7
        check_sum = Checksum(value)
        buffer = struct.pack("<5BiB", HC, cmd, self.ID, 4,
                             HC + cmd + self.ID + 4, value, check_sum)
        info = motor.echo(buffer, 10, 13)
        return cmd, info

    def readEncoder(self):
        cmd = 0x90
        buffer = struct.pack("<5B", HC, cmd, self.ID,
                             0, HC + cmd + self.ID + 0)
        info = motor.echo(buffer, 5, 12)
        if info is not None:
            info = struct.unpack("12B", info)
            # header, command, id, size, cmdSum, encoderLow, encoderHigh, encoderRawLow, encoderRawHigh, encoderOffsetLow, encoderOffsetHigh, dataSum = info
            # print(info,"\n")
            
            cs_s = Checksum(sum(info[:4])) == info[4]           
            ds_s = Checksum(info[5:-1]) == info[-1]
            
            if info[0] == 62 and cs_s and ds_s:
                encoder = info[6] << 8 | info[5]
                encoderRaw = info[8] << 8 | info[7]
                encoderOffset = info[10] << 8 | info[9]
                self.info.update_encoder(encoder)
                self.info.update_encoderRaw(encoderRaw)
                self.info.update_encoderOffset(encoderOffset)
                return True
        return False
    
    def getEncoder(self):
        if self.readEncoder():
            return True, self.info.getEncoder()
        return False, self.info.getEncoder()

    def getAngle(self):
        ret, angle = self.getEncoder()
        return ret, angle/para.uintDegreeEncoder
    
    def bootZero(self):
        print(f"Boot Initialized")
        for _ in range(2):
            if self.ID == 1:
                ret, encoder = self.getEncoder()
                if ret:
                    angle = encoder / para.uintDegreeEncoder
                    print(f'BootInit Angle: {angle}')
                    if angle < 0.0:
                        self.singleTurnVal(0, 0)
                    else:
                        self.singleTurnVal(1, 0)
                else:
                    self.bootZero()
            elif self.ID == 2:
                self.singleTurnVal(1, 0)
            if self.getEncoder() == 0:
                break
            time.sleep(1)
        print(f"{self.mode} Boot Initialized finished")    
        
# Calculate Checksum of received data
def calc_value_Checksum(value):
    value = value & 0xFFFFFFFF
    return value & 0xFF


def Checksum(value):
    if isinstance(value, (tuple, list)):
        val = np.array(value, dtype=np.uint8)
    else:
        val = np.array([value >> 24 & 0xFF, value >> 16 & 0xFF, value >> 8 & 0xFF, value & 0xFF], dtype=np.uint8)
    
    total = np.sum(val, dtype=np.uint32)
    check_sum = np.uint8(total & 0xFF)
    return check_sum



def motorSend(data, size):
    return motor.send(data, size)


def search_command(response, command, length):
    if response is None:
        return None

    # Find all the locations where Head Code appears
    p = [i for i, value in enumerate(response) if value == HC]
    for i in p:
        if i + 1 < len(response) and response[i + 1] == command:
            if i + 13 <= len(response):
                rep = response[i:i + length]  # Starting from i, take 13 bytes
                return rep
    return None


def normalized_angle(angle):
    return (angle + 180) % 360 - 180
