from pid.motorInit import motorSet
from pid.parameter import Parameters
import numpy as np
import struct
from time import sleep as delay

para = Parameters()
motor = motorSet()

X_Center = 1920 / 2
Y_Center = 1080 / 2

rxBuffer = np.zeros(12, dtype="uint8")
HC = np.uint8(62)  # header Code


class motorInformation:
    def __init__(self, ID, maxAngles:float):
        self.ID = ID
        self.encoder = float(0.0)
        self.angle = float(0.0)
        self.speed = float(0.0)
        self.voltage = float(0.0)
        self.powers = float(0.0)
        self.current = float(0.0)

        # The parameter "maxAngles" must be the same as the "max Angle (degree)" value of the motor firmware
        self.maxAngles = maxAngles


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

    def getEncoderAndAngle(self):
        cmd = np.uint8(144)  # 0x90
        check_sum = HC + 144 + self.ID + 0
        data = struct.pack("5B", HC, cmd, self.ID, 0, check_sum)
        n = 1
        while n:
            try:
                motor.send(data, 5)  # send CMD to Motor
                delay(0.00025)  # wait for response
                r = motor.recv()  # receive response
                if (r != False) and (len(r) >= 12):
                    r = struct.unpack(str(len(r)) + "B", r)  # unpack response
                    i = -1
                    if 62 in r:  # 62 is HC
                        i = r.index(HC)  # find index of HC
                    if i >= 0:
                        if r[i + 2] == self.ID:
                            e = r[i + 6] << 8 | r[i + 5]  # extract encoder value            
                            a = float(e / para.uintDegreeEncoder) # calculate angle from "e"

                            if self.info.angle > 359.999:  # arithmetic overflow
                                print("angle read error")
                                return self.info.encoder, self.info.angle
                            else:
                                self.info.angle = a
                                self.info.encoder = e

                            # Determine whether the motor return angle is within the set range
                            if self.info.angle > self.info.maxAngles and self.info.angle < 360.0 - self.info.maxAngles:
                                print("angle read error")
                            else:
                                # Make the return angle "maxAngle" ~ "-maxAngle"
                                if self.info.angle > 360.0-self.info.maxAngles:
                                    self.info.angle = (self.info.angle + 180) % 360 - 180
                                return self.info.encoder, self.info.angle
                if n == 3:  # Failed to get encoder
                    return self.info.encoder, self.info.angle
                n += 1
            except Exception as e:
                print("ERROR:", e)
                return self.info.encoder, self.info.angle

    def bootPosition(self, pos):
        _, angle = self.getEncoderAndAngle()
        pos *= 1.0
        while angle != pos:
            move_val = int((pos-angle) * 100)
            self.incrementTurnVal(move_val)

# Calculate Checksum of received data
def calc_value_Checksum(value):
    value = value & 0xFFFFFFFF
    return value & 0xFF


# Calculate Checksum of send data
def Checksum(value):
    val = np.int32(value)
    arr = np.array(
        [val >> 24 & 0xFF, val >> 16 & 0xFF, val >> 8 & 0xFF, val & 0xFF], dtype="uint8"
    )
    total = np.sum(arr)
    check_sum = np.uint8(total & np.uint8(0xFF))
    return np.uint8(check_sum)


def motorSend(data, size):
    return motor.send(data, size)


def motorRecv():
    return motor.recv()


yaw = motorCtrl(1, 0, 90)
pitch = motorCtrl(2, 0, 45)
row = motorCtrl(3, 0, 0)
    
