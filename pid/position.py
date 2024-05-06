from math import sin, cos, tan


class vector():
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z
        
class verticalTargetPositioning():
    def __init__(self):
        self.zeroPos = vector()
        self.firPos = vector()
        self.groundTargetPos = vector()
        self.zeroTargetAngles = vector()
        self.firTargetAngles = vector()
        self.D_xy = None
    
    def D_xy_Value(self):
        self.D_xy = (self.firPos.z - self.zeroPos.z) / (tan(self.firTargetAngles.y) - tan(self.zeroTargetAngles.y))
    
    def groundTargetPostion(self):
        if (self.firTargetAngles.y != 0.0)and (self.zeroTargetAngles.y != 0.0):
            try:
                self.D_xy_Value()
            except:
                return 0.0, 0.0
            print(self.D_xy)
            self.groundTargetPos.x = self.firPos.x + self.D_xy * cos(self.zeroTargetAngles.z)
            self.groundTargetPos.y = self.firPos.y + self.D_xy * sin(self.zeroTargetAngles.z)
            self.groundTargetPos.z = self.firPos.z - self.D_xy * tan(self.firTargetAngles.y)
            return self.groundTargetPos.y, self.groundTargetPos.x
        return 0.0, 0.0
    
    def update(self, longitude, latitude, altitude,
                  imuRoll=0.0, imuPitch=0.0, imuYaw=0.0,
                  motorRoll=0.0, motorPitch=0.0, motorYaw=0.0):
        
        # self.imuAndGimbalAngleUpdata(imuRoll, imuPitch, imuYaw, motorRoll, motorPitch, motorYaw)
        self.zeroTargetAngles.x = self.firTargetAngles.x
        self.zeroTargetAngles.y = self.firTargetAngles.y
        self.zeroTargetAngles.z = self.firTargetAngles.z
        self.firTargetAngles.y = imuPitch + motorPitch
        self.firTargetAngles.x = imuRoll + motorRoll
        self.firTargetAngles.z = imuYaw + motorYaw
        self.firPos.x= self.zeroPos.x
        self.firPos.y= self.zeroPos.y
        self.firPos.z= self.zeroPos.z
        self.zeroPos.x = latitude
        self.zeroPos.y = longitude
        self.zeroPos.z = altitude

    def imuAndGimbalAngleUpdata(self, imuRoll=0.0, imuPitch=0.0, imuYaw=0.0, 
                                motorRoll=0.0, motorPitch=0.0, motorYaw=0.0):
        self.zeroTargetAngles.x = self.firTargetAngles.x
        self.zeroTargetAngles.y = self.firTargetAngles.y
        self.zeroTargetAngles.z = self.firTargetAngles.z
        self.firTargetAngles.y = imuPitch + motorPitch
        self.firTargetAngles.x = imuRoll + motorRoll
        self.firTargetAngles.z = imuYaw + motorYaw
        