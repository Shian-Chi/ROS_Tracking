from math import sin, cos, tan

class Vector:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

class VerticalTargetPositioning:
    def __init__(self):
        self.zeroPos = Vector()
        self.firPos = Vector()
        
        self.zeroMotor = Vector()
        self.firMotor = Vector()
        
        self.zeroTargetAngles = Vector()
        self.firTargetAngles = Vector()
        
        self.groundTargetPos = Vector()
        self.D_xy = None
    
    # Measure distance
    def D_xy_Value(self):
        try:
            self.D_xy = (self.firPos.z - self.zeroPos.z) / (tan(self.firTargetAngles.y) - tan(self.zeroTargetAngles.y))
        except ZeroDivisionError:
            print("Error: Division by zero in D_xy calculation.")
            self.D_xy = None
        return self.D_xy
    
    # Calculate ground target position
    def groundTargetPosition(self):
        if self.firTargetAngles.y != 0.0 and self.zeroTargetAngles.y != 0.0:
            D_xy = self.D_xy_Value()
            if D_xy is None:
                return 0.0, 0.0
            print(f"Calculated D_xy: {self.D_xy}")
            self.groundTargetPos.x = self.firPos.x + D_xy * cos(self.zeroTargetAngles.z)
            self.groundTargetPos.y = self.firPos.y + D_xy * sin(self.zeroTargetAngles.z)
            self.groundTargetPos.z = self.firPos.z - D_xy * tan(self.firTargetAngles.y)
            return self.groundTargetPos.x, self.groundTargetPos.y
        return 0.0, 0.0
    
    def update(self, longitude, latitude, altitude,
               imuRoll=0.0, imuPitch=0.0, imuYaw=0.0,
               motorRoll=0.0, motorPitch=0.0, motorYaw=0.0):
        
        # Update IMU and gimbal angles
        self.imuAndGimbalAngleUpdata(imuRoll, imuPitch, imuYaw, motorRoll, motorPitch, motorYaw)
        
        # Update positions
        self.firPos.x, self.firPos.y, self.firPos.z = self.zeroPos.x, self.zeroPos.y, self.zeroPos.z
        self.zeroPos.x, self.zeroPos.y, self.zeroPos.z = latitude, longitude, altitude

    def imuAndGimbalAngleUpdata(self, imuRoll=0.0, imuPitch=0.0, imuYaw=0.0, 
                                motorRoll=0.0, motorPitch=0.0, motorYaw=0.0):
        # Update angles for zero and fir target angles
        self.zeroTargetAngles.x, self.zeroTargetAngles.y, self.zeroTargetAngles.z = \
            self.firTargetAngles.x, self.firTargetAngles.y, self.firTargetAngles.z
        
        self.firTargetAngles.y = imuPitch + motorPitch
        self.firTargetAngles.x = imuRoll + motorRoll
        self.firTargetAngles.z = imuYaw + motorYaw
