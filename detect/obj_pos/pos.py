from math import sin, cos, tan, radians
from statistics import mean

class Vector:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

class VerticalTargetPositioning:
    def __init__(self):
        self.positions = []       
        self.motors = []          
        self.targetAngles = []    
        self.groundTargetPos = Vector()
        self.D_xy = None

        self.add_position()
        self.add_position()
        
        self.add_motor()
        self.add_motor()

        self.add_target_angle()
        self.add_target_angle()
    
    def add_position(self, vector=None):
        self.positions.append(vector if vector else Vector())

    def add_motor(self, vector=None):
        self.motors.append(vector if vector else Vector())

    def add_target_angle(self, vector=None):
        self.targetAngles.append(vector if vector else Vector())

    # Measure distance
    def D_xy_Value(self):
        try:
            angle_diff = tan(radians(self.targetAngles[1].y)) - tan(radians(self.targetAngles[0].y))
            if angle_diff == 0:
                print("Error: Angle difference is zero in D_xy calculation.")
                self.D_xy = None
            else:
                self.D_xy = (self.positions[1].z - self.positions[0].z) / angle_diff
        except ZeroDivisionError:
            print("Error: Division by zero in D_xy calculation.")
            self.D_xy = None
        return self.D_xy
    
    # Calculate ground target position
    def groundTargetPosition(self):
        if self.targetAngles[1].y != 0.0 and self.targetAngles[0].y != 0.0:
            D_xy = self.D_xy_Value()
            if D_xy is None:
                return 0.0, 0.0
            self.groundTargetPos.x = self.positions[1].x + D_xy * cos(radians(self.targetAngles[0].z))
            self.groundTargetPos.y = 
            .positions[1].y + D_xy * sin(radians(self.targetAngles[0].z))
            self.groundTargetPos.z = self.positions[1].z - D_xy * tan(radians(self.targetAngles[1].y))
            return self.groundTargetPos.x, self.groundTargetPos.y
        return 0.0, 0.0
    
    def update(self, latitude, longitude, altitude,
               imuRoll=0.0, imuPitch=0.0, imuYaw=0.0,
               motorRoll=0.0, motorPitch=0.0, motorYaw=0.0):
        
        self.positions[1].x, self.positions[1].y, self.positions[1].z = self.positions[0].x, self.positions[0].y, self.positions[0].z
        self.positions[0].x, self.positions[0].y, self.positions[0].z = longitude, latitude, altitude  # 注意这里经度和纬度的顺序

        self.imuAndGimbalAngleUpdate(imuRoll, imuPitch, imuYaw, motorRoll, motorPitch, motorYaw)

    def imuAndGimbalAngleUpdate(self, imuRoll=0.0, imuPitch=0.0, imuYaw=0.0, 
                                motorRoll=0.0, motorPitch=0.0, motorYaw=0.0):
        self.targetAngles[0].x, self.targetAngles[0].y, self.targetAngles[0].z = \
            self.targetAngles[1].x, self.targetAngles[1].y, self.targetAngles[1].z
        
        self.targetAngles[1].x = imuRoll + motorRoll
        self.targetAngles[1].y = imuPitch + motorPitch
        self.targetAngles[1].z = imuYaw + motorYaw

class VerticalTargetPositioningWithAveraging(VerticalTargetPositioning):
    def __init__(self):
        super().__init__()
        self.D_xy_values = []
        self.ground_x_values = []
        self.ground_y_values = []

    def update_and_calculate(self, latitude, longitude, altitude,
                             imuRoll, imuPitch, imuYaw,
                             motorRoll, motorPitch, motorYaw):
        self.update(latitude, longitude, altitude, imuRoll, imuPitch, imuYaw, motorRoll, motorPitch, motorYaw)
        
        if self.positions[1].z != 0.0 and self.targetAngles[0].y != 0.0:
            D_xy = self.D_xy_Value()
            if D_xy is not None:
                self.D_xy_values.append(D_xy)
            
            ground_x, ground_y = self.groundTargetPosition()
            self.ground_x_values.append(ground_x)
            self.ground_y_values.append(ground_y)

            print(f"Sample Data: D_xy = {D_xy}, Ground Target Position = (x={ground_x}, y={ground_y})")
        else:
            print("Not enough data to perform calculation.")

    def get_averages(self):
        avg_D_xy = mean(self.D_xy_values) if self.D_xy_values else 0.0
        avg_ground_x = mean(self.ground_x_values) if self.ground_x_values else 0.0
        avg_ground_y = mean(self.ground_y_values) if self.ground_y_values else 0.0
        return avg_D_xy, avg_ground_x, avg_ground_y

def main():
    vtp = VerticalTargetPositioningWithAveraging()

    data_samples = [
        (30.0, 120.0, 100.0, 10.0, 20.0, 30.0, 5.0, 10.0, 15.0),
        (31.0, 121.0, 95.0, 12.0, 22.0, 32.0, 7.0, 12.0, 17.0),
        (32.0, 122.0, 90.0, 14.0, 24.0, 34.0, 9.0, 14.0, 19.0)
    ]

    for sample in data_samples:
        latitude, longitude, altitude, imuRoll, imuPitch, imuYaw, motorRoll, motorPitch, motorYaw = sample
        vtp.update_and_calculate(latitude, longitude, altitude, imuRoll, imuPitch, imuYaw, motorRoll, motorPitch, motorYaw)

    avg_D_xy, avg_ground_x, avg_ground_y = vtp.get_averages()
    print("\nAverage Values After 3 Samples:")
    print(f"Average D_xy: {avg_D_xy}")
    print(f"Average Ground Target Position: (x={avg_ground_x}, y={avg_ground_y})")

if __name__ == "__main__":
    main()
