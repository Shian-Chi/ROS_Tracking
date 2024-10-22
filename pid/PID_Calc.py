from pid.parameter import Parameters
para = Parameters()


class PID_Ctrl():
    def __init__(self):
        self.kp = 0.0056
        self.ki = 0.000036
        self.kd = 0.000000014
        self.setpoint = [para.HD_Width/2, para.HD_Height/2]
        self.error = [0, 0]
        self.last_error = [0, 0]
        self.integral = [0, 0]
        self.output = [None, None]

    def calculate(self, process_variable):
        self.output = [0, 0]
        ### yaw ###
        self.error[0] = (self.setpoint[0] - process_variable[0]) * -1
        if abs(self.error[0]) > 25:
            self.integral[0] += self.error[0]
            derivative_0 = self.error[0] - self.last_error[0]
            self.output[0] = (self.kp * self.error[0]) + (self.ki * self.integral[0]) + (self.kd * derivative_0)
            self.last_error[0] = self.error[0]

        ### pitch ###
        self.error[1] = (self.setpoint[1] - process_variable[1]) * -1
        if abs(self.error[1]) > 15:
            self.integral[1] += self.error[1]
            derivative_1 = self.error[1] - self.last_error[1]
            self.output[1] = (self.kp * self.error[1]) + (self.ki * self.integral[1]) + (self.kd * derivative_1)
            self.last_error[1] = self.error[1]

        return self.output[0], self.output[1]

    def pid_run(self, *args):
        self.output = self.calculate(args)
        return self.output