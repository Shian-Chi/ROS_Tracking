import serial
import struct
import Jetson.GPIO as GPIO
from time import sleep as delay

GPIO.setmode(GPIO.BOARD)
GPIO.setup(11, GPIO.OUT)

d_t = 1/1000000  # a bit send time
s_t = 0
r_t = 0
BR = 1000000 # 1 Mbps


class motorSet():
    def __init__(self):
        self.errorCount = 0  # error counter
        self.init_serial()
        self.gpioState = False

    def init_serial(self):
            self.ser = serial.Serial(
                port='/dev/ttyTHS0',
                baudrate=BR,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS,
                timeout=1 / BR * 8 * 12
            )
            self.gpioState = False
            self.errorCount = 0  # initialize error counter

    def errorHandler(self):
        self.errorCount += 1
        if self.errorCount == 3:
            self.ser.close()
            self.init_serial()
            n = self.ser.write(b"\r\n")
            if n:
                print("reset SerialPort")

    def gpioHigh(self, pin):
        if not self.gpioState:  # GPIO is off
            GPIO.output(pin, GPIO.HIGH)
            self.gpioState = True

    def gpioLow(self, pin):
        if self.gpioState:  # GPIO is on
            GPIO.output(pin, GPIO.LOW)
            self.gpioState = False

    def send(self, buf=0, size=0):
        try:
            self.gpioHigh(11)
            s_t = d_t*8*size/3 # delay time
            if size > 0:
                wLen = self.ser.write(buf)
            delay(s_t)
            self.gpioLow(11)
            return wLen
        except serial.SerialException as e:
            self.errorHandler()
            print(f"Error in send method: {e}")
            return 0

    def recv(self):
        try:
            read = b''
            n = 0
            while len(read) < 12:
                received_data = self.ser.read_all() # Recv SerialPort data
                read += received_data
                n += 1
                if n == 2: # Recv failed
                    break
            return read
        except serial.SerialException as e:
            print(f"Error in recv method: {e}")
            return False
