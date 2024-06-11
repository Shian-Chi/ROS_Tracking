import serial
import struct
import Jetson.GPIO as GPIO
import time

GPIO.setmode(GPIO.BOARD)
GPIO.setup(11, GPIO.OUT)

class MotorSet:
    def __init__(self):
        self.ser = None
        self.error_count = 0
        self.gpio_state = False
        self.baudrate = 1000000  # 1 Mbps
        self.init_serial()

    def init_serial(self):
        try:
            self.ser = serial.Serial(
                port='/dev/ttyTHS0',
                baudrate=self.baudrate,  
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS,
                timeout=0.1
            )
        except serial.SerialException as e:
            print(f"Failed to initialize serial port: {e}")
            self.error_handler()

    def error_handler(self):
        self.error_count += 1
        if self.error_count >= 3:
            self.ser.close()
            print("Attempting to reset serial port...")
            self.init_serial()

    def gpio_high(self, pin):
        GPIO.output(pin, GPIO.HIGH)
        self.gpio_state = True

    def gpio_low(self, pin):
        GPIO.output(pin, GPIO.LOW)
        self.gpio_state = False

    def send(self, buf=b'\x01', size=0):
        if size == 0:
            size = len(buf)
        try:
            send_time = (size * 8) / self.baudrate
            
            self.gpio_high(11)
            t1 = time.time()
            wLen = self.ser.write(buf)            
            actual_send_time = time.time() - t1
            
            if actual_send_time < send_time:
                time.sleep(send_time - actual_send_time)
            self.gpio_low(11)
                
            return wLen
        except serial.SerialException as e:
            self.error_handler()
            print(f"Error in send method: {e}")
            return 0

    def recv(self, size=12, reRead=3):
        try:
            read = b''
            recound = 0
            while len(read) < size:
                received_data = self.ser.read_all() # Recv SerialPort data
                read += received_data
                              
                recound += 1
                print("serial recound:",recound)
                if recound == reRead: # Recv failed
                    break
                
            return read
        except serial.SerialException as e:
            print(f"Error in recv method: {e}")
            return False

    def echo(self, s_buf, size=0, r_size=12, max_attempts=1):
        sent = self.send(s_buf, size)
        if sent == 0:
            print("Send failed.")
            return b''
        return self.recv(r_size, max_attempts)
