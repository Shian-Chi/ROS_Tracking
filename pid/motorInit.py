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
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
        except serial.SerialException as e:
            print(f"Failed to initialize serial port: {e}")
            self.error_handler()

    def handle_io_error(self):
        print("Handling I/O error...")
        self.ser.close()
        time.sleep(1)  # 等待一段時間，防止設備忙
        self.init_serial()  # 重新初始化串口

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
            wLen = self.ser.write(buf[:size])            
            actual_send_time = time.time() - t1
            
            if actual_send_time < send_time:
                time.sleep(send_time - actual_send_time)
            self.gpio_low(11)
                
            return wLen
        except serial.SerialException as e:
            self.handle_io_error()
            print(f"Error in send method: {e}")
            return 0
        except OSError as e:
            print(f"OS error during send: {e}")
            self.handle_io_error()  # 呼叫重新初始化函數
            return 0
        
    def recv(self, size):
        l = self.ser.in_waiting
        if l > 0:
            self.gpio_low(11)  # 設置 RTS 為低電位
            return self.ser.read(size)  # 讀取指定大小的資料
        else:
            return None  # 如果沒有資料可讀，返回空或者其他適當的值

    def reset_buffer(self):
        self.ser.reset_input_buffer()  # Clear input buffer before retrying
        self.ser.reset_output_buffer() # Clear output buffer before retrying
    
    def echo(self, sBuff, sSize:int, rSize=0, redo=1):
        for _ in range(redo):
            self.send(sBuff, sSize)  # Send data
            start_time = time.time()
            
            while True:
                if self.ser.in_waiting >= rSize:
                    data = self.recv(rSize)  # Receive specified size of data
                    if data is not None and len(data) >= rSize:
                        self.reset_buffer()
                        return data  # Return received data
                    
                if time.time() - start_time >= 0.1:
                    break  # Timeout exceeded, retry if redo > 1
            
            self.reset_buffer()
            
        return None  # Return None if all retries fail
