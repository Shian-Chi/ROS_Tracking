import time

class timer():
    def __init__(self) -> None:
        self.t1 = 0.0
        self.t2 = 0.0
        self.record = []
    
    def start_timer(self):
        self.t1 = time.time()
        self.record.append(self.t1)
        
    def stop_timer(self):
        self.t2 = time.time()
        self.record.append(self.t2)
        return self.t2 - self.t1
    
    def record_time(self):
        self.record.append(time.time())