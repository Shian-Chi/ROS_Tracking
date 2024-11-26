# -*- coding: utf-8 -*-

import gi
import sys
import os
from datetime import datetime, timezone, timedelta
import signal  # 導入 signal 模組
import ntplib

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

timeout = 30  # 秒

# 創建 NTP 客戶端，嘗試從 NTP 服務器獲取時間
try:
    client = ntplib.NTPClient()
    taipei_timezone = timezone(timedelta(hours=8))
    response = client.request('pool.ntp.org')
    dt_utc = datetime.fromtimestamp(response.tx_time, timezone.utc)
    dt_taipei = dt_utc.astimezone(taipei_timezone)
except (ntplib.NTPException, OSError) as e:
    print("Unable to obtain network time, use system time instead")
    taipei_timezone = timezone(timedelta(hours=8))
    dt_taipei = datetime.now(taipei_timezone)

# 格式化時間（格式：年月日時分秒）
formatted_time = dt_taipei.strftime('%Y_%m%d_%H%M%S')

# 初始化GStreamer
Gst.init(None)

# 確保輸出目錄存在
current_path = os.getcwd()
# output_directory = "/home/ubuntu/CSI_Camera_H265/recordings"
output_directory = f"{current_path}/recordings"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# 建立錄制的輸出文件名模板
output_file_template = os.path.join(output_directory, f"recorded_{formatted_time}_%04d.mkv")

# 將超時時間從秒轉換為納秒
max_size_time_ns = int(timeout * 1_000_000_000)  # 秒轉納秒

# 定義GStreamer的管線，使用splitmuxsink來自動分割文件
pipeline = Gst.parse_launch(
    f"rtspsrc location=rtsp://127.0.0.1:8080/test latency=0 name=source ! "
    f"queue ! rtph265depay ! h265parse ! "
    f"splitmuxsink location={output_file_template} max-size-time={max_size_time_ns}"
)

# 管線狀態變化時的回調函數
def on_state_changed(bus, msg):
    old, new, pending = msg.parse_state_changed()
    if msg.src == pipeline and new == Gst.State.PLAYING:
        print("Recording started...")

# 定義一個結束事件的回調函數
def on_eos(bus, msg):
    print("End-of-stream received, stopping pipeline...")
    # 將管線狀態設置為 NULL
    pipeline.set_state(Gst.State.NULL)
    # 退出主循環
    loop.quit()

# 修改後的錯誤處理回調函數
def on_error(bus, msg):
    error, debug_info = msg.parse_error()
    print(f"Error received from element {msg.src.get_name()}: {error.message}")
    print(f"Debugging information: {debug_info if debug_info else 'None'}")
    
    # 檢查錯誤是否與 RTSP 流有關
    if "source" in msg.src.get_name():
        print("RTSP server failure detected. Stopping recording...")
    else:
        print("An error occurred unrelated to the RTSP stream.")
    
    # 發送 EOS 信號，讓管線有機會正確結束
    pipeline.send_event(Gst.Event.new_eos())

# 設置信號處理函數，以處理 SIGINT 和 SIGTERM
def signal_handler(user_data):
    print('Stopping recording, please wait...')
    # 發送 EOS 信號
    pipeline.send_event(Gst.Event.new_eos())
    return True  # 返回 True，讓 GLib 知道信號已被處理

# 使用 GLib 的信號處理，監聽 SIGINT 和 SIGTERM 信號
GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGINT, signal_handler, None)
GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGTERM, signal_handler, None)

# 啟動管線
ret = pipeline.set_state(Gst.State.PLAYING)
if ret == Gst.StateChangeReturn.FAILURE:
    print("Failed to start pipeline")
    sys.exit(1)
else:
    print("Starting recording...")  # 立即顯示，即使實際上的準備可能還在進行中

# 創建GLib主循環並監聽管線的事件
loop = GLib.MainLoop()
bus = pipeline.get_bus()
bus.add_signal_watch()
bus.connect("message::eos", on_eos)
bus.connect("message::error", on_error)
bus.connect("message::state-changed", on_state_changed)  # 監聽狀態變化事件

try:
    loop.run()
except KeyboardInterrupt:
    pass
finally:
    # 停止管線並進行清理
    pipeline.set_state(Gst.State.NULL)
