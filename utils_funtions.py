import os
import re
import glob
import time
import ntplib
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone, timedelta
from pathlib import Path


class Write:
    def __init__(self, filename="file.txt"):
        """
        初始化 Write 類別，設定檔案名稱並初始化統計數據。
        """
        self.filename = filename            # 檔名
        self.valid_detections_count = 0     # 當前有效的檢測次數
        self.total_attempts = 0             # 總執行次數
        self.actual_detection_count = 0     # 當前成功檢測到目標的次數
        self.startTime = time.perf_counter()  # 使用高精度計時器
        self.initialize_file()

    def write(self, mode="a", data="\n"):
        """
        通用寫入方法。
        :param mode: 檔案開啟模式，默認為追加 ("a")。
        :param data: 要寫入的數據 (字串)。
        """
        with open(self.filename, mode) as file:
            file.write(str(data))

    def read(self):
        """
        讀取檔案內容。
        :return: 檔案內容 (字串)。
        """
        with open(self.filename, "r") as file:
            return file.read()

    def clear_log(self):
        """
        清空日誌檔案。
        """
        self.write("w", "")

    def initialize_file(self, initial_text="Results Log"):
        """
        初始化檔案，寫入起始資訊。
        :param initial_text: 初始文本，默認為 "Results Log"。
        """
        self.write("w", f"{initial_text}\nStart Date: {datetime.now()}\n\n")

    def prepend_to_file(self, data):
        """
        在檔案最前面加入資料。
        :param data: 要插入的資料 (字串)。
        """
        original_content = self.read()
        new_content = f"{data}\n{original_content}"
        self.write("w", new_content)

    def log_detection_result(self, detected, valid_detections_count, valid_target_data=None, fps=0):
        """
        記錄單次偵測結果，並處理有效檢測數據。
        :param detected: 是否檢測到目標。
        :param valid_detections_count: 當前有效的檢測次數。
        :param valid_target_data: 有效檢測的附加數據 (可選)。
        """
        self.total_attempts += 1
        result_text = f"Attempt {self.total_attempts}: {'Detected' if detected else 'Not Detected'}\n"

        if detected:
            if valid_detections_count >= 4 and valid_target_data:
                details = ', '.join([f"{key}: {value}" for key, value in valid_target_data.items()])
                result_text += f"  Details: {details}\n"
            else:
                result_text += f"  Insufficient effective detections.\n"
        else:
            result_text += "  No effective detection.\n"
            self.actual_detection_count = 0
        result_text = f"{result_text} \n  {fps}FPS"
        
        self.write("a", result_text)
        return result_text

    def update_summary(self):
        """
        更新檔案總結資訊。
        """
        summary = (
            f"\nSummary:\n"
            f"Total Attempts: {self.total_attempts}\n"
            f"End Date: {datetime.now()}\n"
            f"Total running time: {time.perf_counter() - self.startTime:.2f} sec\n"
        )
        self.write("a", summary)


def current_network_time():
    """
    獲取台北時區的當前時間，主要從 NTP 服務器獲取，若無法獲取則使用系統時間。
    """
    try:
        client = ntplib.NTPClient()
        taipei_timezone = timezone(timedelta(hours=8))
        response = client.request('pool.ntp.org')
        dt_utc = datetime.fromtimestamp(response.tx_time, timezone.utc)
        dt_taipei = dt_utc.astimezone(taipei_timezone)
        status = True
    except (ntplib.NTPException, OSError) as e:
        print(f"Unable to obtain network time, using system time instead. Error: {e}")
        taipei_timezone = timezone(timedelta(hours=8))
        dt_taipei = datetime.now(taipei_timezone)
        status = False

    formatted_time = dt_taipei.strftime('%Y_%m%d_%H%M%S')
    return status, formatted_time


def current_path():
    """
    返回當前工作目錄。
    """
    return os.getcwd()


def create_folder(folder_name, base_path="."):
    """
    創建新檔案夾並檢測是否已存在。
    """
    folder_path = os.path.join(base_path, folder_name)
    if os.path.exists(folder_path):
        return f"Folder '{folder_name}' already exists at {folder_path}."
    else:
        os.makedirs(folder_path)
        return f"Folder '{folder_name}' created successfully at {folder_path}."


def isdocker():
    """
    檢測是否在 Docker 容器中運行。
    """
    return Path('/workspace').exists() or Path('/.dockerenv').exists()


def check_file(file):
    """
    搜索檔案，如果未找到，嘗試根據模式匹配。
    """
    if Path(file).is_file() or file == '':
        return file
    else:
        files = glob.glob('./**/' + file, recursive=True)
        assert len(files), f'File Not Found: {file}'
        assert len(files) == 1, f"Multiple files match '{file}', specify exact path: {', '.join(files)}"
        return files[0]


def increment_path(path, exist_ok=True, sep=''):
    """
    增量生成路徑名，例如 runs/exp -> runs/exp1。
    """
    path = Path(path)
    if (path.exists() and exist_ok) or not path.exists():
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")
        matches = [re.search(rf"{re.escape(path.stem)}{sep}(\d+)", d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{sep}{n}"


def check_imshow():
    """
    檢查是否支持顯示圖像功能。
    """
    import cv2
    try:
        assert not isdocker(), 'cv2.imshow() is disabled in Docker environments'
        cv2.imshow('test', np.zeros((1, 1, 3)))
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return True
    except Exception as e:
        print(f'WARNING: Environment does not support cv2.imshow() or PIL Image.show() image displays\n{e}')
        return False
    

def plot_multiple_line_charts(data, title="Multiple Line Chart", xlabel="X-axis", ylabel="Y-axis", save_as=None):
    """
    繪製多條折線圖在一個圖表中。
    """
    plt.figure(figsize=(10, 6))
    for label, (x, y) in data.items():
        plt.plot(x, y, marker='o', linestyle='-', label=label)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_as:
        plt.savefig(save_as)
        print(f"Chart saved as {save_as}")
    else:
        if check_imshow():  # 確認環境是否支持顯示圖像
            plt.show()
        else:
            print("Cannot show line charts")


def hexStr(response):
    if response is not None:
        hex_string = ' '.join(f'{byte:02x}' for byte in response)
        return hex_string
    return None