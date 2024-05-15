import matplotlib.pyplot as plt
import pandas as pd

# 讀取數據
data = pd.read_csv('./track_time.txt', sep=' ', header=None)
data.columns = ['總時長', '每次紀錄的時間', 'x誤差pixel', 'y誤差pixel', '誤差距離pixel']

# 首先打印列名，確認是否正確
print(data.columns)

# 計算累積時間
data['累積時間'] = data['每次紀錄的時間'].cumsum()

# 繪製折線圖
plt.figure(figsize=(12, 8))

# 繪製 X 誤差
plt.plot(data['累積時間'], data['x誤差pixel'], label='X 誤差 (pixel)', marker='o')

# 繪製 Y 誤差
plt.plot(data['累積時間'], data['y誤差pixel'], label='Y 誤差 (pixel)', marker='x')

# 確認 '誤差距離pixel' 是否是正確的列名
if '誤差距離pixel' in data.columns:
    plt.plot(data['累積時間'], data['誤巫距離pixel'], label='誤差距離 (pixel)', marker='s')
else:
    print("錯誤：DataFrame 中沒有名為 '誤差距離pixel' 的列。")

plt.xlabel('累積時間 (s)')
plt.ylabel('誤差 (pixel)')
plt.title('誤差分析')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
