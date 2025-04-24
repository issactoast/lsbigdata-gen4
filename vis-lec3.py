# 그래프 그리기 실습
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('./data/Obesity2.csv')
df.info()

# 상관행렬 계산
corr_matrix = df[['Age', 'Height', 'Weight']].corr()

# 히트맵 그리기
plt.figure(figsize=(6,5))
sns.heatmap(corr_matrix,
            annot=True, # 그래프 글씨
            cmap="coolwarm",
            fmt=".2f", linewidths=0.5)
plt.title("Heatmap of Feature Correlations")
plt.show()


plt.figure(figsize=(6,5))
sns.scatterplot(data=df,
    x='Age', y='Weight')
plt.ylabel("Weight")
plt.xlabel("Age")
plt.title("Scatter Plot of Height vs Weight")
plt.show()

# 히트맵 2

import matplotlib.pyplot as plt
import seaborn as sns
rows, cols = 4, 5
np.random.seed(42)
voltage_data = 220 + np.random.randn(rows, cols) * 5  # 평균 220V, ±5V의 변동
high_voltage_positions = [(1, 3), (2, 4)]
for r, c in high_voltage_positions:
    voltage_data[r, c] += 15  # 특정 위치의 전압 증가
df_voltage = pd.DataFrame(voltage_data, index=[f"Row {i+1}" for i in range(rows)], 
                          columns=[f"Col {j+1}" for j in range(cols)])
plt.figure(figsize=(9, 6))
sns.heatmap(df_voltage, annot=True, cmap="coolwarm", fmt=".1f", linewidths=0.5, linecolor='black', cbar=True)
plt.show()



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 예제 시계열 데이터 생성
dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
values = np.cumsum(np.random.randn(30)) + 50
df_timeseries = pd.DataFrame({"Date": dates, "Value": values})
plt.figure(figsize=(8,5))
plt.plot(df_timeseries['Date'], 
         df_timeseries['Value'],
         marker='o', linestyle='-')
plt.xlabel("Date")
plt.ylabel("Value")
plt.title("Time Series Line Graph")
plt.xticks(rotation=45)
plt.show()


from statsmodels.graphics.mosaicplot import mosaic
import matplotlib.pyplot as plt

# 모자이크 그래프 그리기
plt.figure(figsize=(8,5))
mosaic(df, ['Gender', 'NObeyesdad'],
       title="Mosaic Plot of Gender vs Obesity Level")
plt.show()


